"""Live squeeze detection for the TickToz experiment.

Reads the force channels from BrainVision Recorder over RDA, runs the same detector as detection.ipynb one sample at
a time, and signals every detected press to script_TickToz.m by incrementing a counter in a flag file:
    <flag-dir>/resp_flag.txt   (Response channel)
    <flag-dir>/reset_flag.txt  (Reset channel)
MATLAB polls these files and treats each increment as a button press. Every press (and every trigger marker) is also
written to timestamped CSV logs in <log-dir>, independent of what MATLAB does with it.

Matching the offline analysis (detection.ipynb, "Create csv" section):
  - baseline: 10 min trailing median / MAD, restarted at every session start (S1)      -> online_baseline.py
  - detector: helpers_detection._step_detector, the exact code detect_presses runs     -> same parameters below
  - exclusions: probes, pauses and inter-session gaps, built live from the trigger markers with the same rules as
    load_exclusion_intervals
  - presses whose onset is < MERGE_GAP_S after the previous confirmed press on the same channel are merged into it
    (merge_close_events): they are logged but do not bump the flag again
Results are identical to the offline ones on the same samples, from the first session start after this script is
connected. Start it before the first session starts (e.g. before the calibration triggers).

Usage (on the recording PC, with RDA enabled in Recorder):
    python live_detection.py --subject TT013
    python live_detection.py --resp-channel "Left Thumb" --reset-channel "Right Thumb"
"""
import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from helpers_detection import (DetectorParams, DetectorState, _step_detector, PROBE_START_CODES, PROBE_END_CODE,
                               SESSION_START_CODE, SESSION_END_CODE)
from online_baseline import RollingBaseline
from rda_client import RDAClient, StartInfo, DataBlock, RDA_PORT_32BIT

########################################################################################################################
# PARAMETERS 
########################################################################################################################

BASELINE_WINDOW_S = 600.0
SESSION_START_MIN_GAP_S = 5.0 # load_session_starts(min_gap_s): repeated S1 closer than this are one session start
MERGE_GAP_S = 5.0 # merge_close_events(gap_s) used for {SUBJECT}_events_auto.csv

RESPONSE_PARAMS = DetectorParams(k_high=28.0, k_low=5, min_duration_s=0.15, max_duration_s=3.5, refractory_s=0,
                                 floor_max=150.0, floor_min=10.0, quiet_s=60.0, decay_rate=0.9,
                                 rise_frac=0.5, rise_alpha=0.7)
# The notebook adds 100 to floor_min for the second channel it processes, which is the Reset channel (`incr`) -> reduces FP
RESET_PARAMS = DetectorParams(**{**RESPONSE_PARAMS.__dict__, "floor_min": RESPONSE_PARAMS.floor_min + 100})

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FLAG_DIR = Path(r"C:\Users\MATLAB.PSL5216024\Documents\TickToz_IloSara\live_detect")  # main_path in MATLAB
DEFAULT_SUBJECTS_CSV = SCRIPT_DIR / "Scoring_Pingouin" / "Subjects.csv"


########################################################################################################################
# BUILDING BLOCKS
########################################################################################################################

class SessionTracker:
    """Live version of load_session_starts + load_exclusion_intervals: follows the trigger markers and says, for
    each sample, whether a new session starts there and whether detection is suppressed.

    Differences from the offline intervals (they cannot matter inside a session): after the last session-end marker,
    detection stays off until the next S1 (offline, a gap that never closes is not excluded), and a probe that is
    abandoned by a session start/end is excluded until then (offline it is dropped entirely -- this only happens in
    the calibration burst, which lies inside a session gap anyway)."""

    def __init__(self):
        self.open_probe = False
        self.open_gap = False
        self.last_start_t = None
        self.session = 0

    @property
    def excluded(self):
        return self.open_probe or self.open_gap

    def on_marker(self, code, t):
        """Returns True if this marker starts a new session (baseline and floor must be reset)."""
        if code == SESSION_END_CODE:
            self.open_probe = False
            self.open_gap = True
        elif code == SESSION_START_CODE:
            self.open_gap = False
            self.open_probe = False
            if self.last_start_t is None or t - self.last_start_t >= SESSION_START_MIN_GAP_S:
                self.last_start_t = t
                self.session += 1
                return True
        elif code in PROBE_START_CODES:
            self.open_probe = True
        elif code in PROBE_END_CODE:
            self.open_probe = False
        return False

    @property
    def label(self):
        return f"Session {self.session}" if self.session else "pre-session"  # same labels as label_sessions


class ChannelDetector:
    def __init__(self, event, channel, params, fs, t0):
        self.event, self.channel, self.params = event, channel, params
        self.baseline = RollingBaseline(fs, BASELINE_WINDOW_S)
        self.state = DetectorState(params, t0)
        self.last_confirmed_onset = None

    def step(self, t, x, new_session, excluded):
        if new_session:
            self.baseline.reset()
        med, sigma = self.baseline.update(x)
        _, press = _step_detector(self.state, t, x, med, sigma, self.params,
                                  new_session=new_session, excluded=excluded)
        return press


class FlagCounter:
    """Monotonic press counter shared with MATLAB through a one-line text file. Continues from the value already in
    the file (MATLAB only reacts to increases). Written atomically; retried if MATLAB is reading it at that moment."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.count = int(float(self.path.read_text().strip() or 0))
        except (OSError, ValueError):
            self.count = 0
        self.pending = True
        self.flush()

    def bump(self):
        self.count += 1
        self.pending = True
        self.flush()

    def flush(self):
        if not self.pending:
            return
        tmp = self.path.with_suffix(".tmp")
        try:
            tmp.write_text(f"{self.count}\n")
            for _ in range(20):
                try:
                    os.replace(tmp, self.path)
                    self.pending = False
                    return
                except PermissionError:  # MATLAB has the file open for reading
                    time.sleep(0.005)
        except OSError as e:
            print(f"  ! could not write {self.path.name}: {e}")
        # still pending: retried on the next data block


class CsvLog:
    def __init__(self, path: Path, columns):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=columns)
        self.w.writeheader()
        self.f.flush()

    def write(self, row):
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()


PRESS_COLUMNS = ["wall_time", "stream_t", "session", "event", "channel", "onset_s", "offset_s", "duration_s",
                 "peak_amp", "rejected", "merged", "flag_count"]
MARKER_COLUMNS = ["wall_time", "stream_t", "code", "type", "description", "session", "excluded_after"]


########################################################################################################################
# MAIN LOOP
########################################################################################################################

def run(host, port, channels, flag_dir, log_dir):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    press_log = CsvLog(log_dir / f"live_presses_{stamp}.csv", PRESS_COLUMNS)
    marker_log = CsvLog(log_dir / f"live_markers_{stamp}.csv", MARKER_COLUMNS)
    flags = {"Response": FlagCounter(flag_dir / "resp_flag.txt"),
             "Reset": FlagCounter(flag_dir / "reset_flag.txt")}
    print(f"Logs  -> {press_log.f.name}\n         {marker_log.f.name}")
    print(f"Flags -> {flag_dir} (resp={flags['Response'].count}, reset={flags['Reset'].count})")

    try:
        while True:
            try:
                with RDAClient(host, port) as client:
                    print(f"Connected to RDA at {host}:{port}, waiting for START...")
                    _stream(client, channels, flags, press_log, marker_log)
            except (ConnectionError, OSError) as e:
                print(f"[{datetime.now():%H:%M:%S}] RDA not available ({e}). Retrying in 2 s...")
                time.sleep(2)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        press_log.close()
        marker_log.close()


def _stream(client, channels, flags, press_log, marker_log):
    detectors, tracker, fs = [], None, None
    col_idx = []
    n_samples, last_block = 0, None

    for msg in client.messages():
        if isinstance(msg, StartInfo):
            missing = [ch for ch in channels.values() if ch not in msg.channel_names]
            if missing:
                raise SystemExit(f"Channel(s) {missing} not in the RDA stream. Available: {msg.channel_names}")
            fs = msg.fs
            detectors = [ChannelDetector(ev, ch, RESET_PARAMS if ev == "Reset" else RESPONSE_PARAMS, fs, 0.0)
                         for ev, ch in channels.items()]
            col_idx = [msg.channel_names.index(d.channel) for d in detectors]
            tracker = SessionTracker()
            n_samples, last_block = 0, None
            print(f"[{datetime.now():%H:%M:%S}] START: {len(msg.channel_names)} channels at {fs:g} Hz. "
                  f"Detecting {', '.join(f'{d.event}={d.channel}' for d in detectors)}")
            continue

        if msg is None:
            print(f"[{datetime.now():%H:%M:%S}] STOP received from Recorder. Waiting for the next START...")
            detectors = []
            continue

        if not detectors:
            continue

        # Dropped blocks: keep the stream clock aligned with the recording
        if last_block is not None and msg.block != last_block + 1:
            lost = msg.block - last_block - 1
            n_samples += max(lost, 0) * len(msg.data)
            print(f"  ! {lost} RDA block(s) dropped (~{max(lost, 0) * len(msg.data) / fs:.2f} s of signal lost)")
        last_block = msg.block

        markers = sorted(msg.markers, key=lambda m: m.position)
        columns = [msg.data[:, c].tolist() for c in col_idx]
        mk = 0
        for j in range(len(msg.data)):
            t = (n_samples + j) / fs
            new_session = False
            while mk < len(markers) and markers[mk].position <= j:
                m = markers[mk]
                mk += 1
                if m.code is None:
                    continue
                if tracker.on_marker(m.code, t):
                    new_session = True
                    print(f"[{datetime.now():%H:%M:%S}] === {tracker.label} starts (t={t:.1f}s) ===")
                marker_log.write({"wall_time": datetime.now().isoformat(timespec="milliseconds"),
                                  "stream_t": round(t, 4), "code": m.code, "type": m.type,
                                  "description": m.description, "session": tracker.label,
                                  "excluded_after": tracker.excluded})

            for d, col in zip(detectors, columns):
                press = d.step(t, col[j], new_session, tracker.excluded)
                if press is not None:
                    _on_press(d, press, t, tracker, flags, press_log)

        n_samples += len(msg.data)
        for f in flags.values():
            f.flush()  # retry any write MATLAB blocked


def _on_press(d, press, t, tracker, flags, press_log):
    merged = False
    if not press["rejected"]:
        merged = d.last_confirmed_onset is not None and press["onset_t"] - d.last_confirmed_onset <= MERGE_GAP_S
        d.last_confirmed_onset = press["onset_t"]
        if not merged:
            flags[d.event].bump()

    press_log.write({"wall_time": datetime.now().isoformat(timespec="milliseconds"), "stream_t": round(t, 4),
                     "session": tracker.label, "event": d.event, "channel": d.channel,
                     "onset_s": round(press["onset_t"], 4), "offset_s": round(press["offset_t"], 4),
                     "duration_s": round(press["offset_t"] - press["onset_t"], 4),
                     "peak_amp": round(press["peak_amp"], 3), "rejected": press["rejected"], "merged": merged,
                     "flag_count": flags[d.event].count})

    what = "rejected (too long)" if press["rejected"] else "merged, no flag" if merged else \
        f"-> {d.event.lower()} flag = {flags[d.event].count}"
    print(f"[{datetime.now():%H:%M:%S}] {tracker.label:<11} {d.event:<8} ({d.channel}) "
          f"onset {press['onset_t']:.2f}s peak {press['peak_amp']:.0f}  {what}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", help="Subject ID, to look up its Response/Reset channels in Subjects.csv")
    ap.add_argument("--resp-channel", help="Response channel name (overrides Subjects.csv)")
    ap.add_argument("--reset-channel", help="Reset channel name (overrides Subjects.csv)")
    ap.add_argument("--subjects-csv", type=Path, default=DEFAULT_SUBJECTS_CSV)
    ap.add_argument("--host", default="localhost", help="PC running BrainVision Recorder")
    ap.add_argument("--port", type=int, default=RDA_PORT_32BIT)
    ap.add_argument("--flag-dir", type=Path, default=DEFAULT_FLAG_DIR, help="must match respFlagPath in MATLAB")
    ap.add_argument("--log-dir", type=Path, help="default: <flag-dir>/logs")
    args = ap.parse_args()

    resp, reset = args.resp_channel, args.reset_channel
    if args.subject and not (resp and reset):
        subjects = pd.read_csv(args.subjects_csv)
        row = subjects.loc[subjects["Subject"] == args.subject]
        if row.empty:
            ap.error(f"{args.subject} not found in {args.subjects_csv}")
        resp = resp or row.iloc[0]["Response"]
        reset = reset or row.iloc[0]["Reset"]
    if not (resp and reset):
        ap.error("give --subject, or both --resp-channel and --reset-channel")

    # Response first, then Reset: the same order as channels_of_interest in the notebook
    channels = {"Response": resp, "Reset": reset}
    run(args.host, args.port, channels, args.flag_dir, args.log_dir or args.flag_dir / "logs")


if __name__ == "__main__":
    main()
