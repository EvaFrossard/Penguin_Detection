from pathlib import Path
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

########################################################################################################################
# READ BRAINVISION FILES
########################################################################################################################

def _header_field(header: str, key: str) -> str:
    for line in header.splitlines():
        if line.startswith(key + "="):
            return line.split("=", 1)[1]
    raise KeyError(key)


def load_channel(vhdr_path: Path, channel: str):
    """Read one channel from a BrainVision recording. Returns (times_s, values, fs_hz)."""
    header = vhdr_path.read_text(encoding="utf-8", errors="ignore")
    n_channels = int(_header_field(header, "NumberOfChannels"))
    fs = 1e6 / float(_header_field(header, "SamplingInterval"))

    names, resolutions = [], []
    for line in header.splitlines():
        if line.startswith("Ch") and "=" in line and line[2].isdigit():
            _, rest = line.split("=", 1)
            parts = rest.split(",")
            names.append(parts[0])
            resolutions.append(float(parts[2]) if len(parts) > 2 and parts[2] else 1.0)
    idx = names.index(channel)

    raw = np.memmap(vhdr_path.with_suffix(".eeg"), dtype="float32", mode="r")
    raw = raw.reshape(-1, n_channels)  # multiplexed: one row per sample, one column per channel
    values = raw[:, idx].astype(np.float64) * resolutions[idx]
    times = np.arange(len(values)) / fs
    return times, values, fs


def load_session_starts(vhdr_path: Path, fs: float, min_gap_s: float = 5.0):
    """Session-start marker onsets (seconds, same absolute time reference as load_channel's `t`), read directly from the 
    .vmrk file. """
    vmrk_path = Path(vhdr_path).with_suffix(".vmrk")
    starts = []
    for line in vmrk_path.read_text(encoding="utf-8", errors="ignore").splitlines(): # i.e. "Mk2=Stimulus,S  1,654968,1,0"
        if not line.startswith("Mk") or "=" not in line:
            continue
        _, rest = line.split("=", 1) # rest = Stimulus,S  1,654968,1,0
        parts = rest.split(",")
        if len(parts) < 3 or parts[0] != "Stimulus" or parts[1] != "S  1":
            continue
        starts.append(int(parts[2]) / fs) # Append the sample number converted to seconds, parts[2]="654968"
    starts.sort()

    kept = []
    for s in starts:
        if not kept or s - kept[-1] >= min_gap_s: # remove close duplicates within min_gap_s seconds
            kept.append(s)
    return kept


# Event codes
PROBE_START_CODES = {2, 3, 4, 12, 13, 14, 7}  # BEHAV / SCHED / MANUAL / resp / reset / safety probe
PROBE_END_CODE = {5, 8}                          # end probe
SESSION_START_CODE = 1                      # START
SESSION_END_CODE = 6                        # END session


def load_events(vhdr_path: Path, fs: float):
    """Read all "Stimulus,S <n>" markers from the .vmrk file. Returns a list of (code, time_s)
    sorted by time, same absolute time reference as load_channel's `t`."""
    vmrk_path = Path(vhdr_path).with_suffix(".vmrk")
    events = []
    for line in vmrk_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("Mk") or "=" not in line:
            continue
        _, rest = line.split("=", 1)
        parts = rest.split(",")
        if len(parts) < 3 or parts[0] != "Stimulus":
            continue
        code = parts[1].strip()
        if not code.startswith("S"):
            continue
        try:
            num = int(code[1:].strip())
        except ValueError:
            continue
        events.append((num, int(parts[2]) / fs))
    events.sort(key=lambda e: e[1])
    return events


def load_exclusion_intervals(vhdr_path: Path, fs: float):
    """Time intervals (seconds) during which press detection should be suppressed: probes (from
    any probe-start marker to the next end-probe marker) and inter-session gaps (from a
    session-end marker to the next session-start marker). Read directly from the .vmrk file.

    A probe never legitimately spans a session boundary. Recordings open with a rapid calibration
    burst that fires every marker code 1-14 once to check the triggers are wired correctly; this
    can leave a probe-start marker with no matching end-probe marker anywhere nearby (its S5 was
    already consumed earlier in the same burst), which would otherwise pair up with some end-probe
    marker far into the next real session and wrongly swallow it. So an open probe is dropped, not
    carried across, the moment a session-start or session-end marker is seen.
    """
    events = load_events(vhdr_path, fs)
    intervals = []

    open_probe = None
    open_gap = None
    for code, ts in events:
        if code == SESSION_END_CODE:
            open_probe = None
            open_gap = ts
        elif code == SESSION_START_CODE:
            if open_gap is not None:
                intervals.append((open_gap, ts))
                open_gap = None
            open_probe = None
        elif code in PROBE_START_CODES:
            if open_probe is None:
                open_probe = ts
        elif code in PROBE_END_CODE and open_probe is not None:
            intervals.append((open_probe, ts))
            open_probe = None

    return sorted(intervals)


########################################################################################################################
# DETECTION LOGIC
########################################################################################################################

def _rolling_baseline_1d(x: np.ndarray, win: int):  
    """ Computes median and MAD over a trailing window of win samples. The MAD is approximated as a rolling median of 
    |x - rolling median|. """
    s = pd.Series(x)
    med = s.rolling(win, min_periods=1).median()
    mad = (s - med).abs().rolling(win, min_periods=1).median()
    sigma = 1.4826 * mad 
    return med.to_numpy(), sigma.to_numpy()


def rolling_baseline(x: np.ndarray, fs: float, window_s: float,
                      t: np.ndarray = None, session_starts=None):
    """ 
    Rolling baseline with an expanding start -> no NanNs at the beginning of the signal.

    Parameters
    ----------
    x : np.ndarray
        The input signal for which the rolling baseline is to be computed.
    fs : float
        The sampling frequency of the input signal.
    window_s : float
        The size of the rolling window in seconds.
    t : np.ndarray, optional
        The time array corresponding to the input signal. If provided, it is used to determine session
        boundaries for resetting the rolling baseline.
    session_starts : list, optional
        A list of session start times. If provided, the rolling baseline is reset at each session start.
    
    Returns
    -------
    med : np.ndarray
        The rolling median of the input signal.
    sigma : np.ndarray
        The rolling MAD (Median Absolute Deviation) of the input signal, scaled to approximate standard deviation.
    """
    win = max(1, int(window_s * fs))
    if not session_starts or t is None:
        return _rolling_baseline_1d(x, win)

    bounds = sorted({0, len(x), *np.searchsorted(t, sorted(session_starts))}) # Find the boundaries of each session by 
    # searching for the session start times in the time array `t`. The `searchsorted` function returns the indices where
    # the session start times would be inserted to maintain order.
    med, sigma = np.empty(len(x)), np.empty(len(x))
    for a, b in zip(bounds[:-1], bounds[1:]):
        med[a:b], sigma[a:b] = _rolling_baseline_1d(x[a:b], win)
    return med, sigma


@dataclass(frozen=True)
class DetectorParams:
    """Tuning parameters of the press detector (see detect_presses for their meaning)."""
    k_high: float = 5.0
    k_low: float = 2.5
    min_duration_s: float = 0.08
    max_duration_s: float = 2.5
    refractory_s: float = 0.15
    floor_max: float = 200.0
    floor_min: float = 20.0
    quiet_s: float = 60.0
    decay_rate: float = 0.9
    rise_frac: float = 0.3
    rise_alpha: float = 0.1


class DetectorState:
    """Everything the detector remembers from one sample to the next, for one channel."""
    __slots__ = ("state", "onset_t", "peak", "refractory_until",
                 "floor", "applied_floor", "last_signal_t", "last_decay_t")

    def __init__(self, params: DetectorParams, t0: float):
        self.state = "REST"
        self.onset_t = self.peak = None      # onset time and peak amplitude of the current candidate press
        self.refractory_until = -np.inf
        self.floor = params.floor_max        # adaptive floor
        self.applied_floor = self.floor      # floor used on the last sample (before any rise at that sample)
        self.last_signal_t = t0              # last confirmed press, or last reset
        self.last_decay_t = t0


def _step_detector(st: DetectorState, t_i, x_i, med_i, sigma_i, p: DetectorParams,
                   new_session=False, excluded=False):
    """Advance the detector by one sample. Used by both detect_presses (offline, over a whole recording) and
    live_detection.py (live, one sample at a time), so the two give identical results on the same samples.

    `new_session`: this is the first sample of a new session -> reset the adaptive floor.
    `excluded`: this sample is inside a probe or an inter-session gap -> no press can be detected.

    Updates `st` in place and returns (st, press): `press` is None, or a dict {"onset_t", "offset_t", "peak_amp",
    "rejected"} when a press ends at this sample. Rejected (too long) presses are returned too.
    """
    if new_session: # new session -> reset
        st.floor = p.floor_max
        st.last_signal_t = st.last_decay_t = t_i

    if t_i - max(st.last_signal_t, st.last_decay_t) >= p.quiet_s: # no confirmed press for quiet_s seconds -> decay
        st.floor = max(med_i + p.floor_min, st.floor * p.decay_rate)
        st.last_decay_t = t_i

    st.applied_floor = st.floor

    if excluded: # inside a probe or an inter-session gap -> no detection possible
        st.state = "REST"  # silently drops any in-progress candidate/active press
        return st, None

    # Calculate the high and low thresholds for the current sample
    hi = min(max(med_i + p.k_high * sigma_i, med_i + st.floor), med_i + 400.0)
    lo = min(med_i + p.k_low * sigma_i, med_i + 100)
    if math.isnan(hi):
        return st, None

    press = None
    if st.state == "REST":
        if t_i >= st.refractory_until and x_i >= hi:
            st.state, st.onset_t, st.peak = "CANDIDATE", t_i, x_i

    elif st.state == "CANDIDATE":
        st.peak = max(st.peak, x_i)
        if x_i < lo:
            st.state = "REST"
        elif t_i - st.onset_t >= p.min_duration_s:
            st.state = "ACTIVE"

    elif st.state == "ACTIVE":
        st.peak = max(st.peak, x_i)
        too_long = (t_i - st.onset_t) > p.max_duration_s
        if x_i < lo or too_long:
            press = {"onset_t": st.onset_t, "offset_t": t_i, "peak_amp": st.peak, "rejected": too_long}
            if not too_long: # confirmed press -> update adaptive floor
                st.floor = min(max(p.rise_alpha * st.floor + (1 - p.rise_alpha) * (p.rise_frac * st.peak),
                    p.floor_min), p.floor_max) # smoothly rise toward rise_frac * peak_amp, but stay within [floor_min, floor_max]
                st.last_signal_t = t_i
            st.state, st.refractory_until = "REFRACTORY", t_i + p.refractory_s

    elif st.state == "REFRACTORY":
        if t_i >= st.refractory_until:
            st.state = "REST"

    return st, press


def detect_presses(t, x, med, sigma, k_high=5.0, k_low=2.5, min_duration_s=0.08, max_duration_s=2.5, refractory_s=0.15,
                   session_starts=None, floor_max=200.0, floor_min=20.0, quiet_s=60.0, decay_rate=0.9, rise_frac=0.3,
                   rise_alpha=0.1, exclusion_intervals=None):
    """Double-threshold press detector, thresholds derived from the rolling baseline: onset when x crosses above
    `high` = med + max(k_high*sigma, floor), offset when it drops back below `low` = med + k_low*sigma. A press must 
    stay above `low` for at least `min_duration_s` to count (rejects noise blips), and is flagged rejected if it runs 
    past `max_duration_s` (drift/repositioning, not a squeeze). A refractory period stops one press from re-triggering 
    on its own trailing edge.

    `floor` is a bidirectional adaptive floor: it starts at `floor_max` and resets at every onset in `session_starts`. 
    If `quiet_s` seconds pass with no CONFIRMED (accepted) press, it decays geometrically (factor `decay_rate` per 
    `quiet_s` window) toward `floor_min`. It rises back toward `rise_frac * peak_amp` (smoothed by `rise_alpha`) 
    whenever a press IS confirmed.

    Parameters
    ----------
    t : np.ndarray
        Time array corresponding to the input signal.
    x : np.ndarray
        Input signal array.
    med : np.ndarray
        Rolling median of the input signal.
    sigma : np.ndarray
        Rolling MAD (Median Absolute Deviation) of the input signal, scaled to approximate standard deviation.
    k_high : float, optional
        Multiplier for the high threshold.
    k_low : float, optional
        Multiplier for the low threshold.
    min_duration_s : float, optional
        Minimum duration (in seconds) for a press to be considered valid.
    max_duration_s : float, optional
        Maximum duration (in seconds) for a press to be considered valid.
    refractory_s : float, optional
        Refractory period (in seconds) after a press is detected.
    session_starts : list, optional
        List of session start times. The adaptive floor resets at each session start.
    floor_max : float, optional
        Maximum value for the adaptive floor.
    floor_min : float, optional
        Minimum value for the adaptive floor.
    quiet_s : float, optional
        Time (in seconds) of inactivity after which the adaptive floor starts to decay.
    decay_rate : float, optional
        Geometric decay rate for the adaptive floor.
    rise_frac : float, optional
        Fraction of the peak amplitude to which the adaptive floor rises after a confirmed press.
    rise_alpha : float, optional
        Smoothing factor for the rise of the adaptive floor after a confirmed press.
    exclusion_intervals : list of (float, float), optional
        Time windows (seconds), e.g. from `load_exclusion_intervals`, during which no press can be
        detected -- typically probes and inter-session gaps. Any candidate/active press is
        discarded the instant it enters such a window, and no new onset can trigger inside one.

    Returns
    -------
    presses : list of dict
        Each dict contains the following keys:  "onset_t", "offset_t", "peak_amp", "rejected"
    floor : np.ndarray
        The per-sample adaptive-floor array actually applied (same length as x), used later for plotting.
    """
    n = len(x)
    starts = sorted(session_starts) if session_starts else []
    sess_idx = np.searchsorted(starts, t, side="right")

    excluded = np.zeros(n, dtype=bool)
    for a, b in (exclusion_intervals or []):
        ia, ib = np.searchsorted(t, [a, b])
        excluded[ia:ib] = True

    params = DetectorParams(k_high, k_low, min_duration_s, max_duration_s, refractory_s,
                            floor_max, floor_min, quiet_s, decay_rate, rise_frac, rise_alpha)
    st = DetectorState(params, t[0] if n else 0.0)
    floor = np.empty(n)
    prev_sess = sess_idx[0] if n else 0
    presses = []

    # Plain Python lists: much faster to index one element at a time than numpy arrays (same float64 values)
    t_l, x_l, med_l, sigma_l = t.tolist(), x.tolist(), med.tolist(), sigma.tolist()
    sess_l, excl_l = sess_idx.tolist(), excluded.tolist()

    for i in range(n):
        new_session = sess_l[i] != prev_sess
        prev_sess = sess_l[i]
        _, press = _step_detector(st, t_l[i], x_l[i], med_l[i], sigma_l[i], params,
                                  new_session=new_session, excluded=excl_l[i])
        floor[i] = st.applied_floor
        if press is not None:
            presses.append(press)

    return presses, floor


########################################################################################################################
# EVENT TABLE EXPORT
########################################################################################################################

def label_sessions(onsets, session_starts):
    """Map each onset time (seconds) to 'pre-session' or 'Session N' (1-based, N = number of
    session starts at or before that onset) -- the same convention used for the 'session' column
    in Scoring_Pingouin_Claude_V4.py's manual export (see get_session())."""
    starts = sorted(session_starts) if session_starts else []
    idx = np.searchsorted(starts, onsets, side="right")
    return [f"Session {i}" if i > 0 else "pre-session" for i in idx]


def merge_close_events(df: pd.DataFrame, gap_s: float = 4.0,
                        onset_col: str = "onset_absolute", offset_col: str = "offset_absolute") -> pd.DataFrame:
    """Collapse events whose onset is within `gap_s` seconds of the previous one into a single
    event spanning from the first onset to the last offset (multiple quick re-triggers of the same
    physical squeeze/release). Other columns are kept from the first row of each group, except
    `peak_amp` (max) and `rejected` (any) when present."""
    if df.empty:
        return df.reset_index(drop=True)
    df = df.sort_values(onset_col).reset_index(drop=True)
    group = (df[onset_col].diff() > gap_s).cumsum()
    agg = {c: "first" for c in df.columns if c not in (onset_col, offset_col)}
    if "peak_amp" in agg:
        agg["peak_amp"] = "max"
    if "rejected" in agg:
        agg["rejected"] = "any"
    agg[onset_col] = "first"
    agg[offset_col] = "last"
    return df.groupby(group).agg(agg).reset_index(drop=True)[df.columns]


########################################################################################################################
# PLOTTING FUNCTIONS
########################################################################################################################

def plot_channel(Subj, Channel, x, t, med, sigma,t0, duration, ymin, ymax, high_thresh, low_thresh, show_signal=False,
                show_baseline=False, show_thresholds=False, show_presses_auto=False, presses=None, show_presses_manual=False,
                manual_presses=None, floor=None):
    mask = (t >= t0) & (t <= t0 + duration)
    # Create an array of the same length as `med` with the adaptive floor values
    floor_arr = floor if floor is not None else np.full_like(med, 200.0)
    high = np.minimum(np.maximum(med + high_thresh * sigma, med + floor_arr), med + 400.0)
    low = np.minimum(med + low_thresh * sigma, med + 100)

    fig, ax = plt.subplots(figsize=(14, 5))
    if show_signal:
        ax.plot(t[mask], x[mask], lw=1, color="steelblue", label=f"{Channel} (raw)")
    if show_baseline:
        ax.plot(t[mask], med[mask], lw=1.2, color="black", label="baseline (median)")
    if show_thresholds:
        ax.plot(t[mask], high[mask], lw=1, ls="--", color="firebrick", label="high threshold")
        ax.plot(t[mask], low[mask], lw=1, ls="--", color="darkorange", label="low threshold")
    if show_presses_auto:
        for p in presses:
            if p["onset_t"] < t0 or p["offset_t"] > t0 + duration:
                continue
            color = "green" if not p["rejected"] else "gray"
            ax.axvspan(p["onset_t"], p["offset_t"], color=color, alpha=0.35)
    if show_presses_manual and manual_presses is not None:
        for _, p in manual_presses.iterrows():
            if p["offset_absolute"] < t0 or p["onset_absolute"] > t0 + duration:
                continue
            ax.axvspan(p["onset_absolute"], p["offset_absolute"], color="mediumpurple", alpha=0.35)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("force")
    ax.set_ylim(ymin, ymax)
    ax.set_title(f"{Subj} — {Channel}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()