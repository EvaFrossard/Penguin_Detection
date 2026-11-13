# Penguin_Detection
The aim of this project is to detect, in real time, the presses on a "penguin" force sensor (a squeeze sensor held in the hand), so that the responses of the time-production task are recorded automatically during sleep-onset experiments.

The same detector runs in two modes:
- **Offline** ([detection.ipynb](detection.ipynb)): on a full BrainVision recording, to tune the parameters and to export the automatic events for comparison with the manual scoring.
- **Live** ([live_detection.py](live_detection.py)): on the stream sent by BrainVision Recorder during the experiment. Every detected press is passed to the MATLAB task ([script_TickToz_Eva.m](script_TickToz_Eva.m)) as a Response or Reset button press.

Both modes call the same per-sample function (`_step_detector` in [helpers_detection.py](helpers_detection.py)), so they give identical results on the same samples.

## Data
Every subject has five force channels: Left Thumb, Left Index+Middle, Left Ring+Pinky, Right Thumb and Right Ring+Pinky. Two of them are used: the **Response** channel and the **Reset** channel, listed per subject in `Subjects.csv` (columns `Subject`, `Response`, `Reset`).

Offline, a channel is read directly from the raw BrainVision recording (`.vhdr`/`.eeg`/`.vmrk`) with `load_channel`. All times are in seconds from the start of the recording.

The trigger markers sent by the MATLAB task are read from the `.vmrk` file (`load_events`) and used to:
- **find the session starts** (`load_session_starts`): every `S  1` marker. Repeats less than 5 s apart count as one start, because MATLAB sends the start trigger twice. The baseline and the adaptive floor are reset at each one.
- **find the windows where no press can count** (`load_exclusion_intervals`). The task is paused or between sessions there, so a press there is not a response:
  - probes and pauses: from a probe-start or pause marker (codes 2, 3, 4, 7, 12, 13, 14) to the next end-probe or resume marker (5, 8);
  - inter-session gaps: from a session-end marker (6) to the next session start (1).

  A probe that is still open when a session starts or ends is dropped instead of carried over. This handles the trigger test at the beginning of a recording, which fires every code once and can leave a probe-start marker with no matching end.

| Code | Event | Code | Event |
|---|---|---|---|
| 1 | Session start | 8 | Resume |
| 2 | Behavioural probe (90 s without a press) | 9 | Undo / correction |
| 3 | Scheduled final probe | 10 | Response button press |
| 4 | Manual probe | 11 | Reset button press |
| 5 | End of probe | 12 | Probe after a response streak (Nap2) |
| 6 | Session end | 13 | Probe after a reset streak (Nap2) |
| 7 | Pause | 14 | Safety probe, 15 min without a probe (Nap2) |

## Repository layout
```text
Code/Live_detection/
  detection.ipynb               # offline detection: tuning, plots, CSV export
  helpers_detection.py          # BrainVision readers, baseline, detector, export helpers, plots
  live_detection.py             # live detection over RDA, writes the flag files read by MATLAB
  online_baseline.py            # sample-by-sample version of rolling_baseline (used live)
  rda_client.py                 # minimal client for BrainVision Recorder's RDA stream
  script_TickToz_Eva.m          # MATLAB time-production task, reads the flag files
  Scoring_Pingouin_Claude_V4.py # Ilona's GUI to visualise the channels and the events
  penguin_detection.md

<DATA_ROOT>/
  TT001/TT001.vhdr, .eeg, .vmrk # raw recording of each subject
  Scoring_Pingouin/
    Subjects.csv                # Response and Reset channel of each subject
    TT001_events.csv            # events from the manual scoring
    TT001_auto_presses.csv      # automatic detections, all candidates (offline export)
    TT001_events_auto.csv       # automatic events, confirmed and merged (offline export)
    ...
```

## Detection algorithm

### Rolling baseline
`rolling_baseline` (`_rolling_baseline_1d`) estimates the local resting level of the signal over a trailing window of `BASELINE_WINDOW_S` seconds:
- `med`: rolling median of the signal.
- `sigma`: rolling MAD (median of `|x - med|`), scaled by 1.4826 so that it approximates a standard deviation.

The median is used instead of the mean so that the baseline is not pulled up by the amplitude peaks of the presses (see https://bop.unibe.ch/JEMR/article/view/JEMR.12.8.3).

`min_periods=1` gives an expanding window at the start, so there are no NaNs at the beginning of the recording. When `session_starts` is given, the signal is split at every session start and the baseline is computed separately in each segment, so the window never carries statistics from one session into the next.

### Double-threshold detector
`detect_presses` runs a per-sample state machine (`REST` → `CANDIDATE` → `ACTIVE` → `REFRACTORY` → `REST`) with two thresholds built from the baseline. Values are in the units of the recording.
- **High threshold** `hi = min(max(med + k_high*sigma, med + floor), med + 400)`: crossing it from `REST` opens a candidate press. It is never lower than `med + floor` (see below) and never higher than `med + 400`.
- **Low threshold** `lo = min(med + k_low*sigma, med + 100)`: dropping back below it ends the press.

A candidate must stay above `lo` for at least `min_duration_s` to become `ACTIVE`, which rejects short noise blips. An active press ends when the signal drops below `lo`. If it lasts longer than `max_duration_s`, it is ended and flagged `rejected` (drift or repositioning of the hand rather than a squeeze).

A press is only reported **at its offset**, once the signal has dropped below `lo`.

A `refractory_s` window after every press stops its own trailing edge from triggering a new one. It was first added against noise, but became unnecessary once the thresholds were adaptive, so it is set to 0. The parameter is kept in case it is needed again.

### Adaptive floor
`floor` sets the minimum height of the high threshold above the baseline. It keeps `hi` from coming too close to the baseline when the signal is very quiet (`sigma` close to 0). It is updated at every sample:
- It starts at `floor_max` and goes back there at every session start.
- If `quiet_s` seconds pass without a *confirmed* (accepted, not rejected) press, it is multiplied by `decay_rate`, and again after every further `quiet_s` of silence. It never goes below `med + floor_min`.
- Every confirmed press pulls it toward `rise_frac * peak_amp`: `floor = rise_alpha * floor + (1 - rise_alpha) * rise_frac * peak_amp`, kept within `[floor_min, floor_max]`.

So the threshold drops when the participant stops pressing (weaker presses when falling asleep are still caught) and adapts to the strength of their recent presses.


## Parameters
The offline values are notebook globals in [detection.ipynb](detection.ipynb). The live values are `RESPONSE_PARAMS` / `RESET_PARAMS` at the top of [live_detection.py](live_detection.py). Both use the same values:

| Parameter | Meaning | Value |
|---|---|---|
| `BASELINE_WINDOW_S` | Trailing window (s) of the rolling baseline | 600 |
| `K_HIGH` | Multiplier of `sigma` for the high (onset) threshold, high to limit false positives | 28 |
| `K_LOW` | Multiplier of `sigma` for the low (offset) threshold, low because the force drops below the baseline after most presses | 5 |
| `MIN_DURATION` | Minimum time (s) above the low threshold for a candidate to be confirmed | 0.15 |
| `MAX_DURATION` | Maximum time (s) above the low threshold before a press is flagged rejected | 3.5 |
| `REFRACTORY` | Time (s) after a press during which no new one can start | 0 |
| `FLOOR_MAX` | Starting and maximum value of the adaptive floor, restored at every session start | 150 |
| `FLOOR_MIN` | Minimum value of the adaptive floor | 10 (Response), 110 (Reset) |
| `QUIET_S` | Time (s) without a confirmed press before the floor decays | 60 |
| `DECAY_RATE` | Factor applied to the floor every `QUIET_S` of silence | 0.9 |
| `RISE_FRAC` | Fraction of `peak_amp` the floor moves toward after a confirmed press | 0.5 |
| `RISE_ALPHA` | Smoothing of that rise (closer to 1 = slower) | 0.7 |
| merge gap | Presses whose onset is at most this many seconds after the previous one are merged | 5 |

The Reset channel uses `FLOOR_MIN + 100` to reduce false positives: in the notebook this is the `incr` added to the second channel of `channels_of_interest`.

## Offline workflow ([detection.ipynb](detection.ipynb))
1. Set `DATA_ROOT`, `SUBJECT`, `CHANNEL` and the parameters.
2. Load the channel, the session starts and the exclusion windows, then compute the baseline.
3. Run `detect_presses` and check the result with `plot_channel`. It draws the signal, baseline, thresholds and automatic presses (green = accepted, grey = rejected), and optionally the manual presses.
4. "Create csv": run the detector on both channels of interest of `SUBJECT` (from `Subjects.csv`) and write the outputs below.
5. Remove the events detected before the first session start (`pre-session`) from `{SUBJECT}_events_auto.csv`.

### Outputs
Both CSVs are written to `<DATA_ROOT>/Scoring_Pingouin/`, where [Scoring_Pingouin_Claude_V4.py](Scoring_Pingouin_Claude_V4.py) reads them:

- **`{SUBJECT}_auto_presses.csv`**: one row per candidate press (confirmed and rejected) on both channels. Columns `channel`, `onset_absolute`, `offset_absolute`, `peak_amp`, `rejected`. Shown as the pink "AUTO" overlay in the GUI (key `d` / button `[3] AUTO`).
- **`{SUBJECT}_events_auto.csv`**: confirmed presses only. Presses whose onset is at most 5 s after the previous one are merged by `merge_close_events` into one event, from the first onset to the last offset (repeated triggers of one squeeze). Columns `subject`, `session` (`Session N` as in the manual export, from `label_sessions`), `event` (`Response`/`Reset`), `channel`, `onset_s`, `offset_s`, `duration_s`. Same format as the manual `{SUBJECT}_events.csv`, so the two can be compared.

## Live detection
```text
BrainVision Recorder --RDA (TCP 51244)--> live_detection.py --resp_flag.txt / reset_flag.txt--> script_TickToz_Eva.m
        ^                                                                                              |
        +----------------------------- trigger box (COM3): S10 / S11 for every press ------------------+
```

[live_detection.py](live_detection.py) runs on the recording PC next to Recorder and MATLAB:

1. **Reading the stream** ([rda_client.py](rda_client.py)): it connects to Recorder's RDA server (32-bit float stream, port 51244). It receives the channel list and sampling rate (START), then blocks of samples with the markers that fall inside them (DATA). 
2. **Following the session** (`SessionTracker`): it applies the rules of `load_session_starts` and `load_exclusion_intervals` to the markers as they arrive. For each sample it knows whether a new session starts there and whether detection is suppressed.
3. **Detecting** (`ChannelDetector`, one per channel):
   - `RollingBaseline` ([online_baseline.py](online_baseline.py)) computes the same rolling median and MAD as `rolling_baseline`, one sample at a time.
   - `_step_detector` is then called with the same parameters as offline.
4. **Merging**: a confirmed press whose onset is at most 5 s after the previous confirmed press on the same channel is merged into it. It is logged but not sent again. This matches `merge_close_events` in `{SUBJECT}_events_auto.csv`.
5. **Sending to MATLAB** (`FlagCounter`): each new press increments a counter in `<flag-dir>/resp_flag.txt` or `<flag-dir>/reset_flag.txt`. The file is written atomically (temporary file + rename), and the write is retried if MATLAB has the file open. The counter carries on from the value already in the file, because MATLAB only reacts to increases.
6. **Logging**: every press (including rejected and merged ones) and every marker is written to `<log-dir>/live_presses_<date_time>.csv` and `live_markers_<date_time>.csv`, whatever MATLAB does with them.

On the MATLAB side, the monitoring loop reads both flag files about every 50 ms. Every increase counts as one Response or Reset button press, exactly like the `R`/`E` keys: it updates the streak counters, resets the 90 s no-response timer, and sends trigger 10 or 11 to the EEG. Counts that arrive before a session starts, during a pause, or during a probe are ignored.

### Requirements
Python ≥ 3.7 (the code uses `dataclasses`, added in 3.7) with `numpy`, `pandas` and `matplotlib` installed under that same interpreter. The recording PC may have an older Python registered as the default `python` — check with `python --version` first; if it's below 3.7, use `py -3.11` (or whichever 3.7+ version is installed) explicitly, and install the three packages under that interpreter too.

### Usage
```text
py -3.11 live_detection.py --subject TT013                                    # channels from Subjects.csv
py -3.11 live_detection.py --resp-channel "Left Thumb" --reset-channel "Right Thumb"
```
| Option | Default |
|---|---|
| `--subject` | Looks up the Response/Reset channels in `--subjects-csv` (default `Scoring_Pingouin/Subjects.csv` next to the script) |
| `--resp-channel`, `--reset-channel` | Override the channels from `Subjects.csv` |
| `--host`, `--port` | `localhost`, `51244` (PC running Recorder) |
| `--flag-dir` | `C:\Users\MATLAB.PSL5216024\Documents\TickToz_IloSara\live_detect` (must match `respFlagPath`/`resetFlagPath` in MATLAB) |
| `--log-dir` | `<flag-dir>/logs` |

Before the experiment:
1. Enable RDA in Recorder (Configuration > Preferences > Remote Data Access).
2. Start `live_detection.py` **before the first session starts**, ideally before the trigger test. It must see every session start to reset the baseline at the same samples as the offline analysis.
3. Then start the MATLAB task. `Ctrl+C` stops the script.

### Live vs offline
From the first session start after the script connects, the live presses are identical to the offline ones computed on the same recording. The only differences are outside the sessions, where nothing is counted anyway:
- After the last session-end marker, live detection stays off until the next session start. Offline, a gap that is never closed is not excluded.
- A probe left open by a session start or end stays excluded live until then. Offline it is dropped. This only happens during the trigger test, which is already between sessions.

### Latency
A press is only known once it has ended (the signal is back below `lo`) and has lasted at least `MIN_DURATION`. MATLAB therefore registers it about one press duration after its onset, plus up to ~50 ms of polling and the RDA block length. MATLAB's own inter-response times, and its triggers 10/11, use this registration time. The true onsets are in `live_presses_*.csv` (`onset_s`) and can be recomputed offline.

### Order of operations
1. Enable RDA in BrainVision Recorder (Configurations -> Pewferences -> Remote Data Access) and start monitoring
2. Start `live_detection.py` with the command above. The terminal should confirm RDA connection and show the response and reset channels.
3. Start `script_TickToz_Eva.m` and run the trigger test or the session as usual.
