# Penguin_Detection
The aim of this project is to create a real-time detector of the presses of a "penguin" muscular tonus sensor, to allow the automatic detection of responses during sleep-onset experiments.


Detection runs on one EMG force channel at a time (e.g. "Right Thumb", "Left Thumb"), loaded directly from the raw BrainVision recording (`.vhdr`/`.eeg`/`.vmrk`) via `load_channel` in [helpers_detection.py](helpers_detection.py). Every subject has the recordings of five channels: Left Thumb, Left Index+Middle, Left Ring+Pinky, Right Thumb, Right Ring+Pinky. Session-start markers and probe/inter-session gaps are read from the `.vmrk` file (`load_session_starts`, `load_exclusion_intervals`) and used to reset the baseline per session and suppress detection where a press can't legitimately occur.

## Repository layout
```text
Scoring_Pingouin/ 
  Subjects.csv # Contains a list of the subjects and their channels of interest
  TT001_events.csv # List of the events obtained by manual detection
  TT001_events_auto.csv # List of the events obtained by automatic detection
  ...
.gitignore
README.md
Scoring_Pingouin_Claude_V4.py # Ilona's GUI to visualize the channels and events
detection.ipynb
helpers_detection.py
```

## Detection Algorithm

### Rolling Baseline
`rolling_baseline` (`_rolling_baseline_1d`) estimates the signal's local resting level with a trailing rolling window of `BASELINE_WINDOW_S` seconds:
- `med`: rolling median of the signal.
- `sigma`: rolling MAD (median of `|x - med|`) scaled by 1.4826 to approximate a standard deviation.
We use the median instead of the meaan so that the value is not affected by the amplitude peaks during the responses. https://bop.unibe.ch/JEMR/article/view/JEMR.12.8.3

`min_periods=1` gives an expanding window at the very start of the recording, so there are no NaNs at the beginning. When `session_starts` is provided, the signal is split at each session boundary and the baseline is computed independently within each segment, so the rolling window never carries statistics across sessions.

### Double-Threshold Algorithm
`detect_presses` runs a per-sample state machine (`REST` → `CANDIDATE` → `ACTIVE` → `REFRACTORY` → `REST`) driven by two thresholds derived from the rolling baseline:
- **High threshold** `hi = min(max(med + k_high*sigma, med + floor), med + 400)`: crossing it from `REST` opens a candidate press. The value of the threshold is bounded by ~floor and ~400 mA.
- **Low threshold** `lo = min(med + k_low*sigma, med + 100)`: dropping back below it ends a press.

A candidate must stay above `lo` for at least `min_duration_s` to be promoted to `ACTIVE` (rejects short noise blips). An active press ends when the signal drops below `lo`, or is flagged `rejected` if it runs longer than `max_duration_s` (drift/repositioning rather than a squeeze). A `refractory_s` window after every press prevents its own trailing edge from re-triggering a new one. This parameter was first added to discard noise, but became obsolete when the thresholds became more adaptive, so it was set to zero but the parameter was still kept in order to allow its use in the future.

`floor` is a bidirectional adaptive floor added to the high threshold, tracked per sample. It was added as a minimal value for high-treshold, so that the high-threshold didn't become close to zero when the baseline is very low.
- Starts at `floor_max` and resets there at every session start.
- Decays geometrically toward `floor_min` (factor `decay_rate`, applied every `quiet_s` of silence) if `quiet_s` seconds pass with no *confirmed* (accepted, non-rejected) press.
- Rises back toward `rise_frac * peak_amp` (smoothed by `rise_alpha`) every time a press is confirmed.

`exclusion_intervals` (probes and inter-session gaps, from `load_exclusion_intervals`) silently drop any in-progress candidate/active press and block new onsets while the signal is inside one of those windows, so that the presses happening during the probes or between two sessions are not taken into account.

## Parameters
Set as notebook globals in [detection.ipynb](detection.ipynb) and passed into `rolling_baseline`/`detect_presses`.

| Parameter | Meaning | Example value |
|---|---|---|
| `BASELINE_WINDOW_S` | Trailing window (s) used to compute the rolling baseline | 600.0 |
| `K_HIGH` | Multiplier on `sigma` for the high (onset) threshold, kept high to limit false positives | 28.0 |
| `K_LOW` | Multiplier on `sigma` for the low (offset) threshold, kept low since force drops below baseline after most events | 5 |
| `MIN_DURATION` | Minimum time (s) above the low threshold for a candidate to be confirmed | 0.15 |
| `MAX_DURATION` | Maximum time (s) above the low threshold before a press is flagged rejected | 3.5 |
| `REFRACTORY` | Time (s) after a press during which a new one cannot trigger | 0 |
| `FLOOR_MAX` | Starting/ceiling value of the adaptive floor, reset at every session start | 150.0 |
| `FLOOR_MIN` | Floor never decays below this value | 10.0 |
| `QUIET_S` | Seconds with no confirmed press before the floor starts decaying | 60.0 |
| `DECAY_RATE` | Multiplicative decay applied to the floor every `QUIET_S` of silence | 0.9 |
| `RISE_FRAC` | Fraction of `peak_amp` the floor rises toward after a confirmed press | 0.5 |
| `RISE_ALPHA` | Smoothing factor for that rise (closer to 1 = slower to adapt) | 0.7 |

## Outputs
Both CSVs are written to `Scoring_Pingouin/` under `DATA_ROOT`, for the GUI in [Scoring_Pingouin_Claude_V4.py](Scoring_Pingouin_Claude_V4.py):

- **`{SUBJECT}_auto_presses.csv`** — one row per raw candidate press (confirmed and rejected), columns `channel`, `onset_absolute`, `offset_absolute`, `peak_amp`, `rejected`. Drives the pink "AUTO" overlay in the scoring GUI.
- **`{SUBJECT}_events_auto.csv`** — confirmed presses only (rejected candidates dropped), with close re-triggers merged (`merge_close_events`) into single events to allow the estimation of an event's duration (starts at the onset of the first press and ends at the offset of the second press if there is one). Columns `subject`, `session`, `event` (`Response`/`Reset`, from `Subjects.csv`), `channel`, `onset_s`, `offset_s`, `duration_s`.
