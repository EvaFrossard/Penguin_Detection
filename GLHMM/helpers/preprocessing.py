"""
Full EEG preprocessing pipeline
===============================
- Load the raw recordings (EEG/<name>.edf, 500 Hz, referenced to Cz)
- Notch + band-pass filter, downsample, add Cz back, remove the ocular and cardiac ICA components, interpolate bad
channels, average reference
- Mark bad time spans as 'BAD_segment' annotations
- Save the result next to the raw file as EEG/<name>_preprocessed.fif

Bad channels, bad time spans and recordings to discard are read from bad_segments_channels.csv (filled from 
preprocessing.ipynb or by hand).
"""

# IMPORTS

import mne
import os
import ast
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

# CONFIGURATION
DIR_EEG = Path(r"C:\Users\eva\Documents\Master\ICM\Ilo\EEG")
QC_CSV = Path(r"C:\Users\eva\Documents\Master\ICM\Ilo\GLHMM\bad_segments_channels.csv")
RAW_SUFFIX = ".edf"
PREPROC_SUFFIX = "_preprocessed.fif"

L_FREQ = 0.3 # High-frequency filter
H_FREQ = 40.0 # Low-frequency filter
NOTCH_FREQ = 50.0 # Notch filter frequency
REFERENCE  = "average"
REF_CHANNEL = "Cz"  # recording reference: absent from the files, added back as a zero channel before re-referencing
TARGET_FS  = 100   # Hz: downsampling rate

# Every channel is typed 'eeg' in the EDF header, so the non-EEG ones are retyped by name
CHANNEL_TYPES = {
    'IO': 'eog', 'EMG': 'emg', 'ECG': 'ecg', 'Respi': 'resp',
    'L_IndexMid': 'misc', 'L_RingPink': 'misc', 'L_Thumb': 'misc', 'R_RingPink': 'misc', 'R_Thumb': 'misc',
    'Left Index+Middl': 'misc', 'Left Ring+Pinky': 'misc', 'Left Thumb': 'misc',
    'Right Ring+Pinky': 'misc', 'Right Thumb': 'misc',
}

HIGH_BAD_THRESHOLD = 0.30  # fraction of bad EEG channels above which the whole recording is discarded -> Discards TT003

# Bad time spans: samples where more than BAD_SPAN_CH_FRACTION of the EEG channels exceed +-BAD_SPAN_AMP_UV
BAD_SPAN_AMP_UV = 250.0
BAD_SPAN_CH_FRACTION = 0.10
BAD_SPAN_PAD_S = 1.0   # margin added on each side of a detected span
BAD_SPAN_LABEL = "BAD_segment"

# ICA: removes the ocular and cardiac components, found from the IO and ECG channels
ICA_SUFFIX = "_ica.fif"  
ICA_N_COMPONENTS = 30
ICA_L_FREQ = 1.0 # high-pass filtering before ICA
ICA_RANDOM_STATE = 42 # for repeatability
# The excluded components are only subtracted where they are active (blinks, eye movements), so that the
# frontal brain activity they also carry during sleep is kept. None: subtract them everywhere (plain ICA)
ICA_ACTIVE_Z = 4.0 # active where the component exceeds this many robust SDs
ICA_ACTIVE_PAD_S = 0.5 # margin
ICA_RAMP_S = 0.5 # length of the 0 -> 1 ramp at the edges, so that no step is introduced

# ── QC TABLE ──────────────────────────────────────────────────────────────────

def load_qc_table(qc_csv):
    """Load bad_segments_channels.csv, return dict keyed by filename."""
    if not os.path.exists(qc_csv):
        return {}
    qc = pd.read_csv(qc_csv)

    def parse_list(x, default):
        return ast.literal_eval(x) if isinstance(x, str) and x.strip().startswith('[') else default

    qc['bad_channels'] = qc['bad_channels'].apply(parse_list, default=[])
    qc['bad_segment'] = qc['bad_segment'].map({True: True, False: False, 'True': True, 'False': False}).fillna(False)

    if 'bad_spans' not in qc: 
        qc['bad_spans'] = None
    qc['bad_spans'] = qc['bad_spans'].apply(parse_list, default=None).astype(object)

    return qc.set_index('filename').to_dict('index')


def save_qc_entry(qc_csv, filename, bad_channels, bad_segment=False, bad_spans=None):
    """Write (or replace) the row of one file in bad_segments_channels.csv.
    bad_spans=None keeps the spans already stored for that file.
    """
    cols = ['filename', 'bad_segment', 'bad_channels', 'bad_spans']
    qc = pd.read_csv(qc_csv) if os.path.exists(qc_csv) else pd.DataFrame(columns=cols)
    if 'bad_spans' not in qc:
        qc['bad_spans'] = np.nan
    old = qc[qc['filename'] == filename]
    if bad_spans is None:
        spans = old['bad_spans'].iloc[0] if len(old) else np.nan
    else:
        spans = str([[float(a), float(b)] for a, b in bad_spans])
    row = pd.DataFrame([[filename, bool(bad_segment), str(list(bad_channels)), spans]], columns=cols)
    qc = pd.concat([qc[qc['filename'] != filename].astype(object), row.astype(object)], ignore_index=True)
    qc.sort_values('filename')[cols].to_csv(qc_csv, index=False)


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _eeg_names(raw):
    return [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=[])]


def load_edf(path, preload=False):
    """Read one raw EDF with correct channel types and standard 10-20 electrode positions."""
    raw = mne.io.read_raw_edf(path, preload=preload, verbose=False)
    raw.set_channel_types({ch: t for ch, t in CHANNEL_TYPES.items() if ch in raw.ch_names}, verbose=False)
    raw.set_montage('standard_1020', verbose=False)
    if preload:
        restore_io(raw, path)
    return raw


def restore_io(raw, path):
    """Replace the IO channel of a preloaded EDF Raw by the one of the original BrainVision recording.

    The EDF export wrote IO with a wrong scale.
    """
    vhdr = Path(path).with_suffix('.vhdr')
    if 'IO' not in raw.ch_names or not vhdr.exists():
        return raw
    io = mne.io.read_raw_brainvision(vhdr, preload=False, verbose=False).get_data(picks=['IO'])[0]
    # The EDF export pads the end of the recording to a whole second
    if not 0 <= raw.n_times - len(io) < raw.info['sfreq']:
        print(f"{Path(path).name}: IO of {vhdr.name} has another length, not restored")
        return raw
    raw._data[raw.ch_names.index('IO')] = np.r_[io, np.full(raw.n_times - len(io), io[-1])]
    return raw


def add_reference_channel(raw, montage="standard_1020"):
    """Add the recording reference (REF_CHANNEL) back to a preloaded Raw as a zero channel, in place.

    Run it while the data are still referenced to it, i.e. before any re-referencing: the average
    reference then turns it into a real signal. Does nothing when the channel is already there.
    """
    if REF_CHANNEL not in raw.ch_names:
        mne.add_reference_channels(raw, REF_CHANNEL, copy=False)
        raw.set_montage(montage, verbose=False)  # gives it its electrode position
    return raw


def detect_bad_channels(raw, high_ratio=3.0, flat_uv=1):
    """Propose bad EEG channels from their amplitude.

    Run it on filtered data that is NOT yet average-referenced. A channel is proposed when its robust SD (MAD-based) is
    more than `high_ratio` times the median over channels, or below `flat_uv` microvolts. The reference channel is
    left out: it is flat until the data are re-referenced.
    Returns (list of proposed bad channels, Series of robust SD in microvolts).
    """
    eeg_names = [ch for ch in _eeg_names(raw) if ch != REF_CHANNEL]
    data = raw.get_data(picks=eeg_names) * 1e6 # convert to microvolts
    sd = 1.4826 * np.median(np.abs(data - np.median(data, axis=1, keepdims=True)), axis=1) 
    robust_sd = pd.Series(sd, index=eeg_names)
    bad = (robust_sd > high_ratio * robust_sd.median()) | (robust_sd < flat_uv)
    return list(robust_sd.index[bad]), robust_sd


def detect_bad_spans(raw, amp_uv=BAD_SPAN_AMP_UV, ch_fraction=BAD_SPAN_CH_FRACTION, pad_s=BAD_SPAN_PAD_S):
    """Propose bad time spans from the amplitude of the EEG channels.

    Run it on fully preprocessed data. A sample is bad when more than `ch_fraction` of the EEG channels exceed 
    +-`amp_uv` microvolts; `pad_s` is added on each side.
    Returns a list of [start_s, end_s], in seconds from the start of `raw`.
    """
    sfreq = raw.info['sfreq']
    data = raw.get_data(picks=_eeg_names(raw))
    bad = (np.abs(data) > amp_uv * 1e-6).mean(axis=0) > ch_fraction
    del data
    pad = int(round(pad_s * sfreq))
    bad = np.convolve(bad, np.ones(2 * pad + 1), mode='same') > 0
    edges = np.flatnonzero(np.diff(np.r_[0, bad.astype(int), 0]))
    return [[round(a / sfreq, 2), round(b / sfreq, 2)] for a, b in zip(edges[::2], edges[1::2])]


def set_bad_spans(raw, bad_spans):
    """Replace the bad-span annotations of `raw` by `bad_spans` ([start_s, end_s] from the start of raw)."""
    keep = [i for i, d in enumerate(raw.annotations.description)
            if not d.upper().startswith('BAD') or d.upper() == 'BAD_ACQ_SKIP']
    annot = raw.annotations[keep]
    for a, b in bad_spans:
        annot.append(raw.first_time + a, b - a, BAD_SPAN_LABEL)
    raw.set_annotations(annot, verbose=False)
    return raw


def get_bad_spans(raw):
    """Return the bad spans annotated on `raw` as [start_s, end_s] from the start of raw.

    Every annotation starting with 'BAD' counts, except the acquisition skips written by the amplifier.
    """
    spans = [[round(float(o - raw.first_time), 2), round(float(o - raw.first_time + d), 2)]
             for o, d, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description)
             if desc.upper().startswith('BAD') and desc.upper() != 'BAD_ACQ_SKIP']
    return sorted(spans)


def interpolate_and_reference(raw, bad_channels=(), montage="standard_1020"):
    """Interpolate bad channels, then re-reference a preloaded Raw in place.

    Non-EEG channels must be dropped or typed correctly beforehand, otherwise they end up in the average reference.
    """
    bads = [ch for ch in bad_channels if ch in raw.ch_names]
    if bads:
        if raw.get_montage() is None:  # interpolation needs electrode positions
            raw.set_montage(montage, on_missing='ignore', verbose=False)
        raw.info['bads'] = bads
        raw.interpolate_bads(reset_bads=True, verbose=False)
    # add the average reference after interpolation, so that the bad channels only affect themselves
    raw.set_eeg_reference(REFERENCE, projection=False, verbose=False)
    return raw


def filter_raw(raw):
    """Notch and band-pass the EEG channels of a preloaded Raw in place."""
    eeg_names = _eeg_names(raw)
    raw.notch_filter(NOTCH_FREQ, picks=eeg_names, verbose=False)
    raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, picks=eeg_names, verbose=False)
    return raw


def _ica_names(raw, bad_channels=()):
    return [ch for ch in _eeg_names(raw) if ch not in bad_channels]


def fit_ica(raw, bad_channels=()):
    """Fit an ICA on the good EEG channels and mark its ocular and cardiac components as excluded.

    Run it on filtered data referenced to the average of the good channels, bad channels NOT yet interpolated, with the 
    bad spans annotated so that they are left out of the fit. The proposal is in `ica.exclude` (detail in `ica.labels_`)
    and has to be reviewed: during sleep a frontal slow-wave component also correlates with the IO channel.
    """
    picks = _ica_names(raw, bad_channels)
    raw_fit = raw if L_FREQ >= ICA_L_FREQ else raw.copy().filter(ICA_L_FREQ, None, picks=picks, verbose=False)
    ica = mne.preprocessing.ICA(n_components=ICA_N_COMPONENTS, random_state=ICA_RANDOM_STATE, max_iter='auto')
    ica.fit(raw_fit, picks=picks, verbose=False)
    return find_artifact_components(ica, raw_fit)


def find_artifact_components(ica, raw):
    """Set `ica.exclude` to the components correlated with IO (ocular) or time-locked to ECG (cardiac)."""
    exclude = []
    ch_types = raw.get_channel_types()
    if 'eog' in ch_types:
        exclude += ica.find_bads_eog(raw, verbose=False)[0]
    if 'ecg' in ch_types:
        exclude += ica.find_bads_ecg(raw, verbose=False)[0]
    ica.exclude = sorted(set(int(i) for i in exclude))
    return ica


def remove_ica_artifacts(raw, ica, z=ICA_ACTIVE_Z, pad_s=ICA_ACTIVE_PAD_S, ramp_s=ICA_RAMP_S):
    """Subtract the components in `ica.exclude` from a preloaded Raw in place, only where they are active.

    A component is active where its time course exceeds `z` robust SDs (MAD-based), widened by `pad_s` on each side; its 
    weight ramps between 0 and 1 over `ramp_s`. With z=None the components are subtracted everywhere.
    Returns {component: fraction of the recording where it was subtracted}.
    """
    if z is None:
        ica.apply(raw, verbose=False)
        return {k: 1.0 for k in ica.exclude}
    sfreq = raw.info['sfreq']
    idx = [raw.ch_names.index(ch) for ch in ica.ch_names]
    data = raw.get_data(picks=idx)
    sources = ica.get_sources(raw).get_data()
    pad = int(round(pad_s * sfreq))
    ramp = np.hanning(int(round(ramp_s * sfreq)) + 1)
    ramp /= ramp.sum()

    removed = {}
    for k in ica.exclude:
        s = sources[k] - np.median(sources[k])
        active = np.abs(s) > z * 1.4826 * np.median(np.abs(s))
        active = np.convolve(active, np.ones(2 * pad + 1), mode='same') > 0
        weight = np.clip(np.convolve(active, ramp, mode='same'), 0, 1)
        # Contribution of component k alone to the channels, from the original data
        ica_k = ica.copy()
        ica_k.exclude = [k]
        artifact = raw.get_data(picks=idx) - ica_k.apply(raw.copy(), verbose=False).get_data(picks=idx)
        data -= weight * artifact
        removed[k] = float(weight.mean())
    raw._data[idx] = data
    return removed


def load_ica(path_ica, raw, bad_channels=()):
    """Read a saved ICA. Returns None when there is none, or when it was fitted on other channels."""
    if not os.path.exists(path_ica):
        return None
    ica = mne.preprocessing.read_ica(path_ica, verbose=False)
    expected = _ica_names(raw, bad_channels)
    if REF_CHANNEL not in raw.ch_names:  # added at the end by add_reference_channel before the ICA is applied
        expected.append(REF_CHANNEL)
    return ica if ica.ch_names == expected else None


def preprocess_raw(raw, bad_channels=(), montage="standard_1020"):
    """Interpolate bad channels, re-reference, notch and band-pass a preloaded Raw in place."""
    interpolate_and_reference(raw, bad_channels, montage)
    return filter_raw(raw)


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def preprocess_recording(raw, bad_channels=(), bad_spans=None, ica=None):
    """Full preprocessing of one preloaded raw recording, in place.

    Filter, downsample, subtract the excluded ICA components where they are active (ICA fitted here when `ica` is None), 
    interpolate `bad_channels`, average reference, then annotate the bad spans (detected automatically when `bad_spans` 
    is None).
    Returns (raw, bad_spans, ica).
    """
    filter_raw(raw)
    raw.resample(TARGET_FS, verbose=False)  # after the low-pass, so nothing aliases
    add_reference_channel(raw)  # Cz, while the data are still referenced to it
    # Average over the good channels only. Without it the ICA isolates the recording reference as a
    # component present on every channel, which correlates with IO and hides the ocular ones
    raw.info['bads'] = [ch for ch in bad_channels if ch in raw.ch_names]
    raw.set_eeg_reference(REFERENCE, projection=False, verbose=False)
    if ica is None:
        # ICA only models artifacts with a fixed topography: the large irregular ones are kept out of the fit
        set_bad_spans(raw, bad_spans if bad_spans is not None else detect_bad_spans(raw))
        ica = fit_ica(raw, bad_channels)
    remove_ica_artifacts(raw, ica)
    interpolate_and_reference(raw, bad_channels)
    if bad_spans is None:
        bad_spans = detect_bad_spans(raw)
    set_bad_spans(raw, bad_spans)
    return raw, bad_spans, ica


def preprocess_all(data_dir=DIR_EEG, qc_csv=QC_CSV):
    """Preprocess every raw recording of `data_dir` and save it next to it as *_preprocessed.fif.

    Recordings flagged bad_segment in the QC table are skipped. Spans detected
    automatically are written back to the QC table so they can be reviewed.
    The ICA of each recording is saved as *_ica.fif and reused as is on the
    next runs, with the components excluded in it: delete the file to fit it again.
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(f"*{RAW_SUFFIX}"))
    print(f"Found {len(files)} recording(s) in {data_dir}\n")

    for path in files:
        qc = load_qc_table(qc_csv).get(path.name, {})
        path_out = path.with_name(path.name.replace(RAW_SUFFIX, PREPROC_SUFFIX))
        if qc.get('bad_segment', False):
            print(f"{path.name}: flagged as bad recording, skipped")
            path_out.unlink(missing_ok=True)  # do not leave an outdated preprocessed file behind
            continue

        raw = load_edf(path, preload=True)
        if np.isnan(raw.get_data()).any():
            print(f"{path.name}: contains NaN, skipped")
            continue

        bad_channels = qc.get('bad_channels', [])
        spans_reviewed = qc.get('bad_spans') is not None
        path_ica = path.with_name(path.name.replace(RAW_SUFFIX, ICA_SUFFIX))
        ica = load_ica(path_ica, raw, bad_channels)
        ica_saved = ica is not None
        raw, bad_spans, ica = preprocess_recording(raw, bad_channels, qc.get('bad_spans'), ica)
        raw.save(path_out, overwrite=True, verbose=False)
        if not ica_saved:
            ica.save(path_ica, overwrite=True, verbose=False)
        if not spans_reviewed:
            save_qc_entry(qc_csv, path.name, bad_channels, False, bad_spans)

        bad_s = sum(b - a for a, b in bad_spans)
        print(f"{path.name} -> {path_out.name}: {len(bad_channels)} channel(s) interpolated, "
              f"ICA component(s) {ica.exclude} removed ({'from ' + path_ica.name if ica_saved else 'detected automatically'}), "
              f"{len(bad_spans)} bad span(s) = {bad_s:.0f} s ({bad_s / raw.times[-1]:.1%}, "
              f"{'from the QC table' if spans_reviewed else 'detected automatically'})")
        del raw


def refresh_qc_table(data_dir=DIR_EEG, qc_csv=QC_CSV, files=None):
    """Rebuild bad_segments_channels.csv for every raw recording of `data_dir` (or only the names in `files`).

    The bad channels of a recording are the automatic proposal of detect_bad_channels plus those already in
    the table, including the rows of the old '<name>_filtered.edf' / '<name>_a_filtered.edf' files, which
    are dropped. bad_segment is kept when it was set, and set when more than HIGH_BAD_THRESHOLD of the EEG
    channels are bad. The bad spans are cleared, so that preprocess_all detects them again on the
    preprocessed data. The previous table is kept as *_backup.csv. Review the proposals by eye
    (preprocessing.ipynb) before relying on them.
    """
    data_dir, qc_csv = Path(data_dir), Path(qc_csv)
    paths = sorted(data_dir.glob(f"*{RAW_SUFFIX}")) if files is None else [data_dir / f for f in files]
    old = load_qc_table(qc_csv)
    if qc_csv.exists():
        shutil.copy(qc_csv, qc_csv.with_name(qc_csv.stem + '_backup.csv'))

    new, replaced = {}, set()
    for path in paths:
        stem = path.name[:-len(RAW_SUFFIX)]
        keys = [k for k in (path.name, f'{stem}_filtered.edf', f'{stem}_a_filtered.edf') if k in old]
        replaced.update(keys)
        bad_channels = sorted(set().union(*[old[k]['bad_channels'] for k in keys]))
        bad_segment = any(old[k]['bad_segment'] for k in keys)
        if bad_segment:
            print(f"{path.name}: flagged as bad recording, kept as is")
        else:
            raw = load_edf(path, preload=True)
            filter_raw(raw)
            raw.resample(TARGET_FS, verbose=False)
            auto, _ = detect_bad_channels(raw)
            eeg_names = _eeg_names(raw)
            bad_channels = [ch for ch in eeg_names if ch in auto or ch in bad_channels]
            bad_segment = len(bad_channels) / len(eeg_names) > HIGH_BAD_THRESHOLD
            print(f"{path.name}: {bad_channels} (automatic {auto}, already in the table "
                  f"{sorted(set(bad_channels) - set(auto))}){', bad recording' if bad_segment else ''}")
            del raw
        new[path.name] = [path.name, bad_segment, str(bad_channels), np.nan]

    kept = [[k, v['bad_segment'], str(v['bad_channels']),
             np.nan if v['bad_spans'] is None else str(v['bad_spans'])] for k, v in old.items() if k not in replaced]
    cols = ['filename', 'bad_segment', 'bad_channels', 'bad_spans']
    table = pd.DataFrame(kept + list(new.values()), columns=cols).sort_values('filename')
    table.to_csv(qc_csv, index=False)
    print(f"\n{qc_csv.name}: {len(new)} recording(s) updated, {len(replaced - set(new))} old row(s) dropped, "
          f"bad spans cleared (previous table in {qc_csv.stem}_backup.csv)")


if __name__ == "__main__":
    preprocess_all()
