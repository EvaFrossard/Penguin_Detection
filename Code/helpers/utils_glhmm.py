import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glhmm import glhmm, preproc, utils, graphics
import mne
from pathlib import Path

from scipy import signal
from scipy.ndimage import uniform_filter1d
from scipy import stats


from IPython.display import clear_output

########################################################################################################################
# LOAD EEG DATA
########################################################################################################################

def find_files(data_dir, pattern, VERBOSE=True):
    files = sorted(data_dir.glob(pattern))
    if VERBOSE: print(f"Found {len(files)} files: {[f.name for f in files]}")
    assert len(files) > 0, f"No files found in {data_dir} matching pattern {pattern}"
    return files

def load_sleep_stages(hyp_df):
    """Hypnogram (one row per scored 30-s epoch) -> {subject: {epoch_idx: sleep_stage}}."""
    hyp_df = hyp_df[hyp_df.recording_part != 'part2']  # part2 files are skipped in load_eeg_data and restart epoch_idx at 0
    return {sub: g.set_index('epoch_idx').sleep_stage.to_dict() for sub, g in hyp_df.groupby('subname')}

def load_eeg_data(fif_files, experiment_stage, session_map, stages_df, full_stage=True, crop_end_probe=True,
                  nap_duration_min=30, verbose=True):
    """Load and concatenate the EEG windows belonging to one experiment stage.
    Returns Y, indices, sessions_meta, fs and the EEG channel names."""

    stage_sessions = session_map[session_map.session == experiment_stage]
    data_raw = stage_sessions.set_index('subname').session_raw

    def event_times(subject, session, event_mask):
        """Get the onset times (s from recording start) of events matching the mask for a given subject and session."""
        rows = stages_df[(stages_df.subname == subject) & (stages_df.session == session) & event_mask]
        return rows.onset_s

    # Get the start and end times of the experiment stage for each subject
    start_s = {subject: event_times(subject, session, stages_df.event.str.startswith('Start')).min()
        for subject, session in data_raw.items()}
    end_s = {subject: event_times(subject, session, stages_df.event.str.startswith('End')).max()
        for subject, session in data_raw.items()}
    # Get the last probe time for each subject (if any)
    last_probe_s = {subject: event_times(subject, session, stages_df.event.str.contains('Probe')
                                & (stages_df.event != 'End Probe')).max()
                                for subject, session in data_raw.items()}

    if verbose:
        print(f"{experiment_stage} start times (s from recording start): {start_s}")
        print(f"{experiment_stage} end times (s from recording start): {end_s}")

    Y_list = []
    starts, ends = [], []
    meta_rows = []
    offset = 0

    for fif_file in fif_files:
        subject = fif_file.stem.split('_')[0]
        if verbose:
            print(f"Processing {subject} from file {fif_file.name}...")
        if subject == 'TT003' or 'part2' in fif_file.name:
            continue
        if np.isnan(start_s.get(subject, np.nan)):
            if verbose:
                print(f"  no {experiment_stage} start found for {subject}, skipping")
            continue

        crop_start_s = start_s[subject]
        raw = mne.io.read_raw_fif(fif_file, preload=False, verbose=0)
        crop_end_s = end_s[subject]
        if crop_end_probe and not np.isnan(last_probe_s.get(subject, np.nan)): # crop at the last probe if it exists
            crop_end_s = last_probe_s[subject]
        if not full_stage: # crop at the start + nap_duration_min if the stage is longer than that
            crop_end_s = min(crop_end_s, crop_start_s + nap_duration_min * 60) 
        raw.crop(tmin=crop_start_s, tmax=crop_end_s, include_tmax=False)

        raw.pick('eeg')
        if not Y_list:
            fs, ch_names = int(raw.info['sfreq']), raw.ch_names
        elif int(raw.info['sfreq']) != fs or raw.ch_names != ch_names:
            raise ValueError(f'{subject}: sampling rate or EEG channels differ from the first session')
        bad_spans = [(max(o - raw.first_time, 0), min(o - raw.first_time + d, raw.times[-1]))
                    for o, d, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description)
                    if desc.upper().startswith('BAD')]
        bad_s = sum(b - a for a, b in bad_spans)

        X = raw.get_data(reject_by_annotation=None, verbose=0).T.astype(np.float32) * 1e6

        Y_list.append(X)
        starts.append(offset)
        ends.append(offset + len(X))
        offset += len(X)
        meta_rows.append({
            'subject': subject,
            'file': str(fif_file),
            'start': starts[-1],
            'end': ends[-1],
            'crop_start_s': crop_start_s,  # absolute time (s from recording start) of the first sample
            'crop_end_s': crop_end_s,  # absolute time (s from recording start) where the window stops
            'duration_min': round(len(X) / fs / 60, 1),
            'bad_s': round(bad_s, 1),
            'bad_spans': bad_spans,
        })
        if verbose:
            print(f"  {subject}: {len(X) / fs / 60:.1f} min, of which {bad_s:.0f} s ({bad_s / (len(X) / fs):.1%}) annotated as bad and kept")

    if not Y_list:
        raise ValueError(f'No EEG files found for experiment stage {experiment_stage!r}')

    Y = np.vstack(Y_list)
    indices = np.column_stack([starts, ends])
    sessions_meta = pd.DataFrame(meta_rows)

    return Y, indices, sessions_meta, fs, ch_names

def band_power_envelopes(X, fs, ch_names, REGIONS, BANDS, FS_ENV=10, ENV_SMOOTH_S=0.5):
    """(n_times, n_channels) EEG -> (n_times_env, n_regions * n_bands) log band envelopes."""
    prefix = [c.rstrip('0123456789z') for c in ch_names]
    region_idx = {r: [i for i, pr in enumerate(prefix) if pr in pre] for r, pre in REGIONS.items()}
    step = int(fs / FS_ENV)
    feats = []
    for lo, hi in BANDS.values():
        sos = signal.butter(4, (lo, hi), 'bandpass', output='sos', fs=fs)
        env = np.abs(signal.hilbert(signal.sosfiltfilt(sos, X, axis=0), axis=0))
        for idx in region_idx.values():
            e = np.median(env[:, idx], axis=1)  # median over channels: robust to a single bad electrode
            e = uniform_filter1d(e, int(ENV_SMOOTH_S * fs), mode='nearest')[step // 2::step]
            feats.append(np.log(e))
    return np.column_stack(feats)



########################################################################################################################
# LOAD PHYSIOLOGICAL DATA
########################################################################################################################

def load_physiological_data(sessions_meta, channel_name, fs, vhdr_dir=None):
    """Re-open each session's recording and crop exactly the same window as its EEG (load_eeg_data dropped the other channels).
    vhdr_dir=None reads the preprocessed .fif (at the EEG rate fs); otherwise the raw BrainVision <subject>.vhdr in vhdr_dir,
    at its original rate (it shares the time axis of the .fif).
    Returns {session index (row of sessions_meta): (n_times,) signal in microvolts}, only for the sessions that have the
    channel, and the sampling rate of the signal."""
    sig_raw, sig_fs = {}, fs
    for s, row in sessions_meta.iterrows():
        if vhdr_dir is None:
            raw = mne.io.read_raw_fif(row.file, preload=False, verbose=0)
        else:
            raw = mne.io.read_raw_brainvision(Path(vhdr_dir) / f'{row.subject}.vhdr', preload=False, verbose=0)
        if channel_name not in raw.ch_names:
            print(f"  no {channel_name} channel for {row.subject}")
            continue
        sig_fs = raw.info['sfreq']
        n = row.end - row.start  # number of EEG samples of this session
        raw.crop(tmin=row.crop_start_s, tmax=row.crop_start_s + n / fs, include_tmax=False)
        sig_raw[s] = raw.get_data(picks=[channel_name], reject_by_annotation=None, verbose=0)[0] * 1e6
        del raw
        clear_output(wait=True)
        if sig_fs == fs:
            assert len(sig_raw[s]) == n, f'{row.subject}: {len(sig_raw[s])} {channel_name} samples for {n} EEG samples'

    print(f'{channel_name} found for {len(sig_raw)} / {len(sessions_meta)} sessions')
    return sig_raw, sig_fs


def emg_envelope(x, fs, EMG_BAND=(20, 250), FS_ENV=10, ENV_SMOOTH_S=0.5):
    """(n_times,) EMG -> (n_times_env,) log RMS in EMG_BAND, on the same grid as band_power_envelopes."""
    sos = signal.butter(8, EMG_BAND, 'bandpass', output='sos', fs=fs)  # order 8: the 50 Hz line noise is ~1000x the muscle band
    step = int(fs / FS_ENV)
    rms = np.sqrt(uniform_filter1d(signal.sosfiltfilt(sos, x) ** 2, int(ENV_SMOOTH_S * fs), mode='nearest'))
    return np.log(rms[step // 2::step])


def verify_gaussianity(feature, feature_names):
    """Histogram + Q-Q plot of each feature column, and their correlation when there are several.
    feature: {session index: (n_times_env,) or (n_times_env, n_features)}; feature_names: one name per column."""
    if isinstance(feature_names, str):
        feature_names = [feature_names]
    # Standardised within each session, as it will be in Y_prep. The HMM only needs the feature to be Gaussian *within* a state:
    # pooled over states the histogram may be skewed or bimodal (e.g. wake vs sleep), which is fine
    z = np.vstack([((e - e.mean(axis=0)) / e.std(axis=0)).reshape(len(e), -1) for e in feature.values()])
    if len(feature_names) > 1:
        print(f'correlation between the features:\n{pd.DataFrame(np.corrcoef(z.T), index=feature_names, columns=feature_names).round(2)}')

    fig, axes = plt.subplots(len(feature_names), 2, figsize=(10, 3.5 * len(feature_names)), squeeze=False)
    xx = np.linspace(-5, 5, 200)
    for i, name in enumerate(feature_names):
        v = z[:, i]
        axes[i, 0].hist(v, bins=200, range=(-5, 5), density=True, color='0.6')
        axes[i, 0].plot(xx, stats.norm.pdf(xx), 'k')
        axes[i, 0].set_title(f'{name}: skew {stats.skew(v):.2f}, excess kurtosis {stats.kurtosis(v):.2f}')
        axes[i, 0].set_xlabel('z-score')
        stats.probplot(v, dist='norm', plot=axes[i, 1])
        axes[i, 1].set_title(f'Q-Q plot, {name}')
    plt.tight_layout()
    plt.show()


def mean_by_stage(feature, feature_names, sessions_meta, sleep_stages, FS_ENV, EPOCH_S=30):
    """Mean of each feature column, z-scored within each session, per subject and sleep stage.
    feature: {session index: (n_times_env,) or (n_times_env, n_features)}; sleep_stages: from load_sleep_stages."""
    if isinstance(feature_names, str):
        feature_names = [feature_names]
    by_stage = []
    for s, e in feature.items():
        row = sessions_meta.loc[s]
        t_abs = row.crop_start_s + (np.arange(len(e)) + 0.5) / FS_ENV  # s from recording start of each envelope sample
        stg = [sleep_stages.get(row.subject, {}).get(int(t // EPOCH_S)) for t in t_abs]
        z = ((e - e.mean(axis=0)) / e.std(axis=0)).reshape(len(e), -1)
        by_stage.append(pd.DataFrame({'subject': row.subject, 'stage': stg, **dict(zip(feature_names, z.T))}))
    by_stage = pd.concat(by_stage).dropna()
    return by_stage.pivot_table(index='subject', columns='stage', values=feature_names).round(2)


def add_feat_to_Y(Y_prep, indices_prep, indices, sessions_meta, feature_names, feat, new_names, missing_reason='missing'):
    """Append the columns of feat {session index: (n_times_env,) or (n_times_env, n_new)} to Y_prep, standardised within
    each subject like the EEG features. Every session must have the same features, so the sessions missing from feat are dropped.
    Returns the updated Y_prep, indices_prep, indices, sessions_meta and feature_names."""
    new_names = list(new_names)
    assert not set(new_names) & set(feature_names), f'{new_names} already added: re-run the data preparation cells first'

    keep = sessions_meta.index.isin(list(feat))
    if not keep.all():
        print(f'dropping {sessions_meta.subject[~keep].tolist()} ({missing_reason})')
    Y_prep = np.vstack([np.column_stack([Y_prep[a:b], feat[s]])
                        for s, (a, b) in zip(sessions_meta.index[keep], indices_prep[keep])])
    lengths = np.diff(indices_prep[keep], axis=1)[:, 0]
    ends_prep = np.cumsum(lengths)
    indices_prep = np.column_stack([ends_prep - lengths, ends_prep])
    indices = indices[keep]  # still points into the raw EEG Y, which is left untouched
    sessions_meta = sessions_meta[keep].reset_index(drop=True)

    n = len(new_names)
    for sub in sessions_meta.subject.unique():
        rows = np.concatenate([np.arange(a, b) for a, b in indices_prep[(sessions_meta.subject == sub).values]])
        Y_prep[rows, -n:] = (Y_prep[rows, -n:] - Y_prep[rows, -n:].mean(axis=0)) / Y_prep[rows, -n:].std(axis=0)

    feature_names = list(feature_names) + new_names
    assert np.isfinite(Y_prep).all(), 'NaN/inf in Y_prep'
    print(f'Y_prep shape: {Y_prep.shape}  ({len(feature_names)} features)')
    print(f'indices_prep shape: {indices_prep.shape}')
    return Y_prep, indices_prep, indices, sessions_meta, feature_names


# The .fif and the BrainVision recording share the same time axis, so the EEG window of each session is cropped directly
# ecg_feat = {}  # session index (row of sessions_meta) -> (n_times_env, len(ECG_FEATURES)), only for the usable ECGs
# for s, row in sessions_meta.iterrows():
#     raw = mne.io.read_raw_brainvision(data_dir / f'{row.subject}.vhdr', preload=False, verbose=0)
#     if 'ECG' not in raw.ch_names:
#         print(f"  no ECG channel for {row.subject}")
#         continue
#     fs_ecg = raw.info['sfreq']
#     n = row.end - row.start  # number of EEG samples of this session
#     x = raw.get_data(picks=['ECG'], tmin=row.crop_start_s, tmax=row.crop_start_s + n / fs, verbose=0)[0] * 1e6
#     del raw
#     t = r_peaks(x, fs_ecg)
#     kept = clean_rr(t)[2] if len(t) > 2 else 0.0
#     if len(t) < len(x) / fs_ecg / 2 or kept < ECG_MIN_RR_KEPT:  # < 30 bpm detected, or too many beats missed / extra
#         print(f"  {row.subject}: unusable ECG ({len(t)} beats, {kept:.0%} of RR intervals kept)")
#         continue
#     ecg_feat[s] = ecg_features(t, indices_prep[s, 1] - indices_prep[s, 0])
#     print(f"  {row.subject}: {len(t)} beats, {kept:.0%} of RR intervals kept, median HR {60 / np.median(np.diff(t)):.0f} bpm")

# print(f'usable ECG for {len(ecg_feat)} / {len(sessions_meta)} sessions')