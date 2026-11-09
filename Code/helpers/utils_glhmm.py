import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glhmm import glhmm, preproc, utils, graphics
import mne
from pathlib import Path

from scipy import signal
from scipy.ndimage import uniform_filter1d
from scipy import stats
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter1d

from IPython.display import clear_output
import logging


########################################################################################################################
# LOAD EEG DATA
########################################################################################################################

def find_files(data_dir, pattern, VERBOSE=True, log=None):
    files = sorted(data_dir.glob(pattern))
    if VERBOSE: print(f"Found {len(files)} files: {[f.name for f in files]}")
    if log: log.info(f"Found {len(files)} files: {[f.name for f in files]}")
    if log: assert len(files) > 0, log.error(f"No files found in {data_dir} matching pattern {pattern}")
    return files

def load_sleep_stages(hyp_df):
    """Hypnogram (one row per scored 30-s epoch) -> {subject: {epoch_idx: sleep_stage}}."""
    hyp_df = hyp_df[hyp_df.recording_part != 'part2']  # part2 files are skipped in load_eeg_data and restart epoch_idx at 0
    return {sub: g.set_index('epoch_idx').sleep_stage.to_dict() for sub, g in hyp_df.groupby('subname')}

def load_eeg_data(fif_files, experiment_stage, session_map, stages_df, full_stage=True, crop_end_probe=True,
                  nap_duration_min=30, verbose=True, log=None):
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

    if log:
        log.info(f"{experiment_stage} start times (s from recording start): {start_s}")
        log.info(f"{experiment_stage} end times (s from recording start): {end_s}")

    Y_list = []
    starts, ends = [], []
    meta_rows = []
    offset = 0

    for fif_file in fif_files:
        subject = fif_file.stem.split('_')[0]
        if verbose:
            print(f"Processing {subject} from file {fif_file.name}...")
        if log:
            log.info(f"EEG: processing {subject} from file {fif_file.name}...")
        if subject == 'TT003' or 'part2' in fif_file.name:
            if log: log.warning(f"  skipping {fif_file.name} (TT003 or part2)")
            continue
        if np.isnan(start_s.get(subject, np.nan)):
            if log: log.warning(f"  no {experiment_stage} start found for {subject}, skipping")
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
        if log:
            log.info(f"  {subject}: {len(X) / fs / 60:.1f} min, of which {bad_s:.0f} s ({bad_s / (len(X) / fs):.1%}) annotated as bad and kept")

    if not Y_list:
        raise ValueError(f'No EEG files found for experiment stage {experiment_stage!r}')

    Y = np.vstack(Y_list)
    indices = np.column_stack([starts, ends])
    sessions_meta = pd.DataFrame(meta_rows)

    return Y, indices, sessions_meta, fs, ch_names

def band_power_envelopes(X, fs, ch_names, REGIONS, BANDS, FS_ENV=10, ENV_SMOOTH_S=0.5):
    """(n_times, n_channels) EEG -> (n_times_env, n_regions * n_bands) log band envelopes."""
    prefix = [c.rstrip('0123456789z') for c in ch_names] # remove trailing characters 
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

def get_logger(log_file, mode='w'):
    """Logger that prints to the notebook and writes to log_file."""
    log = logging.getLogger('glhmm_prep')
    log.setLevel(logging.INFO)
    log.handlers.clear()  # re-running a cell would otherwise add the handlers again and duplicate every line
    log.propagate = False
    fh = logging.FileHandler(log_file, mode=mode, encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s  %(message)s', '%Y-%m-%d %H:%M:%S'))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(message)s'))
    log.addHandler(fh)
    log.addHandler(sh)
    return log


########################################################################################################################
# LOAD PHYSIOLOGICAL DATA
########################################################################################################################

def load_physiological_data(sessions_meta, channel_name, fs, vhdr_dir=None, VERBOSE=False, log=None):
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
            if log: log.warning(f"  no {channel_name} channel for {row.subject}")
            if VERBOSE: print(f"  no {channel_name} channel for {row.subject}")
            continue
        sig_fs = raw.info['sfreq']
        n = row.end - row.start  # number of EEG samples of this session
        raw.crop(tmin=row.crop_start_s, tmax=row.crop_start_s + n / fs, include_tmax=False)
        sig_raw[s] = raw.get_data(picks=[channel_name], reject_by_annotation=None, verbose=0)[0] * 1e6
        del raw
        if sig_fs == fs:
            assert len(sig_raw[s]) == n, f'{row.subject}: {len(sig_raw[s])} {channel_name} samples for {n} EEG samples'

    if log: log.info(f'{channel_name} found for {len(sig_raw)} / {len(sessions_meta)} sessions')
    if VERBOSE: print(f'{channel_name} found for {len(sig_raw)} / {len(sessions_meta)} sessions')
    return sig_raw, sig_fs


def emg_envelope(x, fs, EMG_BAND=(20, 250), FS_ENV=10, ENV_SMOOTH_S=0.5):
    """(n_times,) EMG -> (n_times_env,) log RMS in EMG_BAND, on the same grid as band_power_envelopes."""
    sos = signal.butter(8, EMG_BAND, 'bandpass', output='sos', fs=fs)  # order 8: the 50 Hz line noise is ~1000x the muscle band
    step = int(fs / FS_ENV)
    rms = np.sqrt(uniform_filter1d(signal.sosfiltfilt(sos, x) ** 2, int(ENV_SMOOTH_S * fs), mode='nearest'))
    return np.log(rms[step // 2::step])


def verify_gaussianity(feature, feature_names, save_path=None, log=None, VERBOSE=True):
    """Histogram + Q-Q plot of each feature column, and their correlation when there are several.
    feature: {session index: (n_times_env,) or (n_times_env, n_features)}; feature_names: one name per column."""
    if isinstance(feature_names, str):
        feature_names = [feature_names]
    # Standardised within each session, as it will be in Y_prep. The HMM only needs the feature to be Gaussian *within* a state:
    # pooled over states the histogram may be skewed or bimodal (e.g. wake vs sleep), which is fine
    z = np.vstack([((e - e.mean(axis=0)) / e.std(axis=0)).reshape(len(e), -1) for e in feature.values()])
    if len(feature_names) > 1:
        if VERBOSE: print(f'correlation between the features:\n{pd.DataFrame(np.corrcoef(z.T), index=feature_names, columns=feature_names).round(2)}')
        if log: log.warning(f'correlation between the features:\n{pd.DataFrame(np.corrcoef(z.T), index=feature_names, columns=feature_names).round(2)}')

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
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


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


def add_feat_to_Y(Y_prep, indices_prep, indices, sessions_meta, feature_names, feat, new_names, missing_reason='missing', VERBOSE=True, log=None):
    """Append the columns of feat {session index: (n_times_env,) or (n_times_env, n_new)} to Y_prep, standardised within
    each subject like the EEG features. Every session must have the same features, so the sessions missing from feat are dropped.
    Returns the updated Y_prep, indices_prep, indices, sessions_meta and feature_names."""
    new_names = list(new_names)
    assert not set(new_names) & set(feature_names), f'{new_names} already added: re-run the data preparation cells first'

    keep = sessions_meta.index.isin(list(feat))
    if not keep.all():
        if VERBOSE:
            print(f'dropping {sessions_meta.subject[~keep].tolist()} ({missing_reason})')
        if log:
            log.warning(f'dropping {sessions_meta.subject[~keep].tolist()} ({missing_reason})')
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
    if VERBOSE:
        print(f'Y_prep shape: {Y_prep.shape}  ({len(feature_names)} features)')
        print(f'indices_prep shape: {indices_prep.shape}')
    if log:
        log.info(f'Y_prep shape: {Y_prep.shape}  ({len(feature_names)} features)')
        log.info(f'indices_prep shape: {indices_prep.shape}')
    return Y_prep, indices_prep, indices, sessions_meta, feature_names

def eog_envelope(x, fs, FS_ENV, band, smooth_s):
    """(n_times,) EOG -> (n_times_env,) log amplitude envelope in band, on the same grid as band_power_envelopes."""
    sos = signal.butter(2, band, 'bandpass', output='sos', fs=fs)
    env = np.abs(signal.hilbert(signal.sosfiltfilt(sos, x)))
    step = int(fs / FS_ENV)
    return np.log(uniform_filter1d(env, int(smooth_s * fs), mode='nearest')[step // 2::step])

def r_peaks(x, fs):
    """(n_times,) ECG in microvolts -> times (s) of the R-peaks, with MNE's adaptive QRS detector."""
    info = mne.create_info(['ECG'], fs, 'ecg')
    ev = mne.preprocessing.find_ecg_events(mne.io.RawArray(x[None] * 1e-6, info, verbose=False), ch_name='ECG', verbose=False)[0]
    return ev[:, 0] / fs


def clean_rr(t):
    """Beat times -> (end time, length) of the RR intervals kept, and the fraction kept. An interval is kept between
    30 and 180 bpm and within 20 % of the median of the 11 around it: a missed or extra beat gives one far from its neighbours."""
    rr, tm = np.diff(t), t[1:]
    med = median_filter(rr, size=11, mode='nearest')
    ok = (rr > 0.33) & (rr < 2.0) & (np.abs(rr - med) < 0.2 * med)
    return tm[ok], rr[ok], ok.mean()



def ecg_features(t_beats, n_env, FS_ENV, HR_SMOOTH_S, HRV_WIN_S, ECG_FEATURES):
    """R-peak times (s from the window start) -> (n_env, len(ECG_FEATURES)) log HR / log RMSSD on the envelope grid."""
    tm, rr, _ = clean_rr(t_beats)
    te = (np.arange(n_env) + 0.5) / FS_ENV  # centre of each envelope sample, s from the window start
    feats = {}
    # Heart rate: one value per beat, interpolated onto the grid (held constant before the first / after the last beat)
    hr = PchipInterpolator(tm, 60 / rr, extrapolate=False)(te)
    hr = np.where(te < tm[0], 60 / rr[0], np.where(te > tm[-1], 60 / rr[-1], hr))
    feats['HR'] = np.log(uniform_filter1d(hr, int(HR_SMOOTH_S * FS_ENV), mode='nearest'))
    # RMSSD over a centred HRV_WIN_S window, only from differences between two consecutive kept RR intervals
    consec = np.isclose(np.diff(tm), rr[1:], atol=1e-3)  # the next kept interval starts where this one ends
    td, d2 = tm[1:][consec], np.diff(rr)[consec] ** 2
    csum = np.r_[0, np.cumsum(d2)]
    lo, hi = np.searchsorted(td, te - HRV_WIN_S / 2), np.searchsorted(td, te + HRV_WIN_S / 2)
    ms = np.where(hi - lo >= 10, (csum[hi] - csum[lo]) / np.maximum(hi - lo, 1), np.nan)
    ms = pd.Series(ms).ffill().bfill().values  # windows with < 10 differences (edges, noisy spans) take the closest value
    feats['RMSSD'] = np.log(np.sqrt(ms) * 1000)
    return np.column_stack([feats[k] for k in ECG_FEATURES])

########################################################################################################################
# LOAD BEHAVIOURAL FEATURES
########################################################################################################################

def paused_mask(subject, crop_start_s, te, stages_df):
    """True on the envelope samples inside a probe (from its onset to 'End Probe'), where the task is paused."""
    ev = stages_df[(stages_df.subname == subject) & (stages_df.onset_s >= crop_start_s)].drop_duplicates(['event', 'onset_s'])
    on = ev[ev.event.str.contains('Probe') & (ev.event != 'End Probe')].onset_s.values - crop_start_s
    off = np.sort(ev[ev.event == 'End Probe'].onset_s.values - crop_start_s)
    paused = np.zeros(len(te), bool)
    for a in on:
        b = off[off > a].min() if (off > a).any() else np.inf
        paused |= (te >= a) & (te < b)
    return paused



def behav_features(subject, crop_start_s, n_env, FS_ENV, intervals, presses, BEHAV_FEATURES, PRESS_RATE_SIGMA_S, stages_df):
    """-> (n_env, len(BEHAV_FEATURES)) behavioural features on the envelope grid, or None with fewer than 2 IRT intervals."""
    r = intervals[intervals.subname == subject]
    r = r[r.scorer == r.scorer.value_counts().idxmax()].sort_values('t_start_s') if len(r) else r  # one scorer per subject
    if len(r) < 2:
        return None
    te = (np.arange(n_env) + 0.5) / FS_ENV  # centre of each envelope sample, s from the window start
    feats = {}
    # IRT: log(irt_ratio) at the midpoint of each interval, PCHIP between them (no overshoot), edge values held.
    # Between two presses more than ~90 s apart there is no interval: the value there is only interpolated
    tk = ((r.t_start_s + r.t_end_s) / 2).values
    yk = np.log(r.irt_ratio.values)
    irt = PchipInterpolator(tk, yk, extrapolate=False)(te)
    feats['IRT'] = np.where(te < tk[0], yk[0], np.where(te > tk[-1], yk[-1], irt))
    # Press rates: Response and Reset presses per minute, Gaussian kernel. They fall when the presses stop (lapses, sleep)
    p = presses[presses.subname == subject]
    for name, event in [('response rate', 'Response'), ('reset rate', 'Reset')]:
        idx = (p[p.event == event].t_in_session_s.values * FS_ENV).astype(int)
        counts = np.bincount(idx[(idx >= 0) & (idx < n_env)], minlength=n_env).astype(float)
        feats[name] = gaussian_filter1d(counts, PRESS_RATE_SIGMA_S * FS_ENV, mode='nearest') * FS_ENV * 60
    out = np.column_stack([feats[k] for k in BEHAV_FEATURES])
    # The task is paused during the probes: hold the values reached at the probe onset instead of a rate falling to 0
    out[paused_mask(subject, crop_start_s, te, stages_df)] = np.nan
    return pd.DataFrame(out).ffill().bfill().values


########################################################################################################################
# FULL LOADING FUNCTION
########################################################################################################################

GAMM_DIR = Path('C:/Users/eva/Documents/Master/ICM/Ilo/Data/GAMM')
data_dir = Path('C:/Users/eva/Documents/Master/ICM/Ilo/Data/EEG')
OUT_DIR = Path('C:/Users/eva/Documents/Master/ICM/Ilo/Outputs/GLHMM')


def load_data(BANDS, REGIONS, EXPERIMENT_STAGE = 'Nap1', NAP_DURATION_MIN = 30, FULL_STAGE = True, CROP_END_PROBE = True, PREPROC = True,
              SLEEP_STAGES = True, VERBOSE = False, ENV_SMOOTH_S = 2, FS_ENV = 4, EMG = False, EMG_BAND = None, EOG = False,
              EOG_FEATURES = None, ECG = False, ECG_FEATURES = None, HR_SMOOTH_S = 4, HRV_WIN_S = 30, ECG_MIN_RR_KEPT = 0.8,
              BEHAV = False, BEHAV_FEATURES = None, PRESS_RATE_SIGMA_S = 30, MIN_IRT_S = 5):
    """Load and prepare the EEG, physiological and behavioural features for the HMM.
    The result is cached in OUT_DIR/<TAG>/<TAG>.pkl with the parameters it was computed with, and recomputed when they differ.
    Returns Y_prep, indices_prep, indices, sessions_meta, fs (rate of the raw EEG), fs_prep (rate of Y_prep), ch_names,
    feature_names."""

    # Every parameter the result depends on. Those of a physiological / behavioural feature only count when it is added
    params = dict(EXPERIMENT_STAGE=EXPERIMENT_STAGE, FULL_STAGE=FULL_STAGE, NAP_DURATION_MIN=NAP_DURATION_MIN,
                  CROP_END_PROBE=CROP_END_PROBE, SLEEP_STAGES=SLEEP_STAGES, PREPROC=PREPROC, BANDS=BANDS, REGIONS=REGIONS,
                  ENV_SMOOTH_S=ENV_SMOOTH_S, FS_ENV=FS_ENV)
    if EMG:
        params.update(EMG_BAND=EMG_BAND)
    if EOG:
        params.update(EOG_FEATURES=EOG_FEATURES)
    if ECG:
        params.update(ECG_FEATURES=ECG_FEATURES, HR_SMOOTH_S=HR_SMOOTH_S, HRV_WIN_S=HRV_WIN_S, ECG_MIN_RR_KEPT=ECG_MIN_RR_KEPT)
    if BEHAV:
        params.update(BEHAV_FEATURES=BEHAV_FEATURES, PRESS_RATE_SIGMA_S=PRESS_RATE_SIGMA_S, MIN_IRT_S=MIN_IRT_S)

    # The tag names the folder and the files; the parameters it leaves out are checked against the ones saved in the cache
    MODEL_TAG = 'bandpower' if SLEEP_STAGES else 'raw'  # keeps the two kinds of model apart on disk
    TAG_DURATION = 'full_stage' if FULL_STAGE else f'first_{NAP_DURATION_MIN}min'
    PREPROC_TAG = 'preproc' if PREPROC else 'raw'
    FEAT_TAG = ''.join(f'_{name}' for name, on in [('emg', EMG), ('eog', EOG), ('ecg', ECG), ('behav', BEHAV)] if on)
    TAG = f'{MODEL_TAG}_{EXPERIMENT_STAGE}_{TAG_DURATION}_{PREPROC_TAG}{FEAT_TAG}'
    print(f'Tag: {TAG}')

    # Hypnogram: one row per scored 30-s epoch (08_sleep_stages only has the epochs that contain an event)
    EPOCH_S = 30
    sleep_stages = load_sleep_stages(pd.read_csv(GAMM_DIR / '12_hypnogram_epochs.csv'))  # subject -> {epoch_idx: stage}

    out_dir = OUT_DIR / TAG
    cache_file = out_dir / f'{TAG}.pkl'
    cached = pd.read_pickle(cache_file) if cache_file.exists() else None
    if isinstance(cached, dict) and cached.get('params') == params:
        print(f'Loading the cached data from {cache_file}')
        data = cached['data']
    else:
        if cached is not None:
            print(f'{cache_file} was computed with other parameters: recomputing')
        print('LOADING AND PREPROCESSING EEG DATA')
        os.makedirs(out_dir, exist_ok=True)
        log = get_logger(out_dir / f'{TAG}.log')
        log.info(f'Tag: {TAG}\nparams: {params}')
        fif_files = find_files(data_dir, '*_preprocessed.fif', VERBOSE, log=log)

        # Which raw 'Session N' is Nap2 differs between subjects (e.g. TT004 -> Session 7), so take the mapping from the GAMM tables
        session_map = pd.read_csv(GAMM_DIR / '09_presses_staged.csv')[['subname', 'session', 'session_raw']].drop_duplicates()
        stages_df = pd.read_csv(GAMM_DIR / '08_sleep_stages.csv')

        Y, indices, sessions_meta, fs, ch_names = load_eeg_data(fif_files, EXPERIMENT_STAGE, session_map, stages_df, FULL_STAGE, CROP_END_PROBE, NAP_DURATION_MIN, VERBOSE, log=log)

        if VERBOSE:
            print(f'Y shape: {Y.shape}  (n_samples_total x n_channels)')
            print(f'indices shape: {indices.shape}  (n_sessions x 2, {sessions_meta.subject.nunique()} subjects)')
            print(f'{len(ch_names)} EEG channels, fs = {fs} Hz')
            print(f'Y size in memory: {Y.nbytes / 1e9:.2f} GB')
        if log:
            log.info(f'Y shape: {Y.shape}  (n_samples_total x n_channels)')
            log.info(f'indices shape: {indices.shape}  (n_sessions x 2, {sessions_meta.subject.nunique()} subjects)')
            log.info(f'{len(ch_names)} EEG channels, fs = {fs} Hz')
            log.info(f'Y size in memory: {Y.nbytes / 1e9:.2f} GB')

        if SLEEP_STAGES:
            Y_prep = [band_power_envelopes(Y[a:b], fs, ch_names, REGIONS, BANDS, FS_ENV, ENV_SMOOTH_S) for a, b in indices]
            ends_prep = np.cumsum([x.shape[0] for x in Y_prep])
            indices_prep = np.column_stack([ends_prep - [x.shape[0] for x in Y_prep], ends_prep])
            Y_prep = np.vstack(Y_prep)
            # Standardise within each subject, over all its sessions: differences between two sessions of a subject are kept
            for sub in sessions_meta.subject.unique():
                rows = np.concatenate([np.arange(a, b) for a, b in indices_prep[(sessions_meta.subject == sub).values]])
                Y_prep[rows] = (Y_prep[rows] - Y_prep[rows].mean(axis=0)) / Y_prep[rows].std(axis=0)
            fs_prep = FS_ENV
            feature_names = [f'{r} {b}' for b in BANDS for r in REGIONS]
        elif PREPROC:
            Y_prep, indices_prep, prep_log = preproc.preprocess_data(data=Y, indices=indices, fs=fs,
                dampen_extreme_peaks=5,
                standardise=True)
            fs_prep, feature_names = fs, ch_names
        else:
            Y_prep, indices_prep, fs_prep, feature_names = Y, indices, fs, ch_names


        assert np.isfinite(Y_prep).all(), 'NaN/inf in Y_prep'

        if VERBOSE:
            print(f'Y_prep shape: {Y_prep.shape}  ({fs_prep} Hz, {len(feature_names)} features)')
            print(f'indices_prep shape: {indices_prep.shape}')
        if log:
            log.info(f'Y_prep shape: {Y_prep.shape}  ({fs_prep} Hz, {len(feature_names)} features)')
            log.info(f'indices_prep shape: {indices_prep.shape}')

        if EMG:
            # Load EMG signal if it exists
            emg_raw, _ = load_physiological_data(sessions_meta, 'EMG', fs, VERBOSE=VERBOSE, log=log)

            # Create an EMG-derived feature
            emg_feat = {s: emg_envelope(x, fs, EMG_BAND, FS_ENV, ENV_SMOOTH_S) for s, x in emg_raw.items()}
            for s, e in emg_feat.items():
                assert len(e) == indices_prep[s, 1] - indices_prep[s, 0], f'{sessions_meta.subject[s]}: EMG and EEG features have different lengths'

            verify_gaussianity(emg_feat, 'EMG log RMS', save_path=out_dir / f'{TAG}_emg_gaussianity.png', log=log, VERBOSE=VERBOSE)  # check that the EMG is Gaussian within each session, and per sleep stage

            # Per sleep stage: the EMG should drop from W to N1/N2 (muscle tone)
            if VERBOSE:
                print('\nMean z-scored EMG per subject and sleep stage:')
                print(mean_by_stage(emg_feat, 'EMG', sessions_meta, sleep_stages, FS_ENV, EPOCH_S))
            if log:
                log.info('\nMean z-scored EMG per subject and sleep stage:')
                log.info(mean_by_stage(emg_feat, 'EMG', sessions_meta, sleep_stages, FS_ENV, EPOCH_S))

            Y_prep, indices_prep, indices, sessions_meta, feature_names = add_feat_to_Y(
                Y_prep, indices_prep, indices, sessions_meta, feature_names, emg_feat, ['EMG'], 'no EMG channel', VERBOSE=VERBOSE, log=log)

        if EOG:
            eog_raw, _ = load_physiological_data(sessions_meta, 'IO', fs, VERBOSE=VERBOSE, log=log)  # IO: the EOG channel

            # session index -> (n_times_env, n_eog_features), one column per entry of EOG_FEATURES
            eog_feat = {s: np.column_stack([eog_envelope(x, fs, FS_ENV, band, smooth_s) for band, smooth_s in EOG_FEATURES.values()])
                        for s, x in eog_raw.items()}
            for s, e in eog_feat.items():
                assert len(e) == indices_prep[s, 1] - indices_prep[s, 0], f'{sessions_meta.subject[s]}: EOG and EEG features have different lengths'
     
            verify_gaussianity(eog_feat, list(EOG_FEATURES), save_path=out_dir / f'{TAG}_eog_gaussianity.png', log=log, VERBOSE=VERBOSE)

            # Per sleep stage: eye movements should become rarer from W to N2
            if VERBOSE:
                print('\nMean z-scored EOG per subject and sleep stage:')
                print(mean_by_stage(eog_feat, list(EOG_FEATURES), sessions_meta, sleep_stages, FS_ENV, EPOCH_S))
            if log:
                log.info('\nMean z-scored EOG per subject and sleep stage:')
                log.info(mean_by_stage(eog_feat, list(EOG_FEATURES), sessions_meta, sleep_stages, FS_ENV, EPOCH_S))

            Y_prep, indices_prep, indices, sessions_meta, feature_names = add_feat_to_Y(
            Y_prep, indices_prep, indices, sessions_meta, feature_names, eog_feat, EOG_FEATURES, 'no EOG channel', VERBOSE=VERBOSE, log=log)

        if ECG:
            # The .fif and the BrainVision recording share the same time axis, so the EEG window of each session is cropped directly
            ecg_raw, fs_ecg = load_physiological_data(sessions_meta, 'ECG', fs, vhdr_dir=data_dir, VERBOSE=VERBOSE, log=log)
            ecg_feat = {}  # session index (row of sessions_meta) -> (n_times_env, len(ECG_FEATURES)), only for the usable ECGs
            for s, x in ecg_raw.items():
                row = sessions_meta.loc[s]
                t = r_peaks(x, fs_ecg)
                kept = clean_rr(t)[2] if len(t) > 2 else 0.0
                if len(t) < len(x) / fs_ecg / 2 or kept < ECG_MIN_RR_KEPT:  # < 30 bpm detected, or too many beats missed / extra
                    if VERBOSE:
                        print(f"  {row.subject}: unusable ECG ({len(t)} beats, {kept:.0%} of RR intervals kept)")
                    if log:
                        log.info(f"  {row.subject}: unusable ECG ({len(t)} beats, {kept:.0%} of RR intervals kept)")
                    continue
                ecg_feat[s] = ecg_features(t, indices_prep[s, 1] - indices_prep[s, 0], FS_ENV, HR_SMOOTH_S, HRV_WIN_S, ECG_FEATURES)
                if VERBOSE:
                    print(f"  {row.subject}: {len(t)} beats, {kept:.0%} of RR intervals kept, median HR {60 / np.median(np.diff(t)):.0f} bpm")
                if log:
                    log.info(f"  {row.subject}: {len(t)} beats, {kept:.0%} of RR intervals kept, median HR {60 / np.median(np.diff(t)):.0f} bpm")
            if VERBOSE:
                print(f'usable ECG for {len(ecg_feat)} / {len(sessions_meta)} sessions')
            if log:
                log.info(f'usable ECG for {len(ecg_feat)} / {len(sessions_meta)} sessions')

            verify_gaussianity(ecg_feat, [f'log {name}' for name in ECG_FEATURES], save_path=out_dir / f'{TAG}_ecg_gaussianity.png', log=log, VERBOSE=VERBOSE)

            # Per sleep stage: the heart rate should drop and the RMSSD (vagal tone) rise from W to N2
            
            if VERBOSE:
                print('\nMean z-scored ECG features per subject and sleep stage:')
                print(mean_by_stage(ecg_feat, ECG_FEATURES, sessions_meta, sleep_stages, FS_ENV, EPOCH_S))
            if log:
                log.info('\nMean z-scored ECG features per subject and sleep stage:')
                log.info(mean_by_stage(ecg_feat, ECG_FEATURES, sessions_meta, sleep_stages, FS_ENV, EPOCH_S))
            Y_prep, indices_prep, indices, sessions_meta, feature_names = add_feat_to_Y(
                Y_prep, indices_prep, indices, sessions_meta, feature_names, ecg_feat, ECG_FEATURES, 'no usable ECG', VERBOSE=VERBOSE, log=log)

        if BEHAV:
            # t_start_s / t_end_s / t_in_session_s count from the stage's 'Start' event, which is crop_start_s: same origin as the EEG window
            intervals = pd.read_csv(GAMM_DIR / '02_intervalles.csv')
            intervals = intervals[(intervals.session == EXPERIMENT_STAGE) & intervals.valid]
            too_short = intervals.irt_s < MIN_IRT_S
            if too_short.any():
                if VERBOSE:
                    print(f'leaving out {too_short.sum()} IRT interval(s) shorter than {MIN_IRT_S} s:\n'
                        f"{intervals[too_short][['subname', 't_start_s', 'irt_s']].round(2).to_string(index=False)}")
                if log:
                    log.info(f'leaving out {too_short.sum()} IRT interval(s) shorter than {MIN_IRT_S} s:\n'
                        f"{intervals[too_short][['subname', 't_start_s', 'irt_s']].round(2).to_string(index=False)}")
            intervals = intervals[~too_short]
            presses = pd.read_csv(GAMM_DIR / '09_presses_staged.csv')
            presses = presses[presses.session == EXPERIMENT_STAGE]

            behav_feat = {}  # session index (row of sessions_meta) -> (n_times_env, len(BEHAV_FEATURES))
            for s, row in sessions_meta.iterrows():
                f = behav_features(row.subject, row.crop_start_s, indices_prep[s, 1] - indices_prep[s, 0], FS_ENV, intervals, presses, BEHAV_FEATURES, PRESS_RATE_SIGMA_S, stages_df)
                if f is None:
                    if VERBOSE:
                        print(f"  {row.subject}: fewer than 2 IRT intervals")
                    if log:
                        log.info(f"  {row.subject}: fewer than 2 IRT intervals")
                    continue
                behav_feat[s] = f
                if VERBOSE:
                    print(f"  {row.subject}: {(intervals.subname == row.subject).sum()} IRT intervals, {((presses.subname == row.subject) & (presses.event == 'Response')).sum()} responses, "
                    f"{((presses.subname == row.subject) & (presses.event == 'Reset')).sum()} resets")
                if log:
                    log.info(f"  {row.subject}: {(intervals.subname == row.subject).sum()} IRT intervals, {((presses.subname == row.subject) & (presses.event == 'Response')).sum()} responses, "
                    f"{((presses.subname == row.subject) & (presses.event == 'Reset')).sum()} resets")

            if VERBOSE: print(f'behavioural features for {len(behav_feat)} / {len(sessions_meta)} sessions')
            if log: log.info(f'behavioural features for {len(behav_feat)} / {len(sessions_meta)} sessions')
            verify_gaussianity(behav_feat, BEHAV_FEATURES, save_path=out_dir / f'{TAG}_behav_gaussianity.png', log=log, VERBOSE=VERBOSE)  # check that the behavioural features are Gaussian within each session, and per sleep stage

            # Per sleep stage: the press rates should drop when the participant falls asleep
            if VERBOSE:
                print('\nMean z-scored behavioural features per subject and sleep stage:')
                print(mean_by_stage(behav_feat, BEHAV_FEATURES, sessions_meta, sleep_stages, FS_ENV, EPOCH_S))
            if log:
                log.info('\nMean z-scored behavioural features per subject and sleep stage:')
                log.info(mean_by_stage(behav_feat, BEHAV_FEATURES, sessions_meta, sleep_stages, FS_ENV, EPOCH_S))

            Y_prep, indices_prep, indices, sessions_meta, feature_names = add_feat_to_Y(
                    Y_prep, indices_prep, indices, sessions_meta, feature_names, behav_feat, BEHAV_FEATURES, 'fewer than 2 IRT intervals', VERBOSE=VERBOSE, log=log)

        # Save the preprocessed data with its parameters, to avoid re-running the long load_eeg_data step
        data = (Y_prep, indices_prep, indices, sessions_meta, fs, fs_prep, ch_names, feature_names)
        pd.to_pickle({'params': params, 'data': data}, cache_file)
        print(f'Saved to {cache_file}')
    return data