import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glhmm import glhmm, preproc, utils, graphics
import mne
from pathlib import Path

from scipy import signal
from scipy.ndimage import uniform_filter1d

########################################################################################################################
# LOAD EEG DATA
########################################################################################################################

def find_files(data_dir, pattern, VERBOSE=True):
    files = sorted(data_dir.glob(pattern))
    if VERBOSE: print(f"Found {len(files)} files: {[f.name for f in files]}")
    assert len(files) > 0, f"No files found in {data_dir} matching pattern {pattern}"
    return files

def load_eeg_data(
    fif_files,
    experiment_stage,
    session_map,
    stages_df,
    hyp_df,
    full_stage=True,
    crop_end_probe=True,
    nap_duration_min=30,
    verbose=True,
):
    """Load and concatenate the EEG windows belonging to one experiment stage."""
    stage_sessions = session_map[session_map.session == experiment_stage]
    data_raw = stage_sessions.set_index('subname').session_raw

    def event_times(subject, session, event_mask):
        rows = stages_df[
            (stages_df.subname == subject)
            & (stages_df.session == session)
            & event_mask
        ]
        return rows.onset_s

    start_s = {
        subject: event_times(subject, session, stages_df.event.str.startswith('Start')).min()
        for subject, session in data_raw.items()
    }
    end_s = {
        subject: event_times(subject, session, stages_df.event.str.startswith('End')).max()
        for subject, session in data_raw.items()
    }
    last_probe_s = {
        subject: event_times(
            subject,
            session,
            stages_df.event.str.contains('Probe') & (stages_df.event != 'End Probe'),
        ).max()
        for subject, session in data_raw.items()
    }

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
        if crop_end_probe and not np.isnan(last_probe_s.get(subject, np.nan)):
            crop_end_s = last_probe_s[subject]
        if not full_stage:
            crop_end_s = min(crop_end_s, crop_start_s + nap_duration_min * 60)
        raw.crop(tmin=crop_start_s, tmax=crop_end_s, include_tmax=False)

        raw.pick('eeg')
        fs = int(raw.info['sfreq'])
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
    if len({x.shape[1] for x in Y_list}) != 1:
        raise ValueError('Sessions have different channel counts')

    Y = np.vstack(Y_list)
    indices = np.column_stack([starts, ends])
    sessions_meta = pd.DataFrame(meta_rows)

    return Y, indices, sessions_meta