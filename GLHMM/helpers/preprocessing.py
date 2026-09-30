"""
DOZE Full EEG preprocessing pipeline
=====================================
- Load raw EEG data (30s segments) from Full_EEG/raw
- Preprocess and save the data (filter, notch, reference, interpolate bad channels)
"""

import mne
import os
import ast
import re
import numpy as np
import pandas as pd

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
BASE_DIR        = r"C:\Users\eva\Documents\Master\ICM\Nico"
DIR_RAW_DATA    = r"C:\Users\eva\Documents\Master\ICM\Nico\Full_EEG\raw"
DIR_PREPROC     = r"C:\Users\eva\Documents\Master\ICM\Nico\Full_EEG\preproc"
QC_CSV          = r"C:\Users\Bruker\Documents\DOZE\Preprocessing\bad_segments_channels.csv" # TODO: modify path to QC CSV file


# Preprocessing
REMOVE_BAD_CHANNELS = False
L_FREQ     = 0.3
H_FREQ     = 40.0
NOTCH_FREQ = 50.0
REFERENCE  = "average"

# Channels that are physiological (not EEG) — excluded from power computation
PHYSIO_CHAN = [
    'IO', 'EMG chin', 'EMG arm', 'ECG',
    'Thumb', 'Index+Middle', 'Ring+Pinky', 'Respi',
]

TOTAL_EEG_CHANNELS = 64   # used to compute bad_channel_fraction
HIGH_BAD_THRESHOLD = 0.30  # fraction above which high_bad_channel_load = True
# ───────────────────────────────────────────────────────────────────────────────

os.makedirs(DIR_RAW_DATA, exist_ok=True)
os.makedirs(DIR_PREPROC, exist_ok=True)

# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_qc_table(qc_csv):
    """Load bad_segments_channels.csv, return dict keyed by filename."""
    if not os.path.exists(qc_csv):
        return {}
    qc = pd.read_csv(qc_csv)

    qc['bad_channels'] = qc['bad_channels'].apply(
        lambda x: ast.literal_eval(x)
        if isinstance(x, str) and x.strip().startswith('[') else []
    )

    qc['bad_segment'] = qc['bad_segment'].map(
        {True: True, False: False, 'True': True, 'False': False}
    ).fillna(False)

    return qc.set_index('filename').to_dict('index')

# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

subjects = sorted(
    d for d in os.listdir(DIR_RAW_DATA)
    if os.path.isdir(os.path.join(DIR_RAW_DATA, d))
)
print(f"Found {len(subjects)} participant folder(s): {subjects}\n")

needs_bad_ch_marking = [] 
if REMOVE_BAD_CHANNELS:
    print("IMPLEMENT BAD CHANNEL REMOVAL FUNCTIONALITY HERE\n")
    # TODO: Implement functionality to remove bad channels from the data based on QC table
        #     qc = qc_table[fname_30s]

        # if qc.get('bad_segment', False):
        #     print(f"    Probe {probe_idx:02d}: flagged as bad segment — skipping preprocessing")
        #     matches = df.index[df['segment_id'] == segment_id].tolist()
        #     if matches:
        #         df.at[matches[0], 'bad_segment'] = True
        #     continue
        #     bad_chs = qc.get('bad_channels', [])
        # if not isinstance(bad_chs, list):
        #     bad_chs = []

        # if bad_chs:
        #     available = [ch for ch in bad_chs if ch in raw_seg.ch_names]
        #     raw_seg.info['bads'] = available
        #     raw_seg.interpolate_bads(reset_bads=True, verbose=False)

        #     # Write QC columns to CSV
        # n_bad = len(bad_chs)
        # matches = df.index[df['segment_id'] == segment_id].tolist()
        # if matches:
        #     row_i = matches[0]
        #     df.at[row_i, 'bad_segment']          = qc.get('bad_segment', False)
        #     df.at[row_i, 'n_bad_channels']        = n_bad
        #     df.at[row_i, 'bad_channel_fraction']  = n_bad / TOTAL_EEG_CHANNELS
        #     df.at[row_i, 'high_bad_channel_load'] = (n_bad / TOTAL_EEG_CHANNELS) > HIGH_BAD_THRESHOLD
        #     df.at[row_i, 'bad_channels']          = str(bad_chs)

        # print(
        #     f"    Probe {probe_idx:02d}: preprocessed → {fname_10s} "
        #     f"({n_bad} channel(s) interpolated)"
        # )

for subj in subjects:
    print(f"\n  Preprocessing data")
    subj_missing_qc = []

    path_raw = os.path.join(DIR_RAW_DATA, subj, f"{subj}.vhdr")
    path_preproc = os.path.join(DIR_PREPROC, subj, f"{subj}.fif")
    os.makedirs(os.path.dirname(path_preproc), exist_ok=True)
    print(f"Subject's path: {path_raw}")
    print(f"Preprocessed data path: {path_preproc}")

    raw_seg = mne.io.read_raw_brainvision(path_raw, preload=True, verbose=False)
    eeg_picks = mne.pick_types(raw_seg.info, eeg=True, exclude=[])
    eeg_names = [raw_seg.ch_names[i] for i in eeg_picks]

    raw_seg.set_eeg_reference(REFERENCE, projection=False, verbose=False)
    raw_seg.notch_filter(NOTCH_FREQ, picks=eeg_names, verbose=False)
    raw_seg.filter(l_freq=L_FREQ, h_freq=H_FREQ, picks=eeg_names, verbose=False)
    raw_seg.save(path_preproc, overwrite=True, verbose=False)

    if subj_missing_qc:
        needs_bad_ch_marking.append((subj, subj_missing_qc))
