#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
════════════════════════════════════════════════════════════════════════════════
 Scoring_TickToz_V4.py — scoring manuel Response / Reset sur signaux de force
════════════════════════════════════════════════════════════════════════════════

USAGE
  1. Verifie DATA_ROOT ci-dessous.
  2. Lance :  python Scoring_TickToz_V4.py
  3. Tape l'ID du participant (ex: TT001), le nom du scoreur, la main REPONSE.
  4. Regarde la liste des sessions imprimee, puis ajuste
     ZERO_TIME_AT_START_INDEX si tu veux recadrer t=0.

ARBORESCENCE ATTENDUE
  D:/Ticktoz/data/
    ├── TT001/  (TT001.vhdr + TT001.eeg + TT001.vmrk)
    ├── TT002/
    └── Scoring_Pingouin/        <- TOUTES les sorties, prefixees TT001_...

SORTIES
  <sujet>_events.csv        LE fichier unique (scoring manuel + triggers)
  <sujet>_state.json        memoire de reprise
  <sujet>_signal_cache.npz  cache signal

COLONNES DU CSV
  subname, hand_response, session, event, source, channel, side,
  onset_s, onset_min, time_in_session_s, wall_time, onset_absolute,
  duration_s, delta_since_last_RR, irt, irt_roll, cv_irt_roll,
  streak_response, streak_reset, is_mirror, scorer, scored_at

  delta_since_last_RR : secondes depuis la derniere Response OU Reset
  irt                 : secondes depuis la derniere Response
  irt_roll/cv_irt_roll: moyenne et CV glissants (ROLLING_WINDOW reponses)
  streak_*            : evenements consecutifs du meme type (remis a 0
                        a chaque probe, chaque Break et chaque debut de session)

──────────────────────────── RACCOURCIS CLAVIER ────────────────────────────────
 SCORING
   clic-glisser gauche : creer une annotation
   clic droit          : supprimer l'annotation la plus proche
   1 / 2 / 3           : forcer Response / forcer Reset / label AUTO
   X                   : basculer Response<->Reset sous la souris
   Ctrl+Z / Ctrl+Y     : annuler / refaire (50 niveaux)

 NAVIGATION
   fleches G/D         : +/- 30 s        Page Up/Down : +/- une fenetre
   Shift+fleches       : +/- 1 s         Home / End   : debut / fin
   Ctrl+fleches        : +/- 5 s         molette      : +/- 5 s
   clic molette+glisser: pan libre
   W / E               : session precedente / suivante
   + / -               : zoom            0 : fenetre 30 s
   N                   : normaliser      A : auto-echelle Y sur la fenetre
   M                   : miroir ON/OFF

 FICHIERS
   Ctrl+S              : sauvegarder     P : marquer progression
   T                   : stats IRT/CV    V : export .vmrk
════════════════════════════════════════════════════════════════════════════════
"""

import os
import copy
import json
import warnings
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

warnings.filterwarnings('ignore', message='No coordinate information')
warnings.filterwarnings('ignore', message='Online software filter')
warnings.filterwarnings('ignore', message='Not setting positions')

import matplotlib
for _b in ('TkAgg', 'Qt5Agg', 'QtAgg', 'WXAgg'):
    try:
        matplotlib.use(_b)
        import matplotlib.pyplot as _plt_check
        _plt_check.figure()
        _plt_check.close('all')
        break
    except Exception:
        continue

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, TextBox

# ─────────────────────────────────────────────────────────────────────────────
# CRUCIAL : desactive TOUS les raccourcis clavier internes de matplotlib.
# Sans ca : 's' ouvre "enregistrer l'image", 'p' active le mode pan (le
# clic-glisser n'annote plus !), les fleches naviguent dans l'historique
# de vue, etc. -> c'est la cause principale des comportements erratiques.
# ─────────────────────────────────────────────────────────────────────────────
for _k in list(plt.rcParams):
    if _k.startswith('keymap.'):
        plt.rcParams[_k] = []
plt.rcParams['toolbar'] = 'None'    # supprime la barre d'outils (modes pan/zoom)

import mne
mne.set_log_level('WARNING')


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION   ←←← C'EST ICI QUE TU MODIFIES
# ══════════════════════════════════════════════════════════════════════════════

DATA_ROOT = r"C:\Users\eva\Documents\Master\ICM\Ilo"     # racine contenant TT001, TT002, ...
SUBJECT   = ""                     # vide -> demande au lancement
FILENAME: Optional[str] = None     # None -> "<SUBJECT>.vhdr", sinon auto

OUTPUT_SUBFOLDER = "Scoring_Pingouin"   # dossier unique dans DATA_ROOT

SCORER_NAME   = ""      # vide -> demande au lancement
RESPONSE_HAND = ""      # "R" ou "L" ; vide -> demande au lancement

# ── Sessions ──────────────────────────────────────────────────────────────────
# Le script MATLAB envoie 2 fois EVT.start d'affilee : deux START separes de
# moins de SESSION_MIN_GAP_S secondes sont fusionnes en UNE session.
SESSION_MIN_GAP_S = 5.0

# Noms des sessions dans l'ordre chronologique. Si le nombre de sessions
# detectees ne correspond pas, on retombe sur "Session 1", "Session 2", ...
SESSION_NAMES = ['Wake1', 'Nap1', 'Nap2', 'Wake2']

# Placer t=0 sur le N-ieme START (APRES dedoublonnage).
# None ou 1 -> pas de recadrage.
# >>> Lance une premiere fois avec None : le script imprime la liste numerotee
#     des sessions, tu choisis ensuite le bon numero. <<<
ZERO_TIME_AT_START_INDEX: Optional[int] = None

# ── Scoring ───────────────────────────────────────────────────────────────────
MIRROR_BY_DEFAULT    = True    # miroir gauche/droite actif au demarrage
SHOW_BTN_TRIGGERS    = True    # AFFICHER les triggers boutons S10/S11 sur les traces
INCLUDE_BTN_TRIGGERS = True    # les ECRIRE dans le CSV (independant de l'affichage)
ROLLING_WINDOW       = 5       # nb de reponses pour IRT/CV glissants
AUTOSAVE_EVERY       = 10      # sauvegarde auto toutes les N annotations (0=off)
MIN_ANNOT_DURATION   = 0.010   # duree minimale d'une annotation (s)

FORCE_CHANNELS: Optional[list] = None   # None = auto-detection

# ── Triggers ──────────────────────────────────────────────────────────────────
SESSION_START_DESC = 'Stimulus/S  1'
SESSION_END_DESC   = 'Stimulus/S  6'
PAUSE_DESC         = 'Stimulus/S  7'
RESUME_DESC        = 'Stimulus/S  8'
PROBE_END_DESC     = 'Stimulus/S  5'
RESP_BTN_DESC      = 'Stimulus/S 10'   # trigger bouton REPONSE (EEG brut)
RESET_BTN_DESC     = 'Stimulus/S 11'   # trigger bouton RESET   (EEG brut)

TRIGGER_MAP: dict = {
    'Stimulus/S  1':  ('START',        'session'),
    'Stimulus/S  2':  ('BEHAV probe',  'probe'),
    'Stimulus/S  3':  ('SCHED probe',  'probe'),
    'Stimulus/S  4':  ('MANUAL probe', 'probe'),
    'Stimulus/S  5':  ('end probe',    'probe'),
    'Stimulus/S  6':  ('END session',  'session'),
    'Stimulus/S  7':  ('PAUSE',        'session'),
    'Stimulus/S  8':  ('RESUME',       'session'),
    'Stimulus/S  9':  ('CORRECT/undo', 'response'),
    'Stimulus/S 10':  ('resp BTN',     'response'),
    'Stimulus/S 11':  ('reset BTN',    'response'),
    'Stimulus/S 12':  ('resp probe',   'probe'),
    'Stimulus/S 13':  ('reset probe',  'probe'),
    'Stimulus/S 14':  ('safety probe', 'probe'),
}

# Noms "propres" ecrits dans la colonne 'event' du CSV
EVENT_NAMES = {
    'Stimulus/S  2': 'No Rep Probe',
    'Stimulus/S  3': 'Sched Probe',
    'Stimulus/S  4': 'Manual Probe',
    'Stimulus/S  5': 'End Probe',
    'Stimulus/S  7': 'Break',
    'Stimulus/S  8': 'EndBreak',
    'Stimulus/S  9': 'Correct/Undo',
    'Stimulus/S 10': 'trigger resp BTN',
    'Stimulus/S 11': 'trigger reset BTN',
    'Stimulus/S 12': 'Rep Probe',
    'Stimulus/S 13': 'Reset Probe',
    'Stimulus/S 14': 'Safety Probe',
}
PROBE_EVENTS = ('No Rep Probe', 'Sched Probe', 'Manual Probe',
                'Rep Probe', 'Reset Probe', 'Safety Probe')

WINDOW_DURATION = 30.0
STEP_SIZE       = 30.0
MAX_DISP_POINTS = 5000

# ══════════════════════════════════════════════════════════════════════════════

C = {
    'bg_fig': '#1a1a2e', 'bg_ax': '#16213e', 'spine': '#0f3460',
    'grid': '#1a2e50', 'text': '#dde6ed',
    'signals': ['#00d4ff', '#ff6b6b', '#51cf66', '#ffd43b', '#c084fc'],
    'ev_resp_fc': '#ffd700', 'ev_resp_ec': '#ff8c00',
    'ev_reset_fc': '#9775fa', 'ev_reset_ec': '#7048e8',
    'mirror_fc': '#b8ffd9', 'mirror_ec': '#2ed573',
    'auto_fc': '#ffb3e6', 'auto_ec': '#ff2fb8',
    'trg_session': '#ff4757', 'trg_probe': '#2ed573',
    'trg_response': '#ffa502', 'trg_other': '#a4b0be',
    'trg_respbtn': '#ffa502', 'trg_resetbtn': '#b197fc',
    'btn_nav': '#0f3460', 'btn_nav1': '#1a3d5c', 'btn_sess': '#3d1a5c',
    'btn_undo': '#5c1a1a', 'btn_redo': '#1a2c4a', 'btn_save': '#1a5c1a',
    'btn_zoom': '#2a2a4e', 'btn_mir_on': '#1a4c3a', 'btn_mir_off': '#4c2a1a',
    'btn_prog': '#1a3d5c', 'btn_stats': '#2a3d5c', 'btn_resp': '#5c4a00',
    'btn_reset': '#3a2a5c', 'btn_auto': '#26324a', 'btn_vmrk': '#1a3050',
}


def _lighten(hx: str, a: float = 0.20) -> str:
    r = min(1.0, int(hx[1:3], 16) / 255 + a)
    g = min(1.0, int(hx[3:5], 16) / 255 + a)
    b = min(1.0, int(hx[5:7], 16) / 255 + a)
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'


def _trig_label(desc: str) -> str:
    return TRIGGER_MAP.get(desc, (desc, 'other'))[0]


def _trig_color(desc: str) -> str:
    if desc == RESP_BTN_DESC:
        return C['trg_respbtn']
    if desc == RESET_BTN_DESC:
        return C['trg_resetbtn']
    cat = TRIGGER_MAP.get(desc, ('?', 'other'))[1]
    return {'session': C['trg_session'], 'probe': C['trg_probe'],
            'response': C['trg_response']}.get(cat, C['trg_other'])


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def _load_triggers_df(raw) -> pd.DataFrame:
    if len(raw.annotations) == 0:
        return pd.DataFrame(columns=['onset', 'duration', 'description',
                                     'label', 'category'])
    df = pd.DataFrame({
        'onset':       raw.annotations.onset.tolist(),
        'duration':    raw.annotations.duration.tolist(),
        'description': raw.annotations.description.tolist(),
    }).sort_values('onset').reset_index(drop=True)
    df['label']    = df['description'].map(_trig_label)
    df['category'] = df['description'].map(
        lambda d: TRIGGER_MAP.get(d, (d, 'other'))[1])
    return df


def _print_triggers(triggers: pd.DataFrame):
    if triggers.empty:
        print("  Aucun trigger dans ce fichier.\n")
        return
    print(f"\n  Triggers : {len(triggers)} au total")
    for d in sorted(set(triggers['description'])):
        n = int((triggers['description'] == d).sum())
        print(f"    [{n:3d}x]  '{d}'  ->  '{_trig_label(d)}'")
    print()


def _detect_force_channels(raw) -> list:
    kw = ['force', 'grip', 'fsr', 'pressure', 'index', 'ring', 'pinky',
          'thumb', 'pouce', 'doigt', 'left', 'right', 'gauche', 'droit',
          'hand', 'capteur', 'squeeze', 'middle']
    skip = {'ECG', 'Respi', 'IO', 'EMG'}
    found = [c for c in raw.ch_names
             if any(k in c.lower() for k in kw) and c not in skip]
    if not found:
        found = [c for i, c in enumerate(raw.ch_names)
                 if raw.get_channel_types(picks=[i])[0] in ('misc', 'bio')
                 and c not in skip]
    if found:
        print(f"  Canaux de force detectes : {found}")
    return found


def inspect_and_load(filepath: str, force_chs_cfg: Optional[list]):
    """Retourne (raw, ch_orig_idx, triggers, channels)."""
    print(f"\n{'='*70}")
    print(f"  Chargement : {Path(filepath).name}")
    print(f"{'='*70}")

    raw   = mne.io.read_raw_brainvision(filepath, preload=False)
    names = list(raw.ch_names)
    sfreq = float(raw.info['sfreq'])
    print(f"  Duree : {raw.times[-1]:.1f}s  ({raw.times[-1]/60:.1f} min)"
          f"   |   Fs : {sfreq} Hz   |   {len(names)} canaux")

    triggers = _load_triggers_df(raw)
    _print_triggers(triggers)

    chs = ([c for c in force_chs_cfg if c in names] if force_chs_cfg
           else _detect_force_channels(raw))
    if not chs:
        return None, {}, triggers, []

    ch_orig_idx = {c: names.index(c) for c in chs}
    print(f"  Chargement des donnees ({len(chs)} canaux de force)...")
    raw.pick(chs)
    raw.load_data()
    print("  OK\n")
    return raw, ch_orig_idx, triggers, chs


# ══════════════════════════════════════════════════════════════════════════════
# SESSIONS
# ══════════════════════════════════════════════════════════════════════════════

def dedup_session_starts(triggers: pd.DataFrame,
                         min_gap: float = SESSION_MIN_GAP_S) -> list:
    """Onsets des START, doublons rapproches fusionnes."""
    if triggers.empty:
        return []
    starts = (triggers.loc[triggers['description'] == SESSION_START_DESC,
                           'onset'].sort_values().tolist())
    kept, last = [], -1e18
    for t in starts:
        if float(t) - last >= min_gap:
            kept.append(float(t))
            last = float(t)
    return kept


def build_session_table(triggers: pd.DataFrame) -> pd.DataFrame:
    onsets = dedup_session_starts(triggers)
    if not onsets:
        return pd.DataFrame(columns=['onset', 'session_num', 'session_label'])
    if len(onsets) == len(SESSION_NAMES):
        labels = list(SESSION_NAMES)
    else:
        labels = [f"Session {i+1}" for i in range(len(onsets))]
        print(f"  [!] {len(onsets)} session(s) detectee(s) mais "
              f"{len(SESSION_NAMES)} nom(s) dans SESSION_NAMES "
              f"-> libelles generiques 'Session N'.")
    return pd.DataFrame({'onset': onsets,
                         'session_num': range(1, len(onsets) + 1),
                         'session_label': labels})


def get_session(t: float, sess_trg: pd.DataFrame) -> str:
    if sess_trg.empty:
        return 'N/A'
    before = sess_trg[sess_trg['onset'] <= float(t) + 1e-9]
    return 'pre-session' if before.empty else str(before.iloc[-1]['session_label'])


def get_session_start(t: float, sess_trg: pd.DataFrame) -> float:
    if sess_trg.empty:
        return 0.0
    before = sess_trg[sess_trg['onset'] <= float(t) + 1e-9]
    return float(before.iloc[-1]['onset']) if not before.empty else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# CACHE NPZ
# ══════════════════════════════════════════════════════════════════════════════

def _cache_path(out_dir: str, stem: str) -> str:
    return os.path.join(out_dir, f"{stem}_signal_cache.npz")


def try_load_cache(out_dir, stem, channels, sfreq, expected_offset=0.0):
    path = _cache_path(out_dir, stem)
    if not os.path.exists(path):
        return None, None
    try:
        cache = np.load(path, allow_pickle=True)
        if (list(cache['channels']) != list(channels)
                or not np.isclose(float(cache['sfreq']), sfreq)):
            print("  Cache NPZ incompatible -> ignore.")
            return None, None
        cached_off = float(cache['time_offset']) if 'time_offset' in cache else 0.0
        if not np.isclose(cached_off, expected_offset):
            print(f"  Cache NPZ obsolete (offset {cached_off:.3f}s != "
                  f"{expected_offset:.3f}s) -> ignore.")
            return None, None
        print("  Signal charge depuis le cache NPZ (rapide !)")
        return {ch: cache[ch] for ch in channels}, cache['times']
    except Exception as e:
        print(f"  Cache NPZ invalide ({e}) -> rechargement brut.")
        return None, None


def save_signal_cache(out_dir, stem, channels, signals, times, sfreq,
                      time_offset=0.0):
    try:
        np.savez_compressed(
            _cache_path(out_dir, stem), times=times,
            sfreq=np.array([sfreq]), channels=np.array(channels),
            time_offset=np.array([time_offset]),
            **{ch: signals[ch] for ch in channels})
    except Exception as e:
        print(f"  [!] Erreur cache NPZ : {e}")


# ══════════════════════════════════════════════════════════════════════════════
# RECADRAGE DU TEMPS ZERO
# ══════════════════════════════════════════════════════════════════════════════

def find_zero_offset(triggers: pd.DataFrame, n: Optional[int]) -> float:
    onsets = dedup_session_starts(triggers)
    print(f"  {'─'*70}")
    print(f"  Sessions detectees (temps ABSOLU du fichier brut, "
          f"apres dedoublonnage) :")
    if not onsets:
        print("    (aucune)")
    for i, t in enumerate(onsets, 1):
        name = (SESSION_NAMES[i-1] if len(onsets) == len(SESSION_NAMES)
                else f"Session {i}")
        mark = '   <<<< t = 0 ici' if (n and i == n) else ''
        print(f"    [{i}]  {name:<8}  t = {t:9.3f}s   ({t/60:7.2f} min){mark}")
    print(f"  {'─'*70}")
    if not n or n <= 1:
        if not n:
            print("  ZERO_TIME_AT_START_INDEX = None -> aucun recadrage "
                  "(t=0 = debut du fichier).")
        return 0.0
    if len(onsets) < n:
        print(f"  [!] ZERO_TIME_AT_START_INDEX={n} mais seulement "
              f"{len(onsets)} session(s) -> aucun recadrage.")
        return 0.0
    return float(onsets[n - 1])


def shift_triggers(triggers: pd.DataFrame, off: float) -> pd.DataFrame:
    if not off:
        return triggers
    out = triggers[triggers['onset'] >= off - 1e-9].copy()
    out['onset'] = out['onset'] - off
    return out.reset_index(drop=True)


def shift_signals(signals: dict, times: np.ndarray, off: float):
    if not off:
        return signals, times
    i0 = int(np.searchsorted(times, off))
    print(f"\n  Temps zero recadre sur t_original = {off:.3f}s "
          f"({i0} echantillons retires de l'affichage).")
    print("  (onset_absolute et wall_time restent alignes sur le "
          "fichier .eeg brut.)\n")
    return {c: s[i0:] for c, s in signals.items()}, times[i0:] - off


# ══════════════════════════════════════════════════════════════════════════════
# ANNOTATEUR
# ══════════════════════════════════════════════════════════════════════════════

class ForceAnnotator:

    def __init__(self, signals, times, sfreq, channels, ch_orig_idx,
                 triggers, out_dir, stem, subname, scorer, response_hand,
                 eeg_name=None, time_offset=0.0, meas_date=None,
                 headless=False):

        self.signals, self.times, self.sfreq = signals, times, sfreq
        self.channels, self.n_ch = list(channels), len(channels)
        self.ch_orig_idx = ch_orig_idx
        self.dur, self.n_samp = float(times[-1]), len(times)
        self.out_dir, self.stem = out_dir, stem
        self.subname, self.scorer = subname, scorer
        self.hand = (response_hand or 'R').upper().strip()
        self.eeg_name = eeg_name or f"{stem}.eeg"
        self.time_offset, self.meas_date = float(time_offset), meas_date
        self.headless = headless

        self.csv_path   = os.path.join(out_dir, f"{stem}_events.csv")
        self.state_path = os.path.join(out_dir, f"{stem}_state.json")
        self.vmrk_path  = os.path.join(out_dir, f"{stem}_scoring.vmrk")
        self.auto_path  = os.path.join(out_dir, f"{stem}_auto_presses.csv")

        # ── Sessions ─────────────────────────────────────────────────────────
        self.triggers = triggers
        self.sess_trg = build_session_table(triggers)
        if not self.sess_trg.empty:
            print(f"\n  Sessions utilisees pour le scoring (temps recadre) :")
            ends = triggers[triggers['description'] == SESSION_END_DESC]
            for _, r in self.sess_trg.iterrows():
                nxt = ends[ends['onset'] > r['onset']]
                e = (f"  ->  END t = {nxt.iloc[0]['onset']:.1f}s"
                     if len(nxt) else "")
                print(f"    {r['session_label']:<10} START t = "
                      f"{r['onset']:9.3f}s  ({r['onset']/60:6.2f} min){e}")
        else:
            print("  [!] Aucune session detectee "
                  "(verifie SESSION_START_DESC).")
        print()

        # ── Diagnostic signaux ───────────────────────────────────────────────
        self.signals_norm, self.ylims_raw = {}, {}
        print(f"  {'─'*70}")
        print(f"  Diagnostic des signaux (min / max / std / p1-p99)")
        print(f"  {'─'*70}")
        for ch in self.channels:
            d = signals[ch]
            mn, mx = float(d.min()), float(d.max())
            p1, p99 = np.percentile(d, [1, 99])
            amp = max(mx - mn, 1e-9)
            pad = 0.02 * amp
            self.ylims_raw[ch] = (mn - pad, mx + pad)
            self.signals_norm[ch] = (d - mn) / amp
            print(f"    {ch:<22} min={mn:9.4g}  max={mx:9.4g}"
                  f"  std={d.std():9.4g}  p1-p99=[{p1:.4g}, {p99:.4g}]")
            if (p99 - p1) < 0.05 * amp:
                print("       (pic/artefact isole probable -> l'auto-echelle "
                      "est activee par defaut, touche A)")
        print(f"  {'─'*70}\n")

        self.ylims         = dict(self.ylims_raw)
        self.normalized    = False
        self.autoscale_win = True
        self.mirror_mode   = MIRROR_BY_DEFAULT
        self.show_btn      = SHOW_BTN_TRIGGERS
        self.label_mode    = 'auto'      # 'auto' | 'Response' | 'Reset'
        self.show_auto     = True        # affiche les detections automatiques (rose)
        self.ch_color = {ch: C['signals'][i % len(C['signals'])]
                         for i, ch in enumerate(self.channels)}

        # ── Annotations + etat ───────────────────────────────────────────────
        self.annotations: list = []
        self._load_csv()
        self._load_auto_presses()
        self._n_since_save = 0
        self._history: deque = deque(maxlen=50)
        self._future:  deque = deque(maxlen=50)

        self.t_start, self.win_dur = 0.0, WINDOW_DURATION
        self.progress_t = 0.0
        self._drag  = None
        self._pan   = None
        self._mouse = (None, None)
        self._load_state()

        if headless:
            print(f"\n  [headless] regeneration du CSV pour {stem}")
            self._save()
            return

        self._build_ui()
        self._refresh()
        print(f"\n  Pret !  Scoreur : {self.scorer}  |  Main REPONSE : "
              f"{self.hand}  |  Label : AUTO")
        print("  Clic-glisser = annoter  |  Clic droit = supprimer  |  "
              "1/2/3 = Response/Reset/auto  |  Ctrl+S = sauver\n")
        plt.show()

    # ─────────────────────────────────────────────────────────── cote / label
    @staticmethod
    def _side(ch: str) -> str:
        c = str(ch).lower()
        if 'left' in c or 'gauche' in c:
            return 'left'
        if 'right' in c or 'droit' in c:
            return 'right'
        return 'none'

    def _side_channels(self, ch: str) -> list:
        s = self._side(ch)
        return [ch] if s == 'none' else [c for c in self.channels
                                         if self._side(c) == s]

    def _label_for(self, ch: str) -> str:
        if self.label_mode in ('Response', 'Reset'):
            return self.label_mode
        side = self._side(ch)
        if side == 'none':
            return 'Response'
        resp_side = 'right' if self.hand == 'R' else 'left'
        return 'Response' if side == resp_side else 'Reset'

    # ─────────────────────────────────────────────────────────────────── UI
    def _build_ui(self):
        hr = [5] * self.n_ch + [2.0, 0.5, 0.5]
        self.fig = plt.figure(figsize=(20, max(9.5, 2.0 * self.n_ch + 5.5)))
        self.fig.patch.set_facecolor(C['bg_fig'])
        self._set_title()

        gs = gridspec.GridSpec(len(hr), 1, height_ratios=hr, hspace=0.08,
                               left=0.07, right=0.985, top=0.955, bottom=0.17)
        self.ax_sig = []
        for i in range(self.n_ch):
            ax = self.fig.add_subplot(gs[i])
            self._style(ax)
            self.ax_sig.append(ax)
        self.ax_ov = self.fig.add_subplot(gs[self.n_ch])
        self._style(self.ax_ov)
        for j in (self.n_ch + 1, self.n_ch + 2):
            self.fig.add_subplot(gs[j]).set_visible(False)

        self._btns = []          # garde les references : sinon boutons morts !
        self._make_buttons()
        self._make_legend()

        self.txt_status = self.fig.text(
            0.50, 0.004, '', color=C['text'], fontsize=8,
            ha='center', va='bottom', family='monospace')

        cv = self.fig.canvas
        cv.mpl_connect('key_press_event',      self._on_key)
        cv.mpl_connect('button_press_event',   self._on_press)
        cv.mpl_connect('motion_notify_event',  self._on_move)
        cv.mpl_connect('button_release_event', self._on_release)
        cv.mpl_connect('scroll_event',         self._on_scroll)
        cv.mpl_connect('close_event',          self._on_close)

    def _set_title(self):
        mir = 'Miroir ON' if self.mirror_mode else 'Miroir OFF'
        lbl = 'AUTO' if self.label_mode == 'auto' else f"{self.label_mode} force"
        self.fig.suptitle(
            f"TickToz Scorer  |  {self.subname}  |  Scoreur : {self.scorer}"
            f"  |  Main REPONSE : {self.hand}  |  Label : {lbl}  |  {mir}",
            color=C['text'], fontsize=9, fontweight='bold', y=0.997)

    def _style(self, ax):
        ax.set_facecolor(C['bg_ax'])
        ax.tick_params(colors=C['text'], labelsize=7, length=3)
        for sp in ax.spines.values():
            sp.set_edgecolor(C['spine'])
        ax.grid(True, color=C['grid'], lw=0.4, alpha=0.7, zorder=0)

    def _mk(self, x, w, lbl, col, cb, row):
        axb = self.fig.add_axes([x, row, w, 0.032])
        b = Button(axb, lbl, color=col, hovercolor=_lighten(col, 0.20))
        b.label.set_color(C['text'])
        b.label.set_fontsize(7.5)
        b.on_clicked(cb)
        self._btns.append(b)
        return b

    def _make_buttons(self):
        r1, r2, r3 = 0.122, 0.080, 0.038
        mk = self._mk
        # Rangee 1 : navigation temporelle
        mk(0.070, 0.042, '<<60s', C['btn_nav'],  lambda e: self._nav(-60), r1)
        mk(0.115, 0.042, '<<30s', C['btn_nav'],  lambda e: self._nav(-30), r1)
        mk(0.160, 0.038, '<<10s', C['btn_nav'],  lambda e: self._nav(-10), r1)
        mk(0.201, 0.032, '<<5s',  C['btn_nav1'], lambda e: self._nav(-5),  r1)
        mk(0.236, 0.030, '<<1s',  C['btn_nav1'], lambda e: self._nav(-1),  r1)
        mk(0.269, 0.030, '1s>>',  C['btn_nav1'], lambda e: self._nav(+1),  r1)
        mk(0.302, 0.032, '5s>>',  C['btn_nav1'], lambda e: self._nav(+5),  r1)
        mk(0.337, 0.038, '10s>>', C['btn_nav'],  lambda e: self._nav(+10), r1)
        mk(0.378, 0.042, '30s>>', C['btn_nav'],  lambda e: self._nav(+30), r1)
        mk(0.423, 0.042, '60s>>', C['btn_nav'],  lambda e: self._nav(+60), r1)
        mk(0.470, 0.038, '|<',    C['btn_nav'],
           lambda e: self._set_tstart(0.0), r1)
        mk(0.511, 0.038, '>|',    C['btn_nav'],
           lambda e: self._set_tstart(self.dur - self.win_dur), r1)
        ax_tb = self.fig.add_axes([0.575, r1, 0.070, 0.032])
        ax_tb.set_facecolor('#2a2a4e')
        self.tb = TextBox(ax_tb, 'Aller a (s):', initial='0',
                          color='#2a2a4e', hovercolor='#3a3a6e')
        self.tb.label.set_color(C['text'])
        self.tb.label.set_fontsize(7)
        self.tb.text_disp.set_color(C['text'])
        self.tb.on_submit(self._jump_to)

        # Rangee 2 : sessions / zoom / affichage / fichiers
        mk(0.070, 0.058, '<Sess [W]', C['btn_sess'],
           lambda e: self._prev_session(), r2)
        mk(0.131, 0.058, 'Sess> [E]', C['btn_sess'],
           lambda e: self._next_session(), r2)
        mk(0.196, 0.040, 'Zoom+', C['btn_zoom'], lambda e: self._zoom(0.5), r2)
        mk(0.239, 0.040, 'Zoom-', C['btn_zoom'], lambda e: self._zoom(2.0), r2)
        mk(0.282, 0.050, 'Fen.30s', C['btn_zoom'],
           lambda e: self._zoom_reset(), r2)
        mk(0.335, 0.055, 'Normalis', C['btn_zoom'],
           lambda e: self._toggle_norm(), r2)
        mk(0.393, 0.058, 'AutoEch', C['btn_zoom'],
           lambda e: self._toggle_autoscale(), r2)
        self.b_mirror = mk(
            0.454, 0.070,
            'Miroir:ON' if self.mirror_mode else 'Miroir:OFF',
            C['btn_mir_on'] if self.mirror_mode else C['btn_mir_off'],
            lambda e: self._toggle_mirror(), r2)
        mk(0.530, 0.046, 'Undo', C['btn_undo'], lambda e: self._undo(), r2)
        mk(0.579, 0.046, 'Redo', C['btn_redo'], lambda e: self._redo(), r2)
        mk(0.630, 0.052, 'Marq[P]', C['btn_prog'],
           lambda e: self._set_progress_marker(), r2)
        mk(0.685, 0.050, 'Stats[T]', C['btn_stats'],
           lambda e: self._show_stats(), r2)
        self.b_btn = mk(
            0.740, 0.050,
            'BTN:ON' if self.show_btn else 'BTN:OFF',
            C['btn_mir_on'] if self.show_btn else C['btn_mir_off'],
            lambda e: self._toggle_btn_triggers(), r2)
        mk(0.795, 0.095, 'SAUVEGARDER', C['btn_save'],
           lambda e: self._save(), r2)
        mk(0.895, 0.070, '-> .vmrk', C['btn_vmrk'],
           lambda e: self._export_vmrk(), r2)

        # Rangee 3 : mode de label
        self.b_lab_r = mk(0.070, 0.085, '[1] Response', C['btn_resp'],
                          lambda e: self._set_label_mode('Response'), r3)
        self.b_lab_e = mk(0.158, 0.085, '[2] Reset', C['btn_reset'],
                          lambda e: self._set_label_mode('Reset'), r3)
        self.b_lab_a = mk(0.246, 0.085, '[3] AUTO', C['btn_auto'],
                          lambda e: self._set_label_mode('auto'), r3)
        mk(0.336, 0.105, '[X] R<->E souris', C['btn_auto'],
           lambda e: self._toggle_nearest_label(), r3)
        self.b_auto = mk(
            0.460, 0.085,
            'Auto:ON' if self.show_auto else 'Auto:OFF',
            C['btn_mir_on'] if self.show_auto else C['btn_mir_off'],
            lambda e: self._toggle_auto_presses(), r3)

    def _make_legend(self):
        items = [('-- Session', C['trg_session']), ('-- Probe', C['trg_probe']),
                 ('| respBTN', C['trg_respbtn']),
                 ('| resetBTN', C['trg_resetbtn']),
                 ('** Response', C['ev_resp_ec']),
                 ('** Reset', C['ev_reset_ec']),
                 ('** Miroir', C['mirror_ec']),
                 ('** Auto [D]', C['auto_ec'])]
        self.fig.text(0.455, 0.048, 'Legende :', color=C['text'],
                      fontsize=7, va='center', fontweight='bold')
        for i, (lbl, col) in enumerate(items):
            self.fig.text(0.510 + i * 0.058, 0.048, lbl, color=col,
                          fontsize=7, va='center', fontweight='bold')

    # ────────────────────────────────────────────────────────────── affichage
    def _grid_step(self) -> float:
        for th, st in [(10, 1), (30, 5), (60, 10), (120, 20), (300, 30)]:
            if self.win_dur <= th:
                return float(st)
        return 60.0

    def _ann_colors(self, ann):
        if ann.get('is_mirror', False):
            return C['mirror_fc'], C['mirror_ec']
        if ann.get('event') == 'Reset':
            return C['ev_reset_fc'], C['ev_reset_ec']
        return C['ev_resp_fc'], C['ev_resp_ec']

    def _refresh(self):
        t0 = self.t_start
        t1 = min(t0 + self.win_dur, self.dur)
        i0 = max(0, int(t0 * self.sfreq))
        i1 = min(self.n_samp, int(t1 * self.sfreq) + 1)
        tw = self.times[i0:i1]
        ds = max(1, len(tw) // MAX_DISP_POINTS)

        trg_vis = (self.triggers[(self.triggers['onset'] >= t0 - 0.5) &
                                 (self.triggers['onset'] <= t1 + 0.5)]
                   if not self.triggers.empty else pd.DataFrame())

        for ax, ch in zip(self.ax_sig, self.channels):
            ax.clear()
            self._style(ax)
            sig = (self.signals_norm[ch][i0:i1] if self.normalized
                   else self.signals[ch][i0:i1])

            if self.autoscale_win and not self.normalized and len(sig):
                mn, mx = float(np.min(sig)), float(np.max(sig))
                pad = 0.10 * max(mx - mn, 1e-9)
                ylo, yhi = mn - pad, mx + pad
            else:
                ylo, yhi = self.ylims[ch]
            if yhi <= ylo:
                yhi = ylo + 1e-6

            if len(tw):
                ax.plot(tw[::ds], sig[::ds], color=self.ch_color[ch], lw=0.85,
                        alpha=0.95, rasterized=True, zorder=2)

            for _, trg in trg_vis.iterrows():
                d      = trg['description']
                cat    = TRIGGER_MAP.get(d, ('?', 'other'))[1]
                col    = _trig_color(d)
                is_s   = (cat == 'session')
                is_btn = d in (RESP_BTN_DESC, RESET_BTN_DESC)

                if is_btn and not self.show_btn:
                    continue

                if is_btn:
                    # Triggers boutons EEG : trait plein + repere en BAS de la
                    # trace (les annotations manuelles sont etiquetees en haut,
                    # donc aucune collision), affiche sur TOUS les canaux.
                    ax.axvline(trg['onset'], color=col, lw=1.3, alpha=0.85,
                               ls='-', zorder=3.5)
                    ax.plot([trg['onset']], [ylo + 0.04 * (yhi - ylo)],
                            marker='^', color=col, markersize=4.5,
                            markeredgecolor='none', zorder=6)
                    ax.text(trg['onset'], ylo + 0.10 * (yhi - ylo),
                            'respBTN' if d == RESP_BTN_DESC else 'resetBTN',
                            color=col, fontsize=5, rotation=90, ha='center',
                            va='bottom', zorder=6,
                            bbox=dict(boxstyle='round,pad=0.1',
                                      fc=C['bg_fig'], ec='none', alpha=0.65))
                    continue

                ax.axvline(trg['onset'], color=col, lw=1.8 if is_s else 0.8,
                           alpha=0.95 if is_s else 0.5, ls='--', zorder=4)
                if ch == self.channels[0] or is_s:
                    ax.text(trg['onset'], yhi - 0.01 * (yhi - ylo),
                            EVENT_NAMES.get(d, _trig_label(d)),
                            color=col, fontsize=5, rotation=90, ha='right',
                            va='top', zorder=5,
                            bbox=dict(boxstyle='round,pad=0.1',
                                      fc=C['bg_fig'], ec='none', alpha=0.6))

            if self.show_auto:
                for p in self.auto_presses:
                    if p['channel'] != ch:
                        continue
                    on, off = p['onset'], p['offset']
                    if off < t0 or on > t1:
                        continue
                    ax.axvspan(on, off, fc=C['auto_fc'], ec=C['auto_ec'],
                               alpha=0.35, lw=0.9, zorder=2.7)

            for ann in self.annotations:
                if ann['channel'] != ch:
                    continue
                on, off = ann['onset'], ann['offset']
                if off < t0 or on > t1:
                    continue
                fc, ec = self._ann_colors(ann)
                ax.axvspan(on, off, fc=fc, ec=ec, alpha=0.35, lw=0.9, zorder=3)
                cx  = float(np.clip((on + off) / 2, t0, t1))
                tag = 'R' if ann.get('event') == 'Response' else 'E'
                ax.text(cx, yhi - 0.05 * (yhi - ylo),
                        f"{tag} {ann['duration']:.2f}s", color=ec, fontsize=6,
                        ha='center', va='top', fontweight='bold', zorder=4,
                        bbox=dict(boxstyle='round,pad=0.1', fc=C['bg_fig'],
                                  ec='none', alpha=0.7))

            ax.set_xlim(t0, t1)
            ax.set_ylim(ylo, yhi)
            ax.set_ylabel(ch, color=self.ch_color[ch], fontsize=8,
                          fontweight='bold', labelpad=3)
            step = self._grid_step()
            ax.xaxis.set_major_locator(mticker.MultipleLocator(step))
            ax.xaxis.set_minor_locator(mticker.MultipleLocator(step / 5))
            ax.tick_params(axis='x', which='minor', length=2, color=C['grid'])
            if ch != self.channels[-1]:
                plt.setp(ax.get_xticklabels(), visible=False)

        self.ax_sig[-1].set_xlabel('Temps (s)', color=C['text'], fontsize=8)
        self._refresh_overview(t0, t1)
        self._update_status(t0, t1)
        self.fig.canvas.draw_idle()

    def _refresh_overview(self, t0, t1):
        ax, ch0 = self.ax_ov, self.channels[0]
        ax.clear()
        self._style(ax)
        ds = max(1, self.n_samp // 4000)
        ax.plot(self.times[::ds], self.signals[ch0][::ds],
                color=self.ch_color[ch0], lw=0.4, alpha=0.8, rasterized=True)
        ylo, yhi = self.ylims_raw[ch0]

        if not self.sess_trg.empty:
            for _, r in self.sess_trg.iterrows():
                ax.axvline(r['onset'], color=C['trg_session'], lw=1.5,
                           ls='--', alpha=0.9, zorder=2)
                ax.text(r['onset'], yhi, f" {r['session_label']}",
                        color=C['trg_session'], fontsize=6, rotation=90,
                        ha='left', va='top', zorder=6)

        # Petits traits en bas = triggers boutons EEG (vue d'ensemble)
        if self.show_btn and not self.triggers.empty:
            for d, col in ((RESP_BTN_DESC,  C['trg_respbtn']),
                           (RESET_BTN_DESC, C['trg_resetbtn'])):
                ts = self.triggers.loc[self.triggers['description'] == d,
                                       'onset'].to_numpy()
                if len(ts):
                    ax.vlines(ts, ylo, ylo + 0.18 * (yhi - ylo),
                              color=col, lw=0.6, alpha=0.7, zorder=2.5)

        for ann in self.annotations:
            if ann.get('is_mirror', False):
                continue
            fc, _ = self._ann_colors(ann)
            ax.axvspan(ann['onset'], ann['offset'], fc=fc, alpha=0.5, zorder=3)

        if self.progress_t > 0:
            ax.axvline(self.progress_t, color='#74c0fc', lw=2.2, zorder=5)
            ax.text(self.progress_t + self.dur * 0.003,
                    yhi - 0.05 * (yhi - ylo),
                    f"revu ({self.progress_t/self.dur*100:.0f}%)",
                    color='#74c0fc', fontsize=6, va='top', zorder=6)

        ax.axvspan(t0, t1, fc='white', alpha=0.10, zorder=1)
        ax.axvline(t0, color='white', lw=1.2, zorder=4)
        ax.axvline(t1, color='white', lw=1.2, zorder=4)
        ax.set_xlim(0, self.dur)
        ax.set_ylim(ylo, yhi)
        ax.set_ylabel('Vue\nglobale', color=C['text'], fontsize=7, labelpad=3)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(14))

    def _update_status(self, t0, t1):
        pri = [a for a in self.annotations if not a.get('is_mirror', False)]
        n_r = sum(1 for a in pri if a.get('event') == 'Response')
        n_e = sum(1 for a in pri if a.get('event') == 'Reset')
        rs  = sorted([a for a in pri if a.get('event') == 'Response'],
                     key=lambda a: a['onset'])
        irt_s = (f"  |  dernier IRT:{rs[-1]['onset']-rs[-2]['onset']:.1f}s"
                 if len(rs) >= 2 else '')
        lbl = 'AUTO' if self.label_mode == 'auto' else self.label_mode
        self.txt_status.set_text(
            f"t={t0:.1f}-{t1:.1f}s  |  Fen:{self.win_dur:.0f}s"
            f"  |  {(t0/self.dur*100 if self.dur else 0):.1f}%"
            f"  |  {get_session(t0, self.sess_trg)}"
            f"  |  Response:{n_r}  Reset:{n_e}{irt_s}"
            f"  |  Label:{lbl}"
            f"  |  Z:{len(self._history)} Y:{len(self._future)}"
            f"{'  [NORM]' if self.normalized else ''}"
            f"{'  [AUTO-ECH]' if self.autoscale_win else ''}"
            f"{'  [MIROIR]' if self.mirror_mode else ''}"
            f"{'' if self.show_btn else '  [BTN caches]'}")

    # ───────────────────────────────────────────────────────────── navigation
    def _nav(self, dt):
        self._set_tstart(self.t_start + dt)

    def _set_tstart(self, t):
        self.t_start = float(np.clip(t, 0.0,
                                     max(0.0, self.dur - self.win_dur)))
        self._refresh()

    def _go_center(self, t):
        self._set_tstart(t - self.win_dur / 2)

    def _zoom(self, f):
        c = self.t_start + self.win_dur / 2
        self.win_dur = float(np.clip(self.win_dur * f, 0.5, self.dur))
        self._set_tstart(c - self.win_dur / 2)

    def _zoom_reset(self):
        c = self.t_start + self.win_dur / 2
        self.win_dur = min(WINDOW_DURATION, self.dur)
        self._set_tstart(c - self.win_dur / 2)

    def _toggle_norm(self):
        self.normalized = not self.normalized
        self.ylims = ({ch: (-0.05, 1.05) for ch in self.channels}
                      if self.normalized else dict(self.ylims_raw))
        self._refresh()

    def _toggle_autoscale(self):
        self.autoscale_win = not self.autoscale_win
        print(f"  Auto-echelle : {'ON' if self.autoscale_win else 'OFF'}")
        self._refresh()

    def _toggle_mirror(self):
        self.mirror_mode = not self.mirror_mode
        self.b_mirror.label.set_text(
            'Miroir:ON' if self.mirror_mode else 'Miroir:OFF')
        self.b_mirror.ax.set_facecolor(
            C['btn_mir_on'] if self.mirror_mode else C['btn_mir_off'])
        self._set_title()
        print(f"  Miroir : {'ON' if self.mirror_mode else 'OFF'}")
        self.fig.canvas.draw_idle()

    def _toggle_btn_triggers(self):
            """Touche B : montre/cache les triggers boutons EEG (S10/S11)."""
            self.show_btn = not self.show_btn
            self.b_btn.label.set_text('BTN:ON' if self.show_btn else 'BTN:OFF')
            self.b_btn.ax.set_facecolor(
                C['btn_mir_on'] if self.show_btn else C['btn_mir_off'])
            print(f"  Triggers boutons EEG : {'ON' if self.show_btn else 'OFF'}")
            self._refresh()

    def _toggle_auto_presses(self):
            """Touche D : montre/cache les detections automatiques (rose)."""
            self.show_auto = not self.show_auto
            self.b_auto.label.set_text('Auto:ON' if self.show_auto else 'Auto:OFF')
            self.b_auto.ax.set_facecolor(
                C['btn_mir_on'] if self.show_auto else C['btn_mir_off'])
            print(f"  Detections automatiques : {'ON' if self.show_auto else 'OFF'}")
            self._refresh()

    def _set_label_mode(self, mode):
        self.label_mode = mode
        print(f"  Mode de label : {mode}")
        self._set_title()
        self._refresh()

    def _jump_to(self, text):
        try:
            self._go_center(float(str(text).strip().replace(',', '.')))
        except ValueError:
            print(f"  Valeur invalide : '{text}'")

    def _prev_session(self):
        if self.sess_trg.empty:
            print("  Aucune session detectee.")
            return
        b = self.sess_trg[self.sess_trg['onset'] < self.t_start - 1.0]
        if b.empty:
            print("  Debut de l'enregistrement atteint.")
            return
        r = b.iloc[-1]
        print(f"  -> {r['session_label']}  t={r['onset']:.1f}s")
        self._set_tstart(float(r['onset']))

    def _next_session(self):
        if self.sess_trg.empty:
            print("  Aucune session detectee.")
            return
        a = self.sess_trg[self.sess_trg['onset'] > self.t_start + 1.0]
        if a.empty:
            print("  Fin de l'enregistrement atteinte.")
            return
        r = a.iloc[0]
        print(f"  -> {r['session_label']}  t={r['onset']:.1f}s")
        self._set_tstart(float(r['onset']))

    # ───────────────────────────────────────────────────────────── evenements
    def _on_key(self, ev):
        actions = {
            'right':       lambda: self._nav(+STEP_SIZE),
            'left':        lambda: self._nav(-STEP_SIZE),
            'shift+right': lambda: self._nav(+1.0),
            'shift+left':  lambda: self._nav(-1.0),
            'ctrl+right':  lambda: self._nav(+5.0),
            'ctrl+left':   lambda: self._nav(-5.0),
            'pagedown':    lambda: self._nav(+self.win_dur),
            'pageup':      lambda: self._nav(-self.win_dur),
            'home':        lambda: self._set_tstart(0.0),
            'end':         lambda: self._set_tstart(self.dur - self.win_dur),
            '+': lambda: self._zoom(0.5), '=': lambda: self._zoom(0.5),
            '-': lambda: self._zoom(2.0), '0': self._zoom_reset,
            'n': self._toggle_norm, 'a': self._toggle_autoscale,
            'm': self._toggle_mirror, 'b': self._toggle_btn_triggers,
            'd': self._toggle_auto_presses,
            '1': lambda: self._set_label_mode('Response'),
            '2': lambda: self._set_label_mode('Reset'),
            '3': lambda: self._set_label_mode('auto'),
            'x': self._toggle_nearest_label,
            'w': self._prev_session, 'e': self._next_session,
            'ctrl+z': self._undo, 'ctrl+y': self._redo,
            'ctrl+shift+z': self._redo,
            'p': self._set_progress_marker, 't': self._show_stats,
            'ctrl+s': self._save, 's': self._save,
            'v': self._export_vmrk,
        }
        fn = actions.get(ev.key)
        if fn is not None:
            try:
                fn()
            except Exception as e:
                print(f"  [!] Erreur sur la touche '{ev.key}' : {e}")

    def _ax_ch(self, ax):
        try:
            return self.channels[self.ax_sig.index(ax)]
        except ValueError:
            return None

    def _on_press(self, ev):
        if ev.inaxes is None or ev.xdata is None:
            return
        if ev.inaxes is self.ax_ov:
            if ev.button == 1:
                self._go_center(ev.xdata)
            return
        if ev.button == 2 and ev.inaxes in self.ax_sig:
            xlim = ev.inaxes.get_xlim()
            bbox = ev.inaxes.get_window_extent()
            self._pan = {'x0': ev.x, 't0': self.t_start,
                         'pps': bbox.width / max(xlim[1] - xlim[0], 1e-6)}
            return
        ch = self._ax_ch(ev.inaxes)
        if ch is None:
            return
        if ev.button == 1:
            self._drag = {'ch': ch, 'ax': ev.inaxes, 'x0': float(ev.xdata),
                          'patch': None}
        elif ev.button == 3:
            self._delete_group(float(ev.xdata), ch)

    def _on_move(self, ev):
        if ev.inaxes in self.ax_sig and ev.xdata is not None:
            self._mouse = (self._ax_ch(ev.inaxes), float(ev.xdata))

        if self._pan is not None and ev.x is not None:
            dt = (ev.x - self._pan['x0']) / self._pan['pps']
            self._set_tstart(self._pan['t0'] - dt)
            return

        if self._drag is None or ev.xdata is None:
            return
        if ev.inaxes is not self._drag['ax']:
            return
        if self._drag['patch'] is not None:
            try:
                self._drag['patch'].remove()
            except Exception:
                pass
        xlo, xhi = sorted([self._drag['x0'], float(ev.xdata)])
        ylo, yhi = self._drag['ax'].get_ylim()
        lab = self._label_for(self._drag['ch'])
        fc = C['ev_resp_fc'] if lab == 'Response' else C['ev_reset_fc']
        ec = C['ev_resp_ec'] if lab == 'Response' else C['ev_reset_ec']
        rect = Rectangle((xlo, ylo), max(xhi - xlo, 1e-9), yhi - ylo,
                         fc=fc, ec=ec, alpha=0.30, lw=1.5, zorder=10)
        self._drag['ax'].add_patch(rect)
        self._drag['patch'] = rect
        self.fig.canvas.draw_idle()

    def _on_release(self, ev):
        if ev.button == 2:
            self._pan = None
            return
        if self._drag is None or ev.button != 1:
            return
        if self._drag['patch'] is not None:
            try:
                self._drag['patch'].remove()
            except Exception:
                pass

        if ev.xdata is not None:
            ch  = self._drag['ch']
            on  = float(np.clip(min(self._drag['x0'], ev.xdata), 0.0, self.dur))
            off = float(np.clip(max(self._drag['x0'], ev.xdata), 0.0, self.dur))
            dur = off - on
            if dur >= MIN_ANNOT_DURATION:
                self._push_history()
                lab  = self._label_for(ch)
                sess = get_session(on, self.sess_trg)
                now  = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                targets = self._side_channels(ch) if self.mirror_mode else [ch]
                for tch in targets:
                    self.annotations.append({
                        'onset':     round(on, 5),
                        'offset':    round(off, 5),
                        'duration':  round(dur, 5),
                        'channel':   tch,
                        'side':      self._side(tch),
                        'event':     lab,
                        'is_mirror': (tch != ch),
                        'session':   sess,
                        'scorer':    self.scorer,
                        'scored_at': now})
                extra = (f"  + miroir x{len(targets)-1}"
                         if len(targets) > 1 else '')
                print(f"  [{lab:<8}] {ch:<20} {on:9.3f} -> {off:9.3f}s"
                      f"  dur={dur:.3f}s  {sess}{extra}")
                self._n_since_save += 1
                if AUTOSAVE_EVERY and self._n_since_save >= AUTOSAVE_EVERY:
                    self._save(quiet=True)

        self._drag = None
        self._refresh()

    def _on_scroll(self, ev):
        self._nav(-5.0 if ev.step > 0 else +5.0)

    def _on_close(self, ev):
        if self.annotations:
            print("\n  Fermeture -> sauvegarde automatique...")
            try:
                self._save()
            except Exception as e:
                print(f"  [!] Sauvegarde a la fermeture echouee : {e}")

    # ───────────────────────────────────────────────────────────── undo / redo
    def _push_history(self):
        self._history.append(copy.deepcopy(self.annotations))
        self._future.clear()

    def _undo(self):
        if not self._history:
            print("  Rien a annuler.")
            return
        self._future.append(copy.deepcopy(self.annotations))
        self.annotations = self._history.pop()
        print(f"  Annule.  ({len(self._history)} niveau(x) restant(s))")
        self._refresh()

    def _redo(self):
        if not self._future:
            print("  Rien a refaire.")
            return
        self._history.append(copy.deepcopy(self.annotations))
        self.annotations = self._future.pop()
        print(f"  Refait.  ({len(self._future)} redo restant(s))")
        self._refresh()

    # ──────────────────────────────────────────────────────────── annotations
    def _nearest_index(self, xc, ch):
        best_i, best_d = None, float('inf')
        for i, a in enumerate(self.annotations):
            if a['channel'] != ch:
                continue
            d = abs((a['onset'] + a['offset']) / 2 - xc)
            if d < best_d:
                best_i, best_d = i, d
        if best_i is None or best_d >= self.win_dur * 0.5:
            return None
        return best_i

    def _find_group(self, on, off, side):
        return [i for i, a in enumerate(self.annotations)
                if np.isclose(a['onset'], on, atol=1e-4)
                and np.isclose(a['offset'], off, atol=1e-4)
                and self._side(a['channel']) == side]

    def _delete_group(self, xc, ch):
        i = self._nearest_index(xc, ch)
        if i is None:
            return
        self._push_history()
        rm   = self.annotations[i]
        idxs = self._find_group(rm['onset'], rm['offset'], self._side(ch))
        if not idxs:
            idxs = [i]
        for j in sorted(idxs, reverse=True):
            self.annotations.pop(j)
        print(f"  Supprime : [{ch}] {rm['onset']:.3f}s "
              f"({len(idxs)} ligne(s))  |  Ctrl+Z pour annuler")
        self._refresh()

    def _toggle_nearest_label(self):
        ch, x = self._mouse
        if ch is None or x is None:
            print("  Place la souris sur une annotation puis appuie sur X.")
            return
        i = self._nearest_index(x, ch)
        if i is None:
            print("  Aucune annotation proche de la souris.")
            return
        self._push_history()
        rm  = self.annotations[i]
        new = 'Reset' if rm.get('event') == 'Response' else 'Response'
        grp = self._find_group(rm['onset'], rm['offset'],
                               self._side(rm['channel'])) or [i]
        for j in grp:
            self.annotations[j]['event'] = new
        print(f"  {rm['onset']:.3f}s  ->  {new}")
        self._refresh()

    # ═════════════════════════════════════════════════════════════════════════
    # TABLE D'EVENEMENTS UNIFIEE (coeur du CSV)
    # ═════════════════════════════════════════════════════════════════════════
    def _build_event_table(self) -> pd.DataFrame:
        rows = []

        # 1) triggers
        if not self.triggers.empty:
            kept_starts = (set(np.round(self.sess_trg['onset'].values, 6))
                           if not self.sess_trg.empty else set())
            for _, t in self.triggers.iterrows():
                d = t['description']
                if d == SESSION_START_DESC:
                    if round(float(t['onset']), 6) not in kept_starts:
                        continue                      # doublon -> ignore
                    name = f"Start {get_session(t['onset'], self.sess_trg)}"
                elif d == SESSION_END_DESC:
                    name = f"End {get_session(t['onset'], self.sess_trg)}"
                else:
                    name = EVENT_NAMES.get(d)
                    if name is None:
                        continue
                    if (not INCLUDE_BTN_TRIGGERS
                            and name in ('trigger resp BTN',
                                         'trigger reset BTN')):
                        continue
                rows.append({'event': name, 'source': 'trigger',
                             'onset': float(t['onset']), 'duration_s': np.nan,
                             'channel': '', 'side': '', 'is_mirror': False,
                             'scorer': '', 'scored_at': ''})

        # 2) scoring manuel
        for a in self.annotations:
            rows.append({'event': a.get('event', 'Response'),
                         'source': 'manual',
                         'onset': float(a['onset']),
                         'duration_s': round(float(a['duration']), 4),
                         'channel': a['channel'],
                         'side': a.get('side', self._side(a['channel'])),
                         'is_mirror': bool(a.get('is_mirror', False)),
                         'scorer': a.get('scorer', self.scorer),
                         'scored_at': a.get('scored_at', '')})

        if not rows:
            return pd.DataFrame()

        df = (pd.DataFrame(rows)
              .sort_values(['onset', 'source', 'event'], kind='mergesort')
              .reset_index(drop=True))

        # 3) session + temps dans la session
        df['session'] = df['onset'].map(lambda t: get_session(t, self.sess_trg))
        df['time_in_session_s'] = (
            df['onset'] - df['onset'].map(
                lambda t: get_session_start(t, self.sess_trg))).round(4)

        # 4) deltas / IRT / streaks
        for c in ('delta_since_last_RR', 'irt',
                  'streak_response', 'streak_reset'):
            df[c] = np.nan
        last_rr = last_resp = None
        s_r = s_e = 0
        for i, r in df.iterrows():
            ev = str(r['event'])
            if r['source'] == 'manual' and not r['is_mirror']:
                if last_rr is not None:
                    df.at[i, 'delta_since_last_RR'] = round(
                        r['onset'] - last_rr, 4)
                last_rr = r['onset']
                if ev == 'Response':
                    if last_resp is not None:
                        df.at[i, 'irt'] = round(r['onset'] - last_resp, 4)
                    last_resp = r['onset']
                    s_r, s_e = s_r + 1, 0
                else:
                    s_e, s_r = s_e + 1, 0
                df.at[i, 'streak_response'] = s_r
                df.at[i, 'streak_reset']    = s_e
            elif ev.startswith('Start ') or ev == 'Break':
                last_rr = last_resp = None
                s_r = s_e = 0
            elif ev in PROBE_EVENTS:
                s_r = s_e = 0

        # 5) IRT glissant + CV glissant, par session
        df['irt_roll'] = np.nan
        df['cv_irt_roll'] = np.nan
        mask = ((df['source'] == 'manual') & (~df['is_mirror'])
                & (df['event'] == 'Response'))
        for sess in df.loc[mask, 'session'].unique():
            idx  = df.index[mask & (df['session'] == sess)].tolist()
            irts = df.loc[idx, 'irt'].to_numpy(dtype=float)
            for j, ridx in enumerate(idx):
                w = irts[max(0, j - ROLLING_WINDOW + 1): j + 1]
                w = w[~np.isnan(w)]
                if len(w) >= 2:
                    m, s = float(np.mean(w)), float(np.std(w, ddof=1))
                    df.at[ridx, 'irt_roll'] = round(m, 4)
                    df.at[ridx, 'cv_irt_roll'] = (round(s / m * 100, 2)
                                                  if m > 0 else np.nan)

        # 6) temps absolus + heure murale
        df['onset_absolute']  = (df['onset'] + self.time_offset).round(4)
        df['wall_time']       = self._wall_time_col(df['onset_absolute'])
        df['onset_s']         = df['onset'].round(4)
        df['onset_min']       = (df['onset'] / 60).round(4)
        df['subname']         = self.subname
        df['hand_response']   = self.hand

        cols = ['subname', 'hand_response', 'session', 'event', 'source',
                'channel', 'side', 'onset_s', 'onset_min',
                'time_in_session_s', 'wall_time', 'onset_absolute',
                'duration_s', 'delta_since_last_RR', 'irt', 'irt_roll',
                'cv_irt_roll', 'streak_response', 'streak_reset',
                'is_mirror', 'scorer', 'scored_at']
        return df[cols]

    def _wall_time_col(self, onset_abs: pd.Series):
        if self.meas_date is None:
            return pd.Series([''] * len(onset_abs), index=onset_abs.index)
        try:
            base = pd.Timestamp(self.meas_date)
            wt = base + pd.to_timedelta(onset_abs.astype(float), unit='s')
            return wt.dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
        except Exception:
            return pd.Series([''] * len(onset_abs), index=onset_abs.index)

    # ────────────────────────────────────────────────────────── stats / resume
    def _summary(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        man = df[(df['source'] == 'manual') & (~df['is_mirror'])]
        sessions = (self.sess_trg['session_label'].tolist()
                    if not self.sess_trg.empty
                    else sorted(df['session'].unique()))
        rows = []
        for sess in list(dict.fromkeys(sessions)) + ['ALL']:
            sub = man if sess == 'ALL' else man[man['session'] == sess]
            evs = df  if sess == 'ALL' else df[df['session'] == sess]
            if sub.empty and evs.empty:
                continue
            irts = sub.loc[sub['event'] == 'Response', 'irt'].dropna()
            durs = sub.loc[sub['event'] == 'Response', 'duration_s'].dropna()
            rows.append({
                'session':    sess,
                'n_response': int((sub['event'] == 'Response').sum()),
                'n_reset':    int((sub['event'] == 'Reset').sum()),
                'n_probes':   int(evs['event'].isin(PROBE_EVENTS).sum()),
                'n_breaks':   int((evs['event'] == 'Break').sum()),
                'mean_irt':   round(irts.mean(), 3) if len(irts) else np.nan,
                'median_irt': round(irts.median(), 3) if len(irts) else np.nan,
                'std_irt':    round(irts.std(), 3) if len(irts) > 1 else np.nan,
                'cv_irt_%':   (round(irts.std() / irts.mean() * 100, 2)
                               if len(irts) > 1 and irts.mean() > 0 else np.nan),
                'mean_dur':   round(durs.mean(), 3) if len(durs) else np.nan})
        return pd.DataFrame(rows)

    def _show_stats(self):
        df  = self._build_event_table()
        man = (df[(df['source'] == 'manual') & (~df['is_mirror'])]
               if not df.empty else pd.DataFrame())
        if man.empty:
            print("  Pas encore d'annotation manuelle.")
            return

        fig, (a1, a2) = plt.subplots(2, 1, figsize=(15, 9),
                                     facecolor=C['bg_fig'],
                                     gridspec_kw={'hspace': 0.35})
        fig.suptitle(f"Stats — {self.subname}  |  fenetre glissante = "
                     f"{ROLLING_WINDOW} reponses", color=C['text'],
                     fontsize=10, fontweight='bold')
        for ax in (a1, a2):
            ax.set_facecolor(C['bg_ax'])
            ax.tick_params(colors=C['text'], labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor(C['spine'])
            ax.grid(True, color=C['grid'], lw=0.5, alpha=0.6)

        r = man[man['event'] == 'Response']
        if not r.empty:
            a1.scatter(r['onset_s'], r['irt'], s=22, color=C['ev_resp_ec'],
                       alpha=0.8, label='IRT brut', zorder=3)
            a1.plot(r['onset_s'], r['irt_roll'], lw=2.2, ls='--',
                    color='#00d4ff', zorder=4,
                    label=f'moyenne glissante ({ROLLING_WINDOW})')
            a2.plot(r['onset_s'], r['cv_irt_roll'], lw=1.6, marker='o',
                    markersize=4, color='#51cf66', label='CV glissant')
        a2.axhline(33, color='#ff6b6b', lw=1.2, ls=':', alpha=0.9)
        a2.text(0.01, 0.95, 'CV = 33 %', transform=a2.transAxes,
                color='#ff6b6b', fontsize=7, va='top')

        e = man[man['event'] == 'Reset']
        if not e.empty:
            a1.scatter(e['onset_s'], e['delta_since_last_RR'], s=28,
                       marker='x', color=C['ev_reset_ec'],
                       label='Reset (delta)', zorder=3)

        if not self.sess_trg.empty:
            yhi = a1.get_ylim()[1]
            for _, s in self.sess_trg.iterrows():
                for ax in (a1, a2):
                    ax.axvline(s['onset'], color=C['trg_session'], lw=1.4,
                               ls='--', alpha=0.85)
                a1.text(s['onset'], yhi, f" {s['session_label']}",
                        color=C['trg_session'], fontsize=7, rotation=90,
                        ha='left', va='top')

        a1.set_ylabel('IRT (s)', color=C['text'], fontsize=9)
        a1.set_title('Inter-Response Time (Response -> Response)',
                     color=C['text'], fontsize=8)
        a1.legend(fontsize=7, facecolor=C['bg_ax'], edgecolor=C['spine'],
                  labelcolor=C['text'], loc='upper right')
        a2.set_ylabel('CV IRT (%)', color=C['text'], fontsize=9)
        a2.set_xlabel('Temps (s)', color=C['text'], fontsize=9)
        a2.legend(fontsize=7, facecolor=C['bg_ax'], edgecolor=C['spine'],
                  labelcolor=C['text'], loc='upper right')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show(block=False)

        s = self._summary(df)
        if not s.empty:
            print(f"\n  {'─'*82}")
            print(f"  RESUME — {self.subname}   (main REPONSE : {self.hand})")
            print(f"  {'─'*82}")
            print(s.to_string(index=False))
            print(f"  {'─'*82}\n")

    # ─────────────────────────────────────────────────────── charger / sauver
    def _load_csv(self):
        if not os.path.exists(self.csv_path):
            print("  Aucun CSV existant -> nouveau scoring.")
            return
        try:
            df = pd.read_csv(self.csv_path)
            if 'source' not in df.columns or 'onset_absolute' not in df.columns:
                print("  [!] CSV d'un format different -> ignore "
                      "(le fichier ne sera pas ecrase avant ta 1re sauvegarde).")
                return
            man = df[df['source'] == 'manual'].copy()
            if man.empty:
                print("  CSV present mais aucune annotation manuelle.")
                return
            man['is_mirror'] = man['is_mirror'].map(
                lambda x: str(x).lower() in ('true', '1', 'yes'))
            n0 = len(man)
            man['onset']    = man['onset_absolute'].astype(float) - self.time_offset
            man['duration'] = man['duration_s'].astype(float)
            man['offset']   = man['onset'] + man['duration']
            neg = int((man['onset'] < 0).sum())
            man = man[man['onset'] >= 0].reset_index(drop=True)
            self.annotations = [{
                'onset':     round(float(r['onset']), 5),
                'offset':    round(float(r['offset']), 5),
                'duration':  round(float(r['duration']), 5),
                'channel':   str(r['channel']),
                'side':      str(r['side']) if pd.notna(r.get('side'))
                             else self._side(str(r['channel'])),
                'event':     str(r['event']),
                'is_mirror': bool(r['is_mirror']),
                'session':   get_session(float(r['onset']), self.sess_trg),
                'scorer':    str(r['scorer']) if pd.notna(r.get('scorer')) else self.scorer,
                'scored_at': str(r['scored_at']) if pd.notna(r.get('scored_at')) else '',
            } for _, r in man.iterrows()]
            print(f"\n  {len(self.annotations)} annotation(s) rechargee(s) "
                  f"depuis {Path(self.csv_path).name}")
            if neg:
                print(f"  [!] {neg}/{n0} annotation(s) situee(s) avant le "
                      f"t=0 courant -> ecartee(s).")
        except Exception as e:
            print(f"  [!] Rechargement CSV echoue : {e}")

    def _load_auto_presses(self) -> None:
        """Overlay optionnel (rose) : detections automatiques produites par
        detection.ipynb, exportees en <stem>_auto_presses.csv avec les
        colonnes channel, onset_absolute, offset_absolute[, rejected].
        Purement visuel -- ne touche jamais self.annotations ni le CSV
        sauvegarde."""
        self.auto_presses: list = []
        if not os.path.exists(self.auto_path):
            return
        try:
            df = pd.read_csv(self.auto_path)
            if 'rejected' in df.columns:
                df = df[~df['rejected'].astype(bool)]
            self.auto_presses = [{
                'channel': str(r['channel']),
                'onset':   float(r['onset_absolute'])  - self.time_offset,
                'offset':  float(r['offset_absolute']) - self.time_offset,
            } for _, r in df.iterrows()]
            print(f"  {len(self.auto_presses)} detection(s) automatique(s) "
                  f"chargee(s) depuis {Path(self.auto_path).name}")
        except Exception as e:
            print(f"  [!] Chargement des detections automatiques echoue : {e}")

    def _save(self, quiet: bool = False):
        df = self._build_event_table()
        if df.empty:
            print("  Rien a sauvegarder.")
            return
        try:
            df.to_csv(self.csv_path, index=False, encoding='utf-8-sig')
        except PermissionError:
            print(f"  [X] Impossible d'ecrire {self.csv_path}")
            print(f"      Le fichier est probablement OUVERT dans Excel "
                  f"-> ferme-le puis re-sauvegarde (Ctrl+S).")
            return
        save_signal_cache(self.out_dir, self.stem, self.channels,
                          self.signals, self.times, self.sfreq,
                          time_offset=self.time_offset)
        self._save_state()
        self._n_since_save = 0

        n_man = int(((df['source'] == 'manual') & (~df['is_mirror'])).sum())
        if quiet:
            print(f"  [autosave] {n_man} evenements manuels -> "
                  f"{Path(self.csv_path).name}")
            return
        print(f"\n  {'─'*74}")
        print(f"  CSV  -> {self.csv_path}")
        print(f"  JSON -> {self.state_path}")
        print(f"  {len(df)} lignes  |  {n_man} evenements scores a la main"
              f"  |  scoreur : {self.scorer}  |  main REPONSE : {self.hand}")
        s = self._summary(df)
        if not s.empty:
            print(f"  {'─'*74}")
            print(s.to_string(index=False))
        print(f"  {'─'*74}\n")

    def _save_state(self):
        pri = [a for a in self.annotations if not a.get('is_mirror', False)]
        state = {
            'subname': self.subname, 'scorer': self.scorer,
            'hand_response': self.hand,
            't_start': round(self.t_start, 3),
            'win_dur': round(self.win_dur, 3),
            'progress_t': round(self.progress_t, 3),
            'time_offset': round(self.time_offset, 3),
            'zero_time_at_start_index': ZERO_TIME_AT_START_INDEX,
            'sessions': (self.sess_trg['session_label'].tolist()
                         if not self.sess_trg.empty else []),
            'n_response': sum(1 for a in pri if a.get('event') == 'Response'),
            'n_reset':    sum(1 for a in pri if a.get('event') == 'Reset'),
            'last_saved': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        try:
            with open(self.state_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  [!] Ecriture du state.json echouee : {e}")

    def _load_state(self):
        if not os.path.exists(self.state_path):
            print("  Nouvelle session de scoring.")
            return
        try:
            with open(self.state_path, encoding='utf-8') as f:
                st = json.load(f)
            self.win_dur = float(np.clip(st.get('win_dur', WINDOW_DURATION),
                                         0.5, max(self.dur, 1.0)))
            self.t_start = float(np.clip(st.get('t_start', 0.0), 0.0,
                                         max(0.0, self.dur - self.win_dur)))
            self.progress_t = float(np.clip(st.get('progress_t', 0.0),
                                            0.0, self.dur))
            rr = f"{st.get('n_response','?')} / {st.get('n_reset','?')}"
            print(f"\n  +{'─'*52}+")
            print(f"  |  REPRISE DE SESSION{' '*32}|")
            print(f"  +{'─'*52}+")
            print(f"  |  Scoreur         : {str(st.get('scorer','?')):<31}|")
            print(f"  |  Main REPONSE    : {str(st.get('hand_response','?')):<31}|")
            print(f"  |  Derniere sauveg : {str(st.get('last_saved','?')):<31}|")
            print(f"  |  Position        : t = {self.t_start:<27.1f}|")
            print(f"  |  Response/Reset  : {rr:<31}|")
            if self.progress_t > 0:
                p = (f"{self.progress_t:.1f}s "
                     f"({self.progress_t/self.dur*100:.1f}%)")
                print(f"  |  Revu jusqu'a    : {p:<31}|")
            print(f"  +{'─'*52}+\n")
        except Exception as e:
            print(f"  [!] Chargement de l'etat echoue : {e}")

    def _set_progress_marker(self):
        self.progress_t = min(self.t_start + self.win_dur, self.dur)
        print(f"  Marqueur : revu jusqu'a {self.progress_t:.1f}s "
              f"({self.progress_t/self.dur*100:.1f}%)")
        self._save_state()
        self._refresh()

    def _export_vmrk(self):
        if not self.annotations:
            print("  Rien a exporter.")
            return
        lines = ["Brain Vision Data Exchange Marker File, Version 1.0", "",
                 "[Common Infos]", "Codepage=UTF-8",
                 f"DataFile={self.eeg_name}", "", "[Marker Infos]",
                 "Mk1=New Segment,,1,1,0,00000000000000000000"]
        for i, a in enumerate(sorted(self.annotations,
                                     key=lambda x: x['onset']), 2):
            pos  = int(round((a['onset'] + self.time_offset) * self.sfreq))
            size = max(1, int(round(a['duration'] * self.sfreq)))
            chn  = self.ch_orig_idx.get(a['channel'], 0) + 1
            tag  = '_mirror' if a.get('is_mirror') else ''
            lines.append(f"Mk{i}=Comment,{a['event']}_{a['channel']}"
                         f"_{a.get('session','?')}{tag},{pos},{size},{chn}")
        try:
            with open(self.vmrk_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
            print(f"  Export .vmrk -> {self.vmrk_path}")
        except Exception as e:
            print(f"  [!] Export .vmrk echoue : {e}")


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTREE
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_vhdr(base_dir: str, subname: str,
                  filename: Optional[str]) -> Optional[str]:
    for c in ([filename] if filename else []) + [f"{subname}.vhdr"]:
        p = os.path.join(base_dir, c)
        if os.path.exists(p):
            return p
    found = sorted(Path(base_dir).glob("*.vhdr"))
    if len(found) == 1:
        print(f"  [i] '{subname}.vhdr' introuvable -> utilisation de "
              f"'{found[0].name}'.")
        return str(found[0])
    if len(found) > 1:
        print(f"  [!] Plusieurs .vhdr dans {base_dir} :")
        for f in found:
            print(f"        - {f.name}")
        print("      -> renseigne FILENAME en haut du script.")
    return None


def main(subject: str = None, data_root: str = None, filename: str = None,
         scorer: str = None, response_hand: str = None,
         headless: bool = False) -> bool:
    """Retourne True si le scoring a demarre, False sinon."""
    data_root = data_root or DATA_ROOT

    if not os.path.isdir(data_root):
        print(f"\n  [X] Racine introuvable : {data_root}")
        print("      -> corrige DATA_ROOT en haut du script.")
        return False

    # ── Participant ──────────────────────────────────────────────────────────
    subname = (subject or SUBJECT or "").strip()
    if not subname and not headless:
        try:
            dispo = sorted(d.name for d in Path(data_root).iterdir()
                           if d.is_dir() and d.name != OUTPUT_SUBFOLDER)
        except Exception:
            dispo = []
        if dispo:
            print(f"\n  Participants disponibles dans {data_root} :")
            print(f"    {', '.join(dispo)}")
        while not subname:
            subname = input("\n  ID participant (ex: TT001) : ").strip()
    if not subname:
        print("  Aucun participant indique.")
        return False

    base_dir = os.path.join(data_root, subname)
    if not os.path.isdir(base_dir):
        print(f"\n  [X] Le dossier {base_dir} n'existe pas.")
        return False

    # ── Fichier .vhdr ────────────────────────────────────────────────────────
    fp = _resolve_vhdr(base_dir, subname, filename or FILENAME)
    if fp is None:
        print(f"\n  [X] Aucun .vhdr utilisable dans {base_dir}")
        print("      Le .vhdr, le .eeg ET le .vmrk doivent etre dans "
              "le meme dossier.")
        return False

    # ── Dossier de sortie unique ─────────────────────────────────────────────
    out_dir = os.path.join(data_root, OUTPUT_SUBFOLDER)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n  Participant       : {subname}")
    print(f"  Fichier           : {fp}")
    print(f"  Dossier de sortie : {out_dir}")
    print(f"                      (fichiers prefixes '{subname}_')")

    # ── Scoreur + main de reponse ────────────────────────────────────────────
    scorer = (scorer or SCORER_NAME or "").strip()
    if not scorer and not headless:
        scorer = input("\n  Nom du scoreur (Entree = 'inconnu') : ").strip()
    scorer = scorer or 'inconnu'

    hand = (response_hand or RESPONSE_HAND or "").strip().upper()
    while hand not in ("R", "L"):
        if headless:
            hand = "R"
            break
        hand = input("  Main du participant pour la REPONSE ? "
                     "(R = droite / L = gauche) : ").strip().upper()
    print(f"  Scoreur : {scorer}   |   Main REPONSE : "
          f"{'droite' if hand == 'R' else 'gauche'}")

    # ── Chargement ───────────────────────────────────────────────────────────
    raw, ch_orig_idx, triggers, chs = inspect_and_load(fp, FORCE_CHANNELS)
    if not chs:
        print("  [X] Aucun canal de force trouve -> renseigne FORCE_CHANNELS.")
        return False

    sfreq     = float(raw.info['sfreq'])
    stem      = subname
    eeg_name  = Path(fp).stem + '.eeg'
    meas_date = raw.info.get('meas_date')
    if meas_date is not None:
        print(f"  meas_date (debut reel de l'EEG) : {meas_date}\n")
    else:
        print("  [!] Pas de meas_date -> colonne wall_time vide.\n")

    t_offset = find_zero_offset(triggers, ZERO_TIME_AT_START_INDEX)

    signals, times = try_load_cache(out_dir, stem, chs, sfreq,
                                    expected_offset=t_offset)
    if signals is None:
        signals = {ch: raw.get_data(picks=[raw.ch_names.index(ch)])[0]
                   for ch in chs}
        times   = raw.times.copy()
        signals, times = shift_signals(signals, times, t_offset)
    triggers = shift_triggers(triggers, t_offset)

    print(f"  Canaux scores : {chs}\n")

    ForceAnnotator(signals=signals, times=times, sfreq=sfreq, channels=chs,
                   ch_orig_idx=ch_orig_idx, triggers=triggers,
                   out_dir=out_dir, stem=stem, subname=subname,
                   scorer=scorer, response_hand=hand, eeg_name=eeg_name,
                   time_offset=t_offset, meas_date=meas_date,
                   headless=headless)
    return True


if __name__ == '__main__':
    while True:
        main()
        again = input("\n  Scorer un autre participant ? (o/N) : ").strip().lower()
        if again not in ('o', 'oui', 'y', 'yes'):
            print("\n  Termine.\n")
            break