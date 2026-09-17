"""Setup GUI - Python port of ``Sleep_score_HM_neuron.m`` (PyQt6).

Pick a session folder, enter three channel numbers, auto-detect the motion/EMG
file, choose an output folder and parameters, then launch the state editor.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIntValidator, QDoubleValidator, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox, QFrame,
    QFileDialog, QSizePolicy, QScrollArea, QGraphicsDropShadowEffect,
)

from processing import (compute_channel_spectrogram, process_motion,
                        detect_sampling_rate, cache_path, save_cache, load_cache,
                        find_lfp_source, load_lfp_channel, find_output,
                        LFP_FS_MAX)
from state_editor import StateEditor
from mac_vibrancy import apply_vibrancy

# motion processing modes shown in the dropdown -> process_motion() mode string
MOTION_MODES = {
    "Accelerometer (case 3)": "accelerometer",
    "MEG (case 4)": "meg",
    "File / precomputed (case 5)": "file",
}

# auto-detected motion files, in priority order (raw accelerometer first, to
# suit the default Accelerometer mode)
EMG_CANDIDATES = ["motion.npy", "emg_rms.npy", "emg_data.npy",
                  "theta_delta_ratio.npy"]

# sleep_channels.npy holds the per-rat tetrodes from SLEEP_CHANNELS_<rat> in
# hm_tracker_paths.txt, keyed by Buzsáki role. These are the three channels the
# scorer should look at, in this order, shown under these display names.
SLEEP_ROLE_LABELS = [("cortex", "cortex"), ("sr", "EEG"), ("pyr", "pyr")]

# --- Apple-style palette (light) -------------------------------------------
ACCENT = "#007AFF"     # systemBlue
INK = "#1D1D1F"        # primary label
MUTED = "#8A8A8E"      # systemGray
OK_GREEN = "#34C759"   # systemGreen
WARN = "#FF9500"       # systemOrange
ERR = "#FF3B30"        # systemRed
_BG = "#F2F2F5"
_CARD = "#FFFFFF"
_SEP = "#E4E4E8"
_BORDER = "#D1D1D6"

# Real behind-window blur (NSVisualEffectView) is opt-in via HM_VIBRANCY=1 — on
# some Qt/macOS builds it can hide the content, so the default is a reliable
# opaque light gradient with translucent panels that simulates the frosted look.
_VIBRANCY = bool(os.environ.get("HM_VIBRANCY"))
_ROOT_BG = "transparent" if _VIBRANCY else (
    "qlineargradient(x1:0, y1:0, x2:1, y2:1,"
    " stop:0 #F4F6FB, stop:0.5 #F2F1F8, stop:1 #FAF0F4)")

# Translucent "frosted materials" theme. The window is made see-through (via
# mac_vibrancy) so the NSVisualEffectView blur shows behind these semi-transparent
# panels; without vibrancy it falls back to the system window background.
STYLESHEET = f"""
QMainWindow {{ background: transparent; }}
QWidget#Root {{ background: {_ROOT_BG}; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QLabel {{ color: {INK}; font-size: 13px; background: transparent; }}

QLabel#AppTitle    {{ font-size: 22px; font-weight: 700; color: {INK}; letter-spacing: -0.3px; }}
QLabel#AppSubtitle {{ font-size: 12.5px; color: {MUTED}; }}
QLabel#Step        {{ font-size: 11px; font-weight: 700; color: #ffffff;
                      background: {ACCENT}; border-radius: 10px;
                      min-width: 20px; max-width: 20px; min-height: 20px;
                      max-height: 20px; qproperty-alignment: AlignCenter; }}
QLabel#CardTitle   {{ font-size: 14px; font-weight: 600; color: {INK}; letter-spacing: -0.2px; }}
QLabel#FieldLabel  {{ font-size: 12px; color: #55555A; }}
QLabel#Hint        {{ font-size: 11.5px; color: {MUTED}; }}

QFrame#Card {{ background: rgba(255,255,255,62%);
               border: 1px solid rgba(255,255,255,55%); border-radius: 16px; }}
QFrame#Divider {{ background: rgba(60,60,67,12%); max-height: 1px; border: none; }}

QLineEdit {{ background: rgba(255,255,255,78%); border: 1px solid rgba(60,60,67,16%);
             border-radius: 9px; padding: 7px 11px; font-size: 13px; color: {INK};
             selection-background-color: {ACCENT}; selection-color: #fff; }}
QLineEdit:focus {{ border: 2px solid {ACCENT}; padding: 6px 10px; }}
QLineEdit:read-only {{ background: rgba(245,245,247,55%); color: #3A3A3C; }}
QLineEdit:disabled {{ background: rgba(245,245,247,40%); color: #AFAFB4; }}

QComboBox {{ background: rgba(255,255,255,78%); border: 1px solid rgba(60,60,67,16%);
             border-radius: 9px; padding: 6px 11px; font-size: 13px; min-height: 20px; color: {INK}; }}
QComboBox:focus {{ border: 2px solid {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: #FFFFFF; color: {INK}; border: 1px solid {_SEP};
             border-radius: 8px; padding: 4px; outline: none;
             selection-background-color: {ACCENT}; selection-color: #fff; }}

QCheckBox {{ font-size: 13px; color: {INK}; spacing: 8px; background: transparent; }}
QCheckBox::indicator {{ width: 18px; height: 18px; border-radius: 6px;
                        border: 1px solid rgba(60,60,67,25%); background: rgba(255,255,255,80%); }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border: 1px solid {ACCENT}; }}
QCheckBox::indicator:disabled {{ background: rgba(235,235,238,55%); border: 1px solid #DADADE; }}

QPushButton#Browse {{ background: rgba(255,255,255,72%); border: 1px solid rgba(60,60,67,14%);
                      border-radius: 8px; padding: 6px 15px; font-size: 12.5px;
                      color: {ACCENT}; font-weight: 500; }}
QPushButton#Browse:hover {{ background: rgba(255,255,255,94%); }}
QPushButton#Browse:pressed {{ background: rgba(235,235,240,90%); }}

QPushButton#Launch {{ background: {ACCENT}; color: #ffffff; font-size: 15px;
                      font-weight: 600; border: none; border-radius: 12px; padding: 13px;
                      letter-spacing: -0.2px; }}
QPushButton#Launch:hover {{ background: #0A6CE0; }}
QPushButton#Launch:pressed {{ background: #0A5FC5; }}
QPushButton#Launch:disabled {{ background: rgba(120,130,145,42%); color: rgba(255,255,255,85%); }}

QScrollBar:vertical {{ background: transparent; width: 11px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: rgba(120,120,128,42%); border-radius: 5px; min-height: 32px; }}
QScrollBar::handle:vertical:hover {{ background: rgba(120,120,128,64%); }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
"""


class SetupGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.lfp_folder = ""
        self.emg_file = ""
        self.out_folder = ""
        self.motion_from_nwb = False  # no motion .npy, but the NWB carries it
        self.nwb_path = ""          # session NWB (tracker step 8), "" if none found
        self.scorings = []          # scorings stored in it, newest first
        self.prev_scoring = None    # the one to continue; None = start a fresh one
        self.lfp_source = None      # dict from find_lfp_source, set on folder select
        self.sleep_channels = {}    # {role: tetrode} from sleep_channels.npy

        self.setWindowTitle("Sleep Score Setup")
        self.resize(860, 720)
        self.setMinimumSize(720, 600)
        self._build()
        self._reload_scorings()     # start on the blank entry: nothing resumed
        self._update_ready()

    # ------------------------------------------------------------------ layout
    def _card(self, step, title, subtitle=""):
        """A white rounded card with a numbered step badge + title. Returns the
        card frame and an inner QVBoxLayout to add content rows to."""
        card = QFrame()
        card.setObjectName("Card")
        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(22)
        shadow.setXOffset(0)
        shadow.setYOffset(3)
        shadow.setColor(QColor(0, 0, 0, 24))
        card.setGraphicsEffect(shadow)
        outer = QVBoxLayout(card)
        outer.setContentsMargins(16, 14, 16, 16)
        outer.setSpacing(10)

        head = QHBoxLayout()
        head.setSpacing(9)
        badge = QLabel(str(step))
        badge.setObjectName("Step")
        head.addWidget(badge)
        tt = QLabel(title)
        tt.setObjectName("CardTitle")
        head.addWidget(tt)
        head.addStretch(1)
        if subtitle:
            sub = QLabel(subtitle)
            sub.setObjectName("Hint")
            head.addWidget(sub)
        outer.addLayout(head)
        return card, outer

    def _dot(self):
        d = QLabel("●")
        d.setFixedWidth(16)
        d.setStyleSheet("color: #c7ccd2; font-size: 14px;")
        return d

    def _set_dot(self, dot, state):
        colors = {"ok": OK_GREEN, "warn": WARN, "off": "#c7ccd2"}
        dot.setStyleSheet(f"color: {colors[state]}; font-size: 14px;")

    def _path_row(self, edit, browse_cb, dot, placeholder, tooltip,
                  clear_cb=None):
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(dot)
        edit.setReadOnly(True)
        edit.setPlaceholderText(placeholder)
        edit.setToolTip(tooltip)
        edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row.addWidget(edit, 1)
        btn = QPushButton("Browse…")
        btn.setObjectName("Browse")
        btn.clicked.connect(browse_cb)
        row.addWidget(btn)
        if clear_cb is not None:            # optional: empty the field again
            clr = QPushButton("Clear")
            clr.setObjectName("Browse")
            clr.setToolTip("Empty this field")
            clr.clicked.connect(clear_cb)
            row.addWidget(clr)
        return row

    def _build(self):
        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        self.setStyleSheet(STYLESHEET)

        page = QVBoxLayout(root)
        page.setContentsMargins(0, 0, 0, 0)
        page.setSpacing(0)

        # --- header band --------------------------------------------------
        header = QWidget()
        hb = QVBoxLayout(header)
        hb.setContentsMargins(22, 18, 22, 10)
        hb.setSpacing(2)
        title = QLabel("🧠  Sleep Score")
        title.setObjectName("AppTitle")
        hb.addWidget(title)
        sub = QLabel("Load an LFP recording, pick channels, then open the state editor.")
        sub.setObjectName("AppSubtitle")
        hb.addWidget(sub)
        page.addWidget(header)

        # --- scrollable body of cards ------------------------------------
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        v = QVBoxLayout(body)
        v.setContentsMargins(22, 8, 22, 8)
        v.setSpacing(14)
        scroll.setWidget(body)
        page.addWidget(scroll, 1)

        # ============ Card 1: data sources ================================
        c1, box = self._card(1, "Data sources")

        box.addWidget(self._field_label("Session folder"))
        self.lfp_edit = QLineEdit()
        self.lfp_dot = self._dot()
        box.addLayout(self._path_row(
            self.lfp_edit, self._sel_lfp, self.lfp_dot,
            "folder containing the session .nwb",
            "The session folder written by tracker step 8 — its .nwb holds the "
            "LFP, EMG, motion and every scoring.\n"
            "Older sessions exported as loose .npy (lfp_data.npy / channels_npy/) "
            "still work."))
        self.info_label = QLabel("")
        self.info_label.setObjectName("Hint")
        self.info_label.setWordWrap(True)
        box.addWidget(self.info_label)

        box.addWidget(self._divider())

        box.addWidget(self._field_label("Motion / EMG file  (optional)"))
        self.emg_edit = QLineEdit()
        self.emg_dot = self._dot()
        box.addLayout(self._path_row(
            self.emg_edit, self._sel_emg, self.emg_dot,
            "taken from the .nwb — or browse a .npy to override",
            "Motion / EMG signal used to separate wake from sleep.\n"
            "The session .nwb already carries it, so this only needs setting to "
            "override it or for an older .npy-only session."))
        self.emg_auto = QLabel("")
        self.emg_auto.setObjectName("Hint")
        box.addWidget(self.emg_auto)

        box.addWidget(self._divider())

        box.addWidget(self._field_label("Output / save folder"))
        self.out_edit = QLineEdit()
        self.out_dot = self._dot()
        box.addLayout(self._path_row(
            self.out_edit, self._sel_out, self.out_dot,
            "where the spectrogram cache and backup files are written",
            "The scoring itself goes into the session .nwb. This folder takes the "
            "spectrogram cache and the .npz/.mat backup copies.\n"
            "Defaults to the session folder."))

        box.addWidget(self._divider())

        box.addWidget(self._field_label("Reload state scored by"))
        prow = QHBoxLayout()
        prow.setSpacing(8)
        self.prev_combo = QComboBox()
        self.prev_combo.setToolTip(
            "Continue a scoring already stored in the session .nwb: its labels are "
            "loaded, its scorer name is kept (you are not asked again), and saving "
            "updates that same scorer instead of writing a new file.\n"
            "Leave on the blank entry to start a fresh scoring.")
        self.prev_combo.currentIndexChanged.connect(self._on_prev_pick)
        prow.addWidget(self.prev_combo, 1)
        self.prev_dot = self._dot()
        prow.addWidget(self.prev_dot)
        box.addLayout(prow)
        self.prev_hint = self._hint(
            "Select the session folder first — scorings are read from its .nwb.")
        box.addWidget(self.prev_hint)
        v.addWidget(c1)

        # ============ Card 2: recording & parameters ======================
        c2, box2 = self._card(2, "Recording & parameters")

        chrow = QHBoxLayout()
        chrow.setSpacing(8)
        lbl = self._field_label("Channels to score")
        chrow.addWidget(lbl)
        chrow.addSpacing(6)
        self.ch_edits = []
        for k in range(3):
            e = QLineEdit(str(k + 1))
            e.setFixedWidth(52)
            e.setValidator(QIntValidator(1, 9999, self))
            e.setAlignment(Qt.AlignmentFlag.AlignCenter)
            e.setToolTip(f"1-based channel number for slot {k + 1}")
            self.ch_edits.append(e)
            chrow.addWidget(e)
        chrow.addWidget(self._hint("(1-based; three distinct channels)"))
        chrow.addStretch(1)
        box2.addLayout(chrow)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        grid.addWidget(self._field_label("Sampling rate (Hz)"), 0, 0)
        self.fs_edit = QLineEdit("1500")
        self.fs_edit.setFixedWidth(90)
        self.fs_edit.setValidator(QDoubleValidator(1.0, 1e6, 3, self))
        self.fs_edit.setToolTip("LFP sampling rate. Auto-filled from the session "
                                ".nwb (or lfp_timestamps.npy) when sensible.")
        grid.addWidget(self.fs_edit, 0, 1)
        grid.addWidget(self._field_label("Session name"), 0, 2)
        self.name_edit = QLineEdit("HM_neurons")
        self.name_edit.setToolTip("Base name for saved files. Auto-filled from the folder's session prefix.")
        grid.addWidget(self.name_edit, 0, 3)
        grid.addWidget(self._field_label("Motion type"), 1, 0)
        self.motion_combo = QComboBox()
        self.motion_combo.addItems(list(MOTION_MODES.keys()))
        self.motion_combo.setToolTip("How the motion/EMG file is processed into a movement trace.")
        grid.addWidget(self.motion_combo, 1, 1, 1, 3)
        grid.setColumnStretch(3, 1)
        box2.addLayout(grid)

        self.recompute_chk = QCheckBox("Ignore cache (recompute spectrograms)")
        self.recompute_chk.setToolTip("Force a fresh compute instead of loading the cached spectrograms.")
        box2.addWidget(self.recompute_chk)
        v.addWidget(c2)

        # ============ Card 3: auto-scoring ================================
        c3, box3 = self._card(3, "Auto-scoring (Buzsáki)", "optional")
        self.buzsaki_chk = QCheckBox("Show Buzsáki auto-score in the editor (recomputes on open)")
        self.buzsaki_chk.setChecked(True)
        self.buzsaki_chk.toggled.connect(self._toggle_thresholds)
        box3.addWidget(self.buzsaki_chk)

        import buzsaki_score as _bz
        self.thr_grid = QGridLayout()
        self.thr_grid.setHorizontalSpacing(12)
        self.thr_grid.setVerticalSpacing(4)
        self.swf_edit = QLineEdit(str(_bz.SW_THRESH_FACTOR))
        self.thf_edit = QLineEdit(str(_bz.TH_THRESH_FACTOR))
        self.emgf_edit = QLineEdit("1.0")
        self.minsec_edit = QLineEdit("10")
        gate = ("the dip of its bimodal histogram" if _bz.MOVE_GATE_PCT is None
                else f"the {_bz.MOVE_GATE_PCT:.0f}th percentile")
        fields = [("SW×  (NREM)", self.swf_edit,
                   "Slow-wave threshold multiplier. ↓ = more NREM. 1.0 = the "
                   "histogram dip, as published. On our hand-scored session ×1.25 "
                   "agreed better (the dip there labels most REM as NREM), but that "
                   "did not carry over to other recordings — so set it per session."),
                  ("θ×  (REM)", self.thf_edit, "Theta threshold multiplier. ↓ = more REM."),
                  ("EMG×  (REM gate)", self.emgf_edit,
                   "Movement ceiling for REM: bins with EMG/motion above it can't be REM. "
                   f"1.0 = {gate} of the movement signal; ↑ = more REM."),
                  ("min ep (s)", self.minsec_edit, "Shortest epoch kept, in seconds.")]
        for i, (lbl, edit, tip) in enumerate(fields):
            head = self._field_label(lbl)
            self.thr_grid.addWidget(head, 0, i, Qt.AlignmentFlag.AlignHCenter)
            edit.setFixedWidth(64)
            edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
            edit.setValidator(QDoubleValidator(0.0, 100.0, 3, self))
            edit.setToolTip(tip)
            self.thr_grid.addWidget(edit, 1, i, Qt.AlignmentFlag.AlignHCenter)
        box3.addLayout(self.thr_grid)
        box3.addWidget(self._hint(
            "1.0 = automatic threshold  ·  ↓SW → more NREM  ·  ↓θ → more REM  ·  ↑EMG → laxer REM gate"))
        box3.addWidget(self._hint(
            "These thresholds are the published Buzsáki ones. A model fitted to "
            "your own scoring (fit_auto_score.py) is not used here — it needs "
            "buzsaki_score.py --model, since one fitted to a single session did "
            "not transfer to other recordings."))
        v.addWidget(c3)
        v.addStretch(1)

        # --- footer: status + launch -------------------------------------
        footer = QWidget()
        fb = QVBoxLayout(footer)
        fb.setContentsMargins(22, 6, 22, 16)
        fb.setSpacing(10)
        self.status = QLabel("Ready. Select a session folder to begin.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color: {INK}; font-size: 12px;")
        fb.addWidget(self.status)
        self.launch_btn = QPushButton("Launch State Editor")
        self.launch_btn.setObjectName("Launch")
        self.launch_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.launch_btn.clicked.connect(self._launch)
        fb.addWidget(self.launch_btn)
        page.addWidget(footer)

    # -- tiny widget factories --------------------------------------------
    def _field_label(self, text):
        lbl = QLabel(text)
        lbl.setObjectName("FieldLabel")
        return lbl

    def _hint(self, text):
        lbl = QLabel(text)
        lbl.setObjectName("Hint")
        return lbl

    def _divider(self):
        line = QFrame()
        line.setObjectName("Divider")
        line.setFrameShape(QFrame.Shape.HLine)
        return line

    def _toggle_thresholds(self, on):
        for e in (self.swf_edit, self.thf_edit, self.emgf_edit,
                  self.minsec_edit):
            e.setEnabled(on)

    def _update_ready(self):
        """Enable Launch only when the three required paths are set."""
        ready = bool(self.lfp_folder and (self.emg_file or self.motion_from_nwb)
                     and self.out_folder)
        self.launch_btn.setEnabled(ready)
        if ready:
            self.launch_btn.setText("▶  Launch State Editor")
        else:
            missing = [name for name, val in
                       [("session folder", self.lfp_folder),
                        ("motion/EMG file", self.emg_file or self.motion_from_nwb),
                        ("output folder", self.out_folder)] if not val]
            self.launch_btn.setText(f"Launch State Editor  (set {', '.join(missing)})")

    # ------------------------------------------------------------------ helpers
    def _set_status(self, msg, color=INK):
        self.status.setText(msg)
        self.status.setStyleSheet(f"color: {color}; font-size: 12px;")
        QApplication.processEvents()

    def _show_recording_info(self, folder):
        """Display channel count + duration and auto-fill the sampling rate."""
        src = self.lfp_source
        if src is None:
            self.info_label.setText("")
            return
        n_samples = src["n_samples"]
        chans = src["channels"]

        # An NWB-backed source carries its own rate (read off acquisition/lfp's
        # timestamps); otherwise fall back to lfp_timestamps.npy.
        fs = src.get("fs") or detect_sampling_rate(
            find_output(folder, "lfp_timestamps.npy"))
        rate_note = ""
        # LFP for sleep scoring is ~250–2000 Hz. A detected rate this high means
        # lfp_timestamps.npy holds the RAW acquisition rate (e.g. 30 kHz), not the
        # LFP rate — auto-setting it would make each spectrogram bin ~fs samples
        # (20× too wide) and score only ~1/20 of the recording. Keep the LFP
        # default and warn instead of silently adopting the raw rate.
        if fs and fs > LFP_FS_MAX:
            rate_note = (f"  (⚠ lfp_timestamps.npy implies {int(fs)} Hz = the raw "
                         f"rate, not the LFP rate — keeping {self.fs_edit.text()} Hz. "
                         f"Re-export the LFP or set the rate manually.)")
        elif fs:
            self.fs_edit.setText(str(int(fs)))
            rate_note = f"  (sampling rate auto-set to {int(fs)} Hz)"
        try:
            fs_val = float(self.fs_edit.text())
        except ValueError:
            fs_val = fs or 1000.0
        dur = n_samples / fs_val if fs_val else 0
        layout = "lfp_data.npy" if src["kind"] == "matrix" else "channels_npy/"
        rng = f"{chans[0]}-{chans[-1]}" if chans else "none"
        self.info_label.setText(
            f"{len(chans)} channels ({rng}) via {layout}, {n_samples:,} samples, "
            f"{dur:.1f} s ({dur / 60:.1f} min){rate_note}")

    def _prefill_sleep_channels(self, folder):
        """Fill the three channel boxes from the folder's sleep_channels.npy.

        That file is written by the tracker's LFP export from SLEEP_CHANNELS_<rat>
        (cortex / sr / pyr tetrodes), so the boxes start on the right layers for
        this rat instead of the generic 1/2/3. Typing over a box still wins.
        """
        self.sleep_channels = {}
        p = find_output(folder, "sleep_channels.npy")
        if p is None:
            return
        try:
            sc = np.load(p, allow_pickle=True).item()
        except Exception as exc:
            print(f"Warning: could not read {p}: {exc}")
            return
        if not isinstance(sc, dict):
            return

        self.sleep_channels = sc
        filled = []
        for e, (role, label) in zip(self.ch_edits, SLEEP_ROLE_LABELS):
            ch = sc.get(role)
            if ch is None:
                continue
            e.setText(str(int(ch)))
            e.setToolTip(f"{label} — tetrode {int(ch)} from SLEEP_CHANNELS_<rat>")
            filled.append(f"{label} {int(ch)}")
        if filled:
            self._set_status(f"Channels set from sleep_channels.npy: "
                             f"{', '.join(filled)}.", OK_GREEN)

    def _channel_labels(self, chs):
        """Display name per entered channel: the SLEEP_CHANNELS role it matches,
        else None so the editor falls back to ``Ch <n>``. Matching by value (not
        by slot) keeps the names honest when a box is typed over or reordered."""
        by_ch = {int(ch): label for role, label in SLEEP_ROLE_LABELS
                 if (ch := self.sleep_channels.get(role)) is not None}
        return [by_ch.get(int(c)) for c in chs]

    def _sel_lfp(self):
        folder = QFileDialog.getExistingDirectory(self, "Select LFP Output Folder")
        if not folder:
            return
        self.lfp_folder = folder
        self.lfp_edit.setText(folder)
        self.lfp_source = find_lfp_source(folder)
        if self.lfp_source is None:
            self._set_status("Warning: no .nwb (or lfp_data.npy / channels_npy/) found here.", WARN)
            self.info_label.setText("")
            self._set_dot(self.lfp_dot, "warn")
        else:
            self._set_status("Folder loaded. Enter channels, then Launch.", INK)
            self._set_dot(self.lfp_dot, "ok")
            self._show_recording_info(folder)
            self._prefill_sleep_channels(folder)
            # Auto-detect the rat_sessiondate_ prefix from the folder's files and
            # use it as the session name, so saved files share the session naming.
            from processing import output_prefix
            pfx = output_prefix(folder)
            if pfx:
                self.name_edit.setText(pfx.rstrip("_"))
                self._set_status(f"Folder loaded. Session: {pfx.rstrip('_')}. "
                                 f"Enter channels, then Launch.", INK)

        self.emg_file = ""
        for cand in EMG_CANDIDATES:
            p = find_output(folder, cand)          # prefixed (rat_sessiondate_) or not
            if p is not None:
                self.emg_file = str(p)
                self.emg_edit.setText(str(p))
                self.emg_auto.setText(f"✓ Auto-detected: {os.path.basename(p)}")
                self.emg_auto.setStyleSheet(f"color: {OK_GREEN}; font-size: 11px;")
                self._set_dot(self.emg_dot, "ok")
                break
        # No loose .npy? The session NWB carries motion/EMG too, so scoring can
        # run off that single file alone.
        self.motion_from_nwb = False
        if not self.emg_file:
            import sleep_nwb_store as snwb
            found = snwb.find_session_nwb(folder)
            if found is not None:
                inputs = None
                try:
                    inputs = snwb.read_sleep_inputs(found)
                    self.motion_from_nwb = (inputs.get("motion") is not None
                                            or inputs.get("emg") is not None)
                except Exception as exc:
                    print(f"Warning: could not read motion from the NWB: {exc}")
                finally:
                    snwb.close_inputs(inputs)
            if self.motion_from_nwb:
                self.emg_auto.setText(f"✓ Using motion/EMG from {os.path.basename(str(found))}")
                self.emg_auto.setStyleSheet(f"color: {OK_GREEN}; font-size: 11px;")
                self._set_dot(self.emg_dot, "ok")
        if not self.emg_file and not self.motion_from_nwb:
            self.emg_auto.setText("No EMG file auto-detected — browse for one manually.")
            self.emg_auto.setStyleSheet(f"color: {WARN}; font-size: 11px;")
            self._set_dot(self.emg_dot, "warn")
        if not self.out_folder:
            self.out_folder = folder
            self.out_edit.setText(folder)
            self._set_dot(self.out_dot, "ok")

        # the session NWB written by tracker step 8 — the store for scorings
        import sleep_nwb_store as snwb
        found = snwb.find_session_nwb(folder)
        self.nwb_path = str(found) if found else ""
        self._reload_scorings()
        self._update_ready()

    def _sel_emg(self):
        start = self.lfp_folder or os.getcwd()
        f, _ = QFileDialog.getOpenFileName(self, "Select Motion / EMG File", start,
                                           "NumPy (*.npy)")
        if not f:
            return
        self.emg_file = f
        self.emg_edit.setText(f)
        self.emg_auto.setText(f"✓ Set manually: {os.path.basename(f)}")
        self.emg_auto.setStyleSheet("color: #0060c0; font-size: 11px;")
        self._set_dot(self.emg_dot, "ok")
        self._update_ready()

    def _sel_out(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output / Save Folder")
        if folder:
            self.out_folder = folder
            self.out_edit.setText(folder)
            self._set_dot(self.out_dot, "ok")
            self._update_ready()

    def _reload_scorings(self):
        """Refill the "Reload state scored by" dropdown from the session NWB.

        Always starts blank (a fresh scoring) so nothing is resumed by accident;
        every scorer already stored in the NWB follows, newest first.
        """
        import sleep_nwb_store as snwb

        self.scorings = []
        self.prev_scoring = None
        self.prev_combo.blockSignals(True)
        self.prev_combo.clear()
        self.prev_combo.addItem("")                     # blank = start fresh
        try:
            self.scorings = snwb.list_scorings(self.nwb_path) if self.nwb_path else []
        except Exception as exc:
            print(f"Warning: could not list scorings: {exc}")
        for entry in self.scorings:
            self.prev_combo.addItem(entry["label"])
        self.prev_combo.setCurrentIndex(0)
        self.prev_combo.blockSignals(False)
        self._set_dot(self.prev_dot, "off")

        if not self.lfp_folder:
            self.prev_hint.setText(
                "Select the session folder first — scorings are read from its .nwb.")
        elif not self.nwb_path:
            self.prev_hint.setText(
                "No .nwb here — run tracker step 8 to create it. "
                "Scoring still works; results are saved to the results/ folder.")
        elif self.scorings:
            self.prev_hint.setText(
                f"{len(self.scorings)} scoring(s) in {os.path.basename(self.nwb_path)} "
                f"— pick one to continue it, or leave blank for a fresh one.")
        else:
            self.prev_hint.setText(
                f"No scorings yet in {os.path.basename(self.nwb_path)} — "
                f"this will be the first.")

    def _on_prev_pick(self, index):
        """Remember which stored scoring to resume (index 0 = start fresh)."""
        self.prev_scoring = (self.scorings[index - 1]
                             if 1 <= index <= len(self.scorings) else None)
        self._set_dot(self.prev_dot, "ok" if self.prev_scoring else "off")
        if self.prev_scoring:
            self.prev_hint.setText(
                f"Continuing {self.prev_scoring.get('scorer')}'s scoring — saving "
                f"updates that entry, and you won't be asked for a name.")

    # ------------------------------------------------------------------ launch
    def _launch(self):
        if not self.lfp_folder:
            return self._set_status("Error: select a session folder.", ERR)
        if not self.emg_file and not self.motion_from_nwb:
            return self._set_status("Error: select a motion/EMG file.", ERR)
        if not self.out_folder:
            return self._set_status("Error: select an output folder.", ERR)
        try:
            eeg_fs = float(self.fs_edit.text())
            assert eeg_fs > 0
        except (ValueError, AssertionError):
            return self._set_status("Error: sampling rate must be positive.", ERR)

        chs = []
        for k, e in enumerate(self.ch_edits):
            try:
                val = int(e.text().strip())
                assert val >= 1
            except (ValueError, AssertionError):
                return self._set_status(f"Error: Ch {k + 1} must be a whole number >= 1.", ERR)
            chs.append(val)
        if len(set(chs)) < 3:
            return self._set_status("Error: all 3 channels must differ.", ERR)

        base = self.name_edit.text().strip() or "session"
        # Prefix saved files (-states.mat, cache) with the session's rat_sessiondate_
        # already on the LFP folder, so every generated file shares one naming.
        # The Name field is auto-filled with the detected session token on folder
        # select, so skip re-prefixing when it already carries that token.
        from processing import output_prefix
        pfx = output_prefix(self.lfp_folder)
        if pfx and not base.startswith(pfx) and not base.startswith(pfx.rstrip("_")):
            base = f"{pfx}{base}"
        try:
            self._run(chs, eeg_fs, base)
        except Exception as exc:  # surface any load/compute failure in the GUI
            self._set_status(f"Error: {exc}", ERR)
            raise

    def _run(self, chs, eeg_fs, base):
        source = self.lfp_source or find_lfp_source(self.lfp_folder)
        if source is None:
            return self._set_status(
                f"Error: no .nwb, lfp_data.npy or channels_npy/ found in {self.lfp_folder}", ERR)

        cpath = cache_path(self.out_folder, base)
        cached = None
        if not self.recompute_chk.isChecked() and os.path.isfile(cpath):
            self._set_status("Loading cached spectrograms ...", "#0000aa")
            cached = load_cache(cpath, chs, eeg_fs)

        if cached is not None:
            specs, fos, to, raw_eeg, motion = cached
        else:
            available = set(source["channels"])
            for c in chs:
                if c not in available:
                    return self._set_status(
                        f"Error: channel {c} not available "
                        f"(have {source['channels'][0]}-{source['channels'][-1]}).", ERR)

            specs, fos, raw_eeg = [], [], []
            to = None
            for n, c in enumerate(chs, 1):
                self._set_status(
                    f"Preprocessing + spectrogram, channel {c} ({n}/{len(chs)}) ...",
                    "#0000aa")
                sig = load_lfp_channel(source, c)
                spec, fo, to, cleaned = compute_channel_spectrogram(sig, eeg_fs)
                specs.append(spec)
                fos.append(fo)
                raw_eeg.append(cleaned)

            mode = MOTION_MODES.get(self.motion_combo.currentText(), "accelerometer")
            self._set_status(f"Loading + processing motion ({mode}) ...", "#0000aa")
            motion_raw = (np.load(self.emg_file, mmap_mode="r") if self.emg_file
                          else self._motion_from_nwb())
            motion = process_motion(motion_raw, raw_eeg[0].size, eeg_fs, mode=mode)
            if motion.size != to.size:
                motion = np.interp(np.linspace(0, 1, to.size),
                                   np.linspace(0, 1, motion.size), motion.ravel())

            self._set_status("Caching spectrograms for fast reload ...", "#0000aa")
            try:
                save_cache(cpath, chs, eeg_fs, specs, fos, to, raw_eeg, motion)
            except Exception as exc:
                print(f"Warning: could not write cache: {exc}")

        auto_states, auto_ts = self._buzsaki_labels(chs, eeg_fs)
        overlays = self._load_overlays()

        # Continuing a stored scoring? Load its labels and carry its scorer name
        # through, so the editor neither asks for a name nor starts a new entry.
        states, labeled_by = None, None
        if self.prev_scoring:
            import sleep_nwb_store as snwb
            scorer = self.prev_scoring.get("scorer")
            self._set_status(f"Loading {scorer}'s scoring from "
                             f"{os.path.basename(self.nwb_path)} ...", "#0000aa")
            try:
                states, _, _ = snwb.read_scoring(
                    self.nwb_path, name=self.prev_scoring["name"], timestamps=to)
                labeled_by = scorer
            except Exception as exc:
                print(f"Warning: could not load the stored scoring: {exc}")
                states = None

        self._set_status("Launching state editor ...", OK_GREEN)
        editor = StateEditor(base, specs, fos, to, motion, raw_eeg, eeg_fs,
                             out_folder=self.out_folder, chs=chs,
                             ch_labels=self._channel_labels(chs),
                             auto_states=auto_states, auto_states_ts=auto_ts,
                             overlays=overlays, states=states,
                             labeled_by=labeled_by, nwb_path=self.nwb_path,
                             results_folder=os.path.join(self.lfp_folder, "results"))
        self.hide()
        editor.show()
        self.show()
        self._reload_scorings()      # the scoring just saved joins the dropdown
        saved = getattr(editor, "results_path", None)
        scorer = getattr(editor, "labeled_by", None)
        if getattr(editor, "nwb_saved", False):
            msg = (f"State editor closed. {scorer}'s scoring saved into "
                   f"{os.path.basename(self.nwb_path)}.")
        elif saved:
            msg = f"State editor closed. Scoring saved to {os.path.basename(saved)}."
        else:
            msg = "State editor closed."
        self._set_status(msg, OK_GREEN)

    def _motion_from_nwb(self):
        """Movement trace out of the session NWB (accelerometer, else EMG).

        Used when the folder has no motion ``.npy`` — everything the scorer
        needs is then read from the one NWB.
        """
        import sleep_nwb_store as snwb

        inputs = None
        try:
            inputs = snwb.read_sleep_inputs(self.nwb_path)
            for key in ("motion", "emg"):
                if inputs.get(key) is not None:
                    return np.asarray(inputs[key], dtype=np.float64).ravel()
            raise ValueError("the NWB has neither motion nor emg_from_lfp")
        finally:
            snwb.close_inputs(inputs)

    def _buzsaki_labels(self, chs, eeg_fs):
        """Return (states, timestamps) Buzsáki auto-labels to show, or (None, None).

        Recomputes fresh from the LFP folder every time (so scorer changes always
        take effect) and overwrites buzsaki_states.npz. Runs at the GUI's
        validated LFP rate ``eeg_fs`` (never the raw rate a bad
        lfp_timestamps.npy may imply). Only falls back to a saved npz if the
        recompute fails.
        """
        if not self.buzsaki_chk.isChecked():
            return None, None
        import buzsaki_score as bz

        def _f(edit, default):
            try:
                return float(edit.text())
            except ValueError:
                return default
        kw = dict(sw_factor=_f(self.swf_edit, bz.SW_THRESH_FACTOR),
                  th_factor=_f(self.thf_edit, bz.TH_THRESH_FACTOR),
                  emg_factor=_f(self.emgf_edit, 1.0),
                  min_secs=_f(self.minsec_edit, 10.0))
        try:
            self._set_status("Computing Buzsáki auto-score ...", "#0000aa")
            res, ch = bz.score_from_lfp_output(self.lfp_folder, channel=chs[0],
                                               fs=eeg_fs, **kw)
            scorer = res.get("model") or "threshold scorer"
            # Into the session NWB, beside the manual scorings but in its own
            # container — so it is never offered in the "Reload state scored by"
            # dropdown, and no buzsaki_states.npz is left lying around.
            self._save_auto_states(res, scorer, ch)
            self._set_status(f"Auto-score recomputed ({scorer}).", OK_GREEN)
            return res["states"], res["timestamps"]
        except Exception as exc:
            import traceback
            traceback.print_exc()          # full details on the console
            saved = self._load_auto_states()
            if saved[0] is not None:
                self._set_status(f"Compute failed; loaded the stored auto-score "
                                 f"({exc})", WARN)
                return saved
            for folder in (self.lfp_folder, self.out_folder):
                f = find_output(folder, bz.DEFAULT_OUT)   # legacy .npz, if any
                if f is not None:
                    self._set_status(f"Compute failed; loaded saved labels ({exc})", WARN)
                    return bz.load_states(f)
            self._set_status(f"Buzsáki auto-score skipped: {exc}", WARN)
            return None, None

    def _save_auto_states(self, res, scorer, channel):
        """Store the auto-score in the session NWB (no-op without one)."""
        if not self.nwb_path:
            return
        try:
            import sleep_nwb_store as snwb
            snwb.write_auto_states(self.nwb_path, res["states"], res["timestamps"],
                                   source=scorer,
                                   extra={"channel": int(channel)})
        except Exception as exc:
            print(f"Warning: could not store the auto-score in the NWB: {exc}")

    def _load_auto_states(self):
        """The auto-score already stored in the session NWB, else (None, None)."""
        if not self.nwb_path:
            return None, None
        try:
            import sleep_nwb_store as snwb
            states, ts, _ = snwb.read_auto_states(self.nwb_path)
            return states, ts
        except Exception as exc:
            print(f"Warning: could not read the stored auto-score: {exc}")
            return None, None

    def _load_overlays(self):
        """Load EMG-from-LFP + theta/delta signals to overlay on the motion axis.

        Returns a list of (label, values, timestamps|None); each is z-scored and
        aligned to the session's 1 s bins by the editor. Missing files are skipped.
        (Awakeness is not overlaid — it mixes EMG with theta/delta, so it reads as
        high in REM. We show theta/delta itself instead: high in REM + active WAKE,
        low in NREM — cleanly complementing the EMG/motion traces.)
        """
        overlays = []
        emg = ets = td = None

        # Prefer the session NWB — it is the only source for a folder that holds
        # nothing but the .nwb.
        if self.nwb_path:
            import sleep_nwb_store as snwb
            inputs = None
            try:
                inputs = snwb.read_sleep_inputs(self.nwb_path)
                if inputs.get("emg") is not None:
                    emg = np.asarray(inputs["emg"]).ravel()
                    ets = inputs.get("emg_timestamps")
                series = inputs.get("derived", {}).get("theta_delta_ratio")
                if series is not None:
                    td = np.asarray(series[::100], dtype=float)
            except Exception as exc:
                print(f"Warning: could not read overlays from the NWB: {exc}")
            finally:
                snwb.close_inputs(inputs)

        if emg is None:                      # legacy .npy (5 Hz, own timestamps)
            ef = find_output(self.lfp_folder, "emg_from_lfp_5hz.npy") or \
                find_output(self.lfp_folder, "emg_from_lfp.npy")
            if ef is not None:
                try:
                    emg = np.load(ef).ravel()
                    tsf = find_output(self.lfp_folder, "emg_from_lfp_timestamps.npy")
                    ets = np.load(tsf).ravel() if tsf is not None else None
                except Exception as exc:
                    print(f"Warning: could not load EMG overlay: {exc}")
        if emg is not None:
            overlays.append(("EMG", emg, ets))

        if td is None:
            tf = find_output(self.lfp_folder, "theta_delta_ratio.npy")
            if tf is not None:
                try:
                    td = np.asarray(np.load(tf, mmap_mode="r")[::100], dtype=float)
                except Exception as exc:
                    print(f"Warning: could not load theta/delta overlay: {exc}")
        # log-scaled (heavily skewed), decimated above, and smoothed (~15 s) so
        # the NREM/REM trend is legible.
        if td is not None:
            from scipy.ndimage import uniform_filter1d
            td = np.log10(np.clip(td, 1e-6, None))
            td = uniform_filter1d(td, size=225, mode="nearest")  # ~15 s @ 15 Hz
            overlays.append(("theta/delta", td, None))
        return overlays


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    try:                       # use the real macOS system UI font (San Francisco)
        from PyQt6.QtGui import QFontDatabase
        app.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont))
    except Exception:
        pass
    gui = SetupGUI()
    gui.show()
    gui.raise_()
    gui.activateWindow()
    if _VIBRANCY:                # opt-in real behind-window blur (HM_VIBRANCY=1)
        apply_vibrancy(gui)
    app.exec()


if __name__ == "__main__":
    main()
