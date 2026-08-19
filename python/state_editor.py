"""Interactive sleep-state editor (Python port of ``TheStateEditor.m``).

Displays, for up to three LFP channels, a whitened multitaper spectrogram, a
motion/EMG trace and the raw LFP, plus a colour-coded state bar.  States are
scored at 1 s resolution by arming a state (keys 0-5) and clicking the two time
bounds.  Work is saved to a MATLAB-compatible ``<base>-states.mat`` file so it
interoperates with the original MATLAB toolkit.

State codes:  0 none, 1 awake, 2 light/drowsy, 3 NREM, 4 intermediate, 5 REM.
Every bin starts as NREM (3) by default, so scoring means re-labelling the
non-NREM stretches rather than covering the whole recording.

On launch the editor asks who is scoring ("Labeled by"); on close it saves the
scoring automatically to a ``results/`` subfolder of the input folder as
``results_<date>_<name>.npz`` / ``.mat``.
"""

from __future__ import annotations

import datetime as _datetime
import os

import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from scipy.io import loadmat, savemat
from scipy.signal.windows import hann

from processing import matlab_round
from mac_vibrancy import apply_vibrancy

# The editor window is Qt (PyQt6). Importing Qt needs no running QApplication,
# so this is safe even in headless tests (which set MPLBACKEND=Agg and never
# build a window — see _setup_backend).
try:
    from PyQt6.QtCore import Qt, QEventLoop
    from PyQt6.QtGui import QAction
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QMessageBox,
                                 QToolBar, QFileDialog, QSlider, QComboBox,
                                 QInputDialog,
                                 QWidget as _QWidget, QSizePolicy as _QSizePolicy,
                                 QPushButton as _QPushButton, QLabel as _QLabel)
    _HAVE_QT = True
except Exception:                                   # pragma: no cover
    _HAVE_QT = False

# State -> RGB (0-1), taken from TheStateEditor.m
STATE_COLORS = {
    0: np.array([1.00, 1.00, 1.00]),            # white  - no state
    1: np.array([0.00, 0.00, 0.00]),            # black  - awake
    2: np.array([255, 236, 79]) / 255.0,        # yellow - light/drowsy
    3: np.array([6, 113, 148]) / 255.0,         # blue   - NREM
    4: np.array([19, 166, 50]) / 255.0,         # green  - intermediate
    5: np.array([207, 46, 49]) / 255.0,         # red    - REM
}
STATE_NAMES = {0: "none", 1: "awake", 2: "light/drowsy",
               3: "NREM", 4: "intermediate", 5: "REM"}
DEFAULT_STATE = 3     # every bin starts as NREM; scoring re-labels the rest

MAX_FREQ = 60.0       # default visible frequency extent (Hz)
HANNING_W = 10        # temporal smoothing window for the spectrogram
RESOLUTION = 0.5      # frequency binning resolution (Hz)
DOWNSAMPLE = 4        # plot every Nth LFP sample
PAN_FRAC = 0.15       # fraction of the window moved by arrow keys
MIN_VIEW_WINDOW = 10.0  # smallest main-view window (s)
MIN_EPOCH_S = 10.0      # smallest scored epoch (s) — manual assignments span >= 10 s
EEG_STEPS = [0.25, 0.5, 1, 2, 5, 15, 30, 60]   # '-'/'=' LFP width steps
ARROW_STEP = 1        # bins the time cursor moves per ← → press (1 bin = 1 s)
EMG_SENS = 0.8        # EMG 1/0-band threshold multiplier (<1 = more sensitive)
# Spectrogram colormaps offered in the toolbar picker: the modern perceptually-
# uniform maps first, then the classics. Filtered at build time against what
# this matplotlib actually provides.
CMAP_CHOICES = ["turbo", "viridis", "plasma", "inferno", "magma", "cividis",
                "twilight", "twilight_shifted", "cubehelix", "nipy_spectral",
                "gist_ncar", "rainbow", "jet", "coolwarm", "Spectral",
                "hot", "cool", "bone", "gray"]
_VIBRANCY = bool(os.environ.get("HM_VIBRANCY"))   # opt-in macOS behind-window blur

# Apple-style chrome for the editor's Qt toolbars.
_TB_STYLE = (
    "QToolBar{background:rgba(246,247,250,70%); border:none;"
    " border-bottom:1px solid rgba(60,60,67,10%); padding:6px 10px; spacing:3px;}"
    "QToolBar::separator{background:rgba(60,60,67,14%); width:1px; margin:4px 6px;}"
    "QToolBar QLabel{color:#55555A; font-size:12px; background:transparent;}"
    "QToolButton{padding:5px 12px; border-radius:7px; font-size:13px; color:#1D1D1F;}"
    "QToolButton:hover{background:rgba(0,0,0,7%);}"
    "QToolButton:pressed{background:rgba(0,0,0,12%);}")
_SLIDER_STYLE = (
    "QSlider::groove:horizontal{height:4px; background:#D8D8DC; border-radius:2px;}"
    "QSlider::sub-page:horizontal{background:#007AFF; border-radius:2px;}"
    "QSlider::add-page:horizontal{background:#D8D8DC; border-radius:2px;}"
    "QSlider::handle:horizontal{width:16px; height:16px; margin:-6px 0; border-radius:8px;"
    " background:#FFFFFF; border:1px solid #C7C7CC;}"
    "QSlider::handle:horizontal:hover{border:1px solid #A9A9AF;}")

HELP_LINES = [
    ("1-5", "arm a state (then Space Space to score an epoch)"),
    ("← →", "move the time cursor (1 s per press)"),
    ("Space", "confirm epoch bound (1st = start, 2nd = apply)"),
    ("0", "arm 'no state' (erase)"),
    ("c", "cancel the armed state"),
    ("click", "move the time cursor here (never scores)"),
    ("Shift+← →", "pan the window left / right"),
    ("Home / End", "jump to start / end"),
    ("scroll", "zoom in / out around the cursor"),
    ("↑ ↓", "spectrogram contrast up / down"),
    ("- / =", "narrow / widen the LFP window"),
    ("r", "reset time axis to full extent"),
    ("u", "undo last state change"),
    ("", ""),
    ("e", "toggle add-event mode; click to drop a mark"),
    ("d", "toggle delete-event mode; click near a mark"),
    ("[ / ]", "previous / next event number (1-10)"),
    ("n / p", "jump to next / previous event"),
    ("", ""),
    ("s / l", "save / load states + events"),
    ("h", "toggle this help"),
]

EVENT_COLOR = "magenta"


def _fmt_hms(x, _pos=None):
    """Format a time in seconds as h:mm:ss (or mm:ss under an hour) for the main
    time axis, so long overnight recordings read in hours at a glance."""
    x = max(0.0, float(x))
    h, rem = divmod(int(round(x)), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


if _HAVE_QT:
    class _EditorWindow(QMainWindow):
        """Top-level Qt window hosting the editor's matplotlib canvas.

        Closing it auto-saves the scoring to the results folder and stops the
        nested event loop that ``StateEditor.show()`` spins."""

        def __init__(self, editor):
            super().__init__()
            self._editor = editor
            self._loop = None

        def closeEvent(self, ev):
            if not self._editor._confirm_close():
                ev.ignore()
                return
            ev.accept()
            if self._loop is not None:
                self._loop.quit()
                self._loop = None


class StateEditor:
    def __init__(self, base_name, specs, fos, to, motion, raw_eeg, eeg_fs,
                 out_folder=".", states=None, chs=None, ch_labels=None,
                 auto_states=None, auto_states_ts=None,
                 auto_label="Auto", overlays=None,
                 labeled_by=None, results_folder=None):
        self.base_name = base_name
        self.eeg_fs = float(eeg_fs)
        self.out_folder = out_folder
        # Auto-save on close goes here (results/ inside the input folder).
        self.results_folder = results_folder or os.path.join(out_folder, "results")
        self.labeled_by = labeled_by       # scorer name, asked on launch if unset
        self.n_ch = len(specs)
        self.chs = list(chs) if chs is not None else list(range(1, self.n_ch + 1))
        # Anatomical panel names (cortex / EEG / pyr) resolved from
        # SLEEP_CHANNELS_<rat>; falls back to the bare channel number.
        self.ch_labels = self._resolve_ch_labels(ch_labels)
        self.to = np.asarray(to, dtype=float)
        self.lims = (float(self.to[0]), float(self.to[-1]))
        self.n_bins = self.to.size

        self.states = (np.full(self.n_bins, DEFAULT_STATE, dtype=int)
                       if states is None
                       else np.asarray(states, dtype=int).copy())
        self.history = []          # for undo

        # --- optional auto-scored (Buzsáki) labels, shown in an extra panel ----
        self.auto_label = auto_label
        self.auto_states = self._align_auto_states(auto_states, auto_states_ts)

        # --- display-ready spectrograms (bin freq, smooth time, log) --------
        self.spec_disp, self.fo = self._prepare_specs(specs, fos)

        # --- LFP traces (downsampled, scaled like MATLAB) -------------------
        self.eeg = [(np.asarray(e, dtype=float)[::DOWNSAMPLE] / 2150.0) / 1000.0
                    for e in raw_eeg]
        self.eeg_x = np.arange(1, self.eeg[0].size + 1) / (self.eeg_fs / DOWNSAMPLE)
        self.eeg_show = 2.0        # seconds of LFP shown

        # --- motion (z-scored) ----------------------------------------------
        m = np.asarray(motion, dtype=float).ravel()
        if m.size != self.n_bins:                      # be tolerant of length
            m = np.interp(np.linspace(0, 1, self.n_bins),
                          np.linspace(0, 1, m.size), m)
        self.motion = self._zscore(m)

        # --- extra overlays on the motion axis (EMG, theta/delta, ...) -------
        # Each is aligned to the 1 s bins and z-scored so several signals share
        # one y-axis. ``overlays`` is a list of (label, values, timestamps|None).
        self.overlays = []
        for label, vals, ts in (overlays or []):
            try:
                self.overlays.append((label, self._zscore(self._align_signal(vals, ts))))
            except Exception as exc:
                print(f"Warning: could not add overlay {label}: {exc}")

        # --- scoring interaction state --------------------------------------
        self.current_state = None
        self.pending_bound = None
        self.pending_line = []
        self.dirty = False          # unsaved changes?
        self.help_visible = False
        # movable time cursor (← → step it, space confirms epoch bounds)
        self._dt = float(np.median(np.diff(self.to))) if self.n_bins > 1 else 1.0
        self.cursor_time = float((self.lims[0] + self.lims[1]) / 2)
        self._slider_guard = False

        # --- event marking state --------------------------------------------
        self.events = []            # list of [event_num, time_s]
        self.event_num = 1          # currently active event number (1-10)
        self.event_mode = None      # None | 'add' | 'delete'
        self.event_artists = []     # drawn vertical lines + labels

        self._setup_backend()
        self._build_figure()
        self._build_side_panel()
        self._connect()

    def _resolve_ch_labels(self, ch_labels):
        """One display name per scored channel.

        ``ch_labels`` comes from SLEEP_CHANNELS_<rat> (cortex / EEG / pyr). Any
        slot without a name — no config, or a channel the user typed over —
        falls back to ``Ch <n>`` so a panel is never left unidentified.
        """
        given = list(ch_labels) if ch_labels else []
        out = []
        for i, ch in enumerate(self.chs):
            name = given[i] if i < len(given) else None
            out.append(str(name) if name else f"Ch {ch}")
        return out

    def _panel_title(self, i):
        """Axis label for scored channel ``i``: ``cortex (24)``, or ``Ch 24``
        when no anatomical name is known (the number is already in there)."""
        label = self.ch_labels[i]
        return label if label == f"Ch {self.chs[i]}" else f"{label} ({self.chs[i]})"

    def _align_auto_states(self, auto_states, auto_states_ts):
        """Resample provided auto-labels onto this session's 1 s bins (nearest)."""
        if auto_states is None:
            return None
        a = np.asarray(auto_states, dtype=int).ravel()
        if a.size == self.n_bins and auto_states_ts is None:
            return a.copy()
        if auto_states_ts is not None:
            ts = np.asarray(auto_states_ts, dtype=float).ravel()
            idx = np.clip(np.searchsorted(ts, self.to), 0, ts.size - 1)
            left = np.clip(idx - 1, 0, ts.size - 1)
            choose_left = np.abs(ts[left] - self.to) < np.abs(ts[idx] - self.to)
            idx = np.where(choose_left, left, idx)
            return a[idx]
        src = np.linspace(0, 1, a.size)
        dst = np.linspace(0, 1, self.n_bins)
        return a[np.clip(np.searchsorted(src, dst), 0, a.size - 1)]

    @staticmethod
    def _zscore(x):
        """Z-score, ignoring NaNs (returns a copy)."""
        x = np.asarray(x, dtype=float).copy()
        valid = ~np.isnan(x)
        if valid.any():
            x[valid] = (x[valid] - x[valid].mean()) / (x[valid].std() + 1e-12)
        return x

    def _align_signal(self, values, ts=None):
        """Resample a signal onto this session's 1 s bins (self.to).

        With ``ts`` (same length as ``values``) it interpolates on real time;
        otherwise it assumes the signal spans the whole recording uniformly.
        """
        v = np.asarray(values, dtype=float).ravel()
        if ts is not None:
            ts = np.asarray(ts, dtype=float).ravel()
            n = min(v.size, ts.size)
            return np.interp(self.to, ts[:n], v[:n])
        if v.size == self.n_bins:
            return v
        return np.interp(np.linspace(0.0, 1.0, self.n_bins),
                         np.linspace(0.0, 1.0, v.size), v)

    # ------------------------------------------------------------------ setup
    def _prepare_specs(self, specs, fos):
        """Bin frequency, smooth across time, log-scale; normalise channels."""
        fo0 = np.asarray(fos[0], dtype=float)
        df = np.median(np.diff(fo0))
        f1 = max(1, matlab_round(np.sum((fo0 >= 2) & (fo0 <= 4)) / (2.0 / RESOLUTION)))

        win = hann(HANNING_W)
        win = win / win.sum()
        out = []
        binned_fo = None
        for spec in specs:                                  # spec: (n_freq, n_time)
            n_freq = spec.shape[0]
            rows, fo_centres = [], []
            for i in range(0, n_freq - f1, f1):
                rows.append(spec[i:i + f1, :].mean(axis=0))
                fo_centres.append(fo0[i:i + f1].mean())
            binned = np.array(rows)                         # (n_bf, n_time)
            # temporal smoothing of each frequency row
            smoothed = np.array([np.convolve(r, win, mode="same") for r in binned])
            out.append(np.log(np.clip(smoothed, 1e-300, None)))
            binned_fo = np.array(fo_centres)

        # Normalise channels 2..n to the dynamic range of channel 1 (<= MAX_FREQ)
        if self.n_ch > 1:
            mask = binned_fo <= MAX_FREQ
            min1 = out[0][mask, :].min()
            max1 = out[0][mask, :].max()
            for i in range(1, self.n_ch):
                s = out[i]
                min2 = s[mask, :].min()
                s = s - min2
                max2 = s[mask, :].max()
                s = s / (max2 + 1e-12)
                out[i] = s * (max1 - min1) + min1
        return out, binned_fo

    def _setup_backend(self):
        """Create the Figure and attach a canvas.

        Interactive: a Qt window (``_EditorWindow``) with a ``FigureCanvasQTAgg``.
        Headless (``MPLBACKEND=Agg``, e.g. tests, or no Qt): a bare Agg canvas
        and no window, so the editor can be built and driven without a display."""
        self.fig = Figure(figsize=(16, 9.6), dpi=110)
        self.fig.patch.set_facecolor("#f4f5f7")
        headless = os.environ.get("MPLBACKEND", "").lower() == "agg" or not _HAVE_QT
        if headless:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            FigureCanvasAgg(self.fig)               # sets self.fig.canvas
            self.win = None
        else:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            self.win = _EditorWindow(self)
            win_bg = "transparent" if _VIBRANCY else (
                "qlineargradient(x1:0, y1:0, x2:1, y2:1,"
                " stop:0 #F2F5FB, stop:1 #F6F0F5)")
            self.win.setStyleSheet(f"QMainWindow{{background:{win_bg};}}")
            canvas = FigureCanvasQTAgg(self.fig)    # sets self.fig.canvas
            self.win.setCentralWidget(canvas)
            self.win.resize(1500, 900)
            canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.canvas = self.fig.canvas
            self._build_toolbar()
            return
        self.canvas = self.fig.canvas

    def _build_toolbar(self):
        """A slim toolbar of the most-used actions, mirroring the keyboard
        shortcuts. Each action re-focuses the canvas so the single-key shortcuts
        keep working afterwards."""
        tb = QToolBar("Scoring", self.win)
        tb.setMovable(False)
        tb.setStyleSheet(_TB_STYLE)
        self.win.addToolBar(tb)

        def add(text, fn, tip):
            a = QAction(text, self.win)
            a.setToolTip(tip)
            a.triggered.connect(lambda: (fn(), self.canvas.setFocus()))
            tb.addAction(a)

        add("💾  Save .npy", self.save_states_npz, "Save scoring as NumPy .npz")
        add("💾  Save .mat", self.save_states, "Save scoring as MATLAB -states.mat  (s)")
        add("📂  Load", self.load_states, "Load a saved .npz / .mat scoring  (l)")
        tb.addSeparator()
        add("↺  Reset view", lambda: self._set_xlim(*self.lims), "Reset time axis to full extent  (r)")
        add("⟲  Undo", self._undo, "Undo last state change  (u)")
        tb.addSeparator()
        add("❔  Help", self._toggle_help, "Toggle the keyboard/mouse help overlay  (h)")

        hint = _QLabel("  then Space Space to mark an epoch  ·  press h for help  ")
        hint.setStyleSheet("color:#7a8189; font-size:12px;")
        tb.addWidget(hint)
        # push the auto↔manual agreement readout to the far right of the toolbar
        spacer = _QWidget()
        spacer.setSizePolicy(_QSizePolicy.Policy.Expanding, _QSizePolicy.Policy.Preferred)
        spacer.setStyleSheet("background:transparent;")
        tb.addWidget(spacer)
        self.match_lbl = _QLabel("")
        self.match_lbl.setToolTip(
            "Agreement between your manual scoring and the auto (Buzsáki) labels over "
            "the bins you've scored.\nκ = Cohen's kappa (chance-corrected): "
            "<0 poor · .2–.4 fair · .4–.6 moderate · .6–.8 substantial · >.8 near-perfect")
        self.match_lbl.setStyleSheet("color:#1D1D1F; font-size:12px; font-weight:600;"
                                     " padding-right:10px;")
        tb.addWidget(self.match_lbl)
        self.win.statusBar().setStyleSheet("color:#5a6069;")
        self.win.statusBar().showMessage("Ready — no unsaved changes")

        # second row: colour-coded state-arm toggle buttons (click to arm, click
        # again to un-arm). They stay in sync with the 0–5 / c keyboard shortcuts.
        states_tb = QToolBar("States", self.win)
        states_tb.setMovable(False)
        states_tb.setStyleSheet(_TB_STYLE + "QToolBar{spacing:6px;}")
        self.win.addToolBarBreak()
        self.win.addToolBar(states_tb)
        arm_lbl = _QLabel("  Arm state: ")
        arm_lbl.setStyleSheet("color:#55555A; font-size:12px; font-weight:600;")
        states_tb.addWidget(arm_lbl)
        labels = {0: "erase", 1: "awake", 2: "light", 3: "NREM",
                  4: "interm", 5: "REM"}
        self._state_btns = {}
        for s in (1, 2, 3, 4, 5, 0):
            r, g, b = (STATE_COLORS[s] * 255).astype(int)
            fg = "#ffffff" if (0.299 * r + 0.587 * g + 0.114 * b) < 140 else "#111111"
            btn = _QPushButton(f"{s}  {labels[s]}")
            btn.setCheckable(True)
            btn.setToolTip(f"Arm state {s} ({STATE_NAMES[s]}) — click again to un-arm  (key {s})")
            btn.setStyleSheet(
                f"QPushButton{{background:rgb({r},{g},{b}); color:{fg};"
                f" border:1px solid rgba(0,0,0,0.12); border-radius:8px;"
                f" padding:6px 13px; font-size:12px;}}"
                f"QPushButton:checked{{border:2.5px solid #007AFF; font-weight:700;}}")
            btn.clicked.connect(lambda _checked, st=s: self._arm_state_toggle(st))
            states_tb.addWidget(btn)
            self._state_btns[s] = btn

        # live stats readout, to the right of the state buttons (replaces the old
        # right-margin legend + info panel).
        states_tb.addSeparator()
        self.stats_lbl = _QLabel("")
        self.stats_lbl.setStyleSheet("color:#1D1D1F; font-size:12px; padding-left:8px;")
        states_tb.addWidget(self.stats_lbl)

    def _arm_state_toggle(self, s):
        """Toggle-arm state ``s`` from a button: arm it, or un-arm if already
        armed. Mirrors pressing key ``s`` (then ``c`` to cancel)."""
        self.current_state = None if self.current_state == s else s
        self.event_mode = None
        self.pending_bound = None
        self._clear_pending_line()
        self._set_title()
        self.canvas.setFocus()

    def _sync_state_buttons(self):
        """Reflect the armed state on the buttons (keeps keyboard + buttons in
        sync)."""
        if not hasattr(self, "_state_btns"):
            return
        for s, btn in self._state_btns.items():
            btn.setChecked(self.current_state == s)

    def _set_window_title(self, text):
        if getattr(self, "win", None) is not None:
            self.win.setWindowTitle(text)

    def _build_figure(self):
        # free the editor's single-key shortcuts from matplotlib's default keymap
        # (← → s p h r l etc. are otherwise hijacked for nav / save / log-scale)
        for key, keep in [("keymap.back", ["backspace"]), ("keymap.forward", ["v"]),
                          ("keymap.home", []), ("keymap.pan", []),
                          ("keymap.save", ["ctrl+s"]), ("keymap.yscale", []),
                          ("keymap.xscale", []), ("keymap.zoom", [])]:
            matplotlib.rcParams[key] = keep
        matplotlib.rcParams.update({
            "font.family": "sans-serif", "font.size": 9,
            "axes.titlesize": 10, "axes.labelsize": 9,
            "axes.edgecolor": "#8a8f98", "axes.linewidth": 0.8,
            "xtick.color": "#4a4f57", "ytick.color": "#4a4f57",
            "xtick.labelsize": 8, "ytick.labelsize": 8,
            "figure.facecolor": "#f4f5f7", "axes.facecolor": "#ffffff",
        })
        self._set_window_title(f"Sleep scoring — {self.base_name}")

        n = self.n_ch
        left, width = 0.065, 0.905
        # vertical layout: (auto bar) / state bar / spectrograms / motion / LFP traces
        has_auto = self.auto_states is not None
        if has_auto:
            self.ax_state = self.fig.add_axes([left, 0.944, width, 0.046])
            self.ax_auto = self.fig.add_axes([left, 0.888, width, 0.046])
            spec_top = 0.872
        else:
            self.ax_state = self.fig.add_axes([left, 0.940, width, 0.050])
            self.ax_auto = None
            spec_top = 0.918
        spec_h = (spec_top - 0.34) / n
        self.ax_spec, self.spec_imgs = [], []
        for i in range(n):
            y = spec_top - (i + 1) * spec_h
            self.ax_spec.append(self.fig.add_axes([left, y, width, spec_h - 0.005]))
        self.ax_motion = self.fig.add_axes([left, 0.225, width, 0.08])
        # LFP stack fills the space freed by moving the sliders up to the toolbar.
        eeg_h = 0.05
        eeg_top = 0.16
        self.ax_eeg = [self.fig.add_axes([left, eeg_top - i * (eeg_h + 0.012),
                                          width, eeg_h]) for i in range(n)]

        self._draw_state_bar()
        if has_auto:
            self._draw_auto_bar()

        cmap = "turbo"      # modern perceptually-uniform rainbow (jet-like)
        self.cursor_lines = []
        mid = (self.lims[0] + self.lims[1]) / 2
        for i, ax in enumerate(self.ax_spec):
            mask = self.fo <= MAX_FREQ
            img = ax.imshow(
                self.spec_disp[i][mask, :], origin="lower", aspect="auto",
                extent=[self.to[0], self.to[-1], self.fo[mask][0], self.fo[mask][-1]],
                cmap=cmap)
            self.spec_imgs.append(img)
            ax.set_ylabel(f"{self._panel_title(i)}\nFreq (Hz)")
            ax.set_xlim(self.lims)
            if i != n - 1:
                ax.set_xticklabels([])
            else:
                # main time axis: label ticks as h:mm:ss (long recordings read
                # in hours). The underlying coordinates stay in seconds.
                ax.xaxis.set_major_formatter(FuncFormatter(_fmt_hms))
            self.cursor_lines.append(ax.axvline(mid, color="w", ls="--", lw=0.8))

        # remember the auto colour limits, display band and cmap for the toolbar
        # controls (frequency-band / contrast sliders + colormap picker).
        self._spec_clim0 = [img.get_clim() for img in self.spec_imgs]
        self._fmax = MAX_FREQ
        self._contrast = 1.0

        # Motion / EMG / theta-delta as stacked "ridgeline" lanes, each thresholded at
        # its bimodal-histogram dip and shown as a bold 1/0 band — filled above
        # threshold (=1), blank below (=0) — plus a faint trace + dotted threshold line.
        #   * Motion -> smoothed |Δ| (movement onsets stand out; drift removed).
        #   * EMG / theta-delta -> LEVEL (already ~0 when quiet; |Δ| would wrongly
        #     zero out sustained tonic EMG and flat high-theta REM).
        # Colour-blind-safe colours (Okabe–Ito): Motion & EMG high = wake; theta/delta
        # high (others low) = REM; all low = NREM.
        import buzsaki_score as _bz
        from scipy.ndimage import uniform_filter1d
        MOTION_PALETTE = ["#009e73", "#d55e00", "#0072b2", "#cc79a7"]
        traces = [("Motion", self.motion)] + list(self.overlays)
        yticks = []
        self.motion_thresholds = {}
        for i, (label, a) in enumerate(traces):
            color = MOTION_PALETTE[i % len(MOTION_PALETTE)]
            base = float(i)
            sig = np.asarray(a, dtype=float)
            metric = (uniform_filter1d(np.abs(np.diff(sig, prepend=sig[:1])),
                                       size=7, mode="nearest")
                      if label.lower() == "motion" else sig)
            finite = metric[np.isfinite(metric)]
            lo, hi = (np.percentile(finite, 1), np.percentile(finite, 99)) \
                if finite.size else (0.0, 1.0)
            span_i = (hi - lo) or 1.0
            norm = np.clip((metric - lo) / span_i, 0.0, 1.0) * 0.88
            default = float(np.median(finite)) if finite.size else np.inf
            thr = (_bz.bimodal_threshold(finite, default=default)
                   if finite.size else np.inf)
            # EMG: same bimodal algorithm, just a bit more sensitive (× EMG_SENS < 1).
            if label.lower() == "emg":
                thr *= EMG_SENS
            self.motion_thresholds[label] = float(thr)
            on = metric > thr
            thr_y = base + float(np.clip((thr - lo) / span_i, 0.0, 1.0)) * 0.88
            self.ax_motion.axhline(base, color="#e2e5e9", lw=0.5, zorder=1)
            # 1/0 band: lightly filled where the signal is above its threshold
            self.ax_motion.fill_between(self.to, base, base + 0.88, where=on,
                                        step="mid", color=color, alpha=0.3,
                                        linewidth=0, zorder=2)
            # continuous trace + dotted threshold line for reference
            self.ax_motion.plot(self.to, base + norm, color=color, lw=0.6,
                                alpha=0.55, zorder=3)
            self.ax_motion.axhline(thr_y, color=color, ls=":", lw=0.7, alpha=0.9,
                                   zorder=4)
            yticks.append(base + 0.44)
        self.ax_motion.set_ylim(0, len(traces))
        self.ax_motion.set_yticks(yticks)
        self.ax_motion.set_yticklabels([lbl for lbl, _ in traces], fontsize=7.5)
        for i, tl in enumerate(self.ax_motion.get_yticklabels()):
            tl.set_color(MOTION_PALETTE[i % len(MOTION_PALETTE)])
            tl.set_fontweight("bold")
        self.ax_motion.tick_params(axis="y", length=0)
        self.ax_motion.set_xlim(self.lims)
        self.ax_motion.set_xticklabels([])
        self.cursor_lines.append(self.ax_motion.axvline(mid, color="k", ls="--", lw=0.8))

        self.eeg_lines, self.eeg_cursor, self.eeg_yabs = [], [], []
        self.eeg_scale = 1.0                       # raw-trace amplitude gain
        for i, ax in enumerate(self.ax_eeg):
            (ln,) = ax.plot([], [], color="y", lw=0.5)
            self.eeg_lines.append(ln)
            ax.set_facecolor("black")
            ax.set_ylabel(self._panel_title(i), fontsize=8.5, rotation=0,
                          ha="right", va="center", labelpad=6)
            yabs = np.percentile(np.abs(self.eeg[i]), 99.5) or 1.0
            self.eeg_yabs.append(yabs)
            ax.set_ylim(-yabs, yabs)
            ax.set_yticks([])                          # raw trace: no y-axis numbers
            # vertical cursor marking the current time (centre), matching the
            # spectrogram cursor above.
            self.eeg_cursor.append(ax.axvline(mid, color="w", ls="--", lw=0.8))
            if i != n - 1:
                ax.set_xticklabels([])
        self.ax_eeg[-1].set_xlabel("Time (s)")

        self._update_eeg(mid)
        if getattr(self, "win", None) is not None:
            self._build_slider_toolbar()
        self._set_title()

    def _build_slider_toolbar(self):
        """View controls as a top Qt toolbar row: main time window + position,
        and the raw-LFP window + gain. Replaces the old bottom matplotlib sliders."""
        span = self.lims[1] - self.lims[0]
        self._win_min = min(10.0, span) if span > 0 else 1.0
        self._sc_win = self._sc_pos = 1.0        # 1 unit = 1 s
        self._sc_raw = self._sc_gain = 10.0      # 1 unit = 0.1

        tb = QToolBar("View", self.win)
        tb.setMovable(False)
        tb.setStyleSheet(_TB_STYLE + _SLIDER_STYLE)
        self.win.addToolBarBreak()
        self.win.addToolBar(tb)

        self._slider_guard = True                # muffle the initial setValue events
        self.win_slider, self._lbl_win = self._add_qslider(
            tb, "Window", self._win_min, span, span, self._sc_win, "%.0f s",
            self._on_win_slider, 180)
        self.pos_slider, self._lbl_pos = self._add_qslider(
            tb, "Position", self.lims[0], self.lims[1], self.lims[0], self._sc_pos,
            "%.0f s", self._on_pos_slider, 180)
        tb.addSeparator()
        self.rawwin_slider, self._lbl_raw = self._add_qslider(
            tb, "Raw win", 0.5, min(30.0, span) if span > 0 else 30.0, self.eeg_show,
            self._sc_raw, "%.1f s", self._on_rawwin_slider, 120)
        self.rawgain_slider, self._lbl_gain = self._add_qslider(
            tb, "Raw gain", 0.2, 10.0, self.eeg_scale, self._sc_gain, "%.1f",
            self._on_rawgain_slider, 120)

        # second row — spectrogram display: frequency band + colormap depth + cmap
        tb2 = QToolBar("Display", self.win)
        tb2.setMovable(False)
        tb2.setStyleSheet(_TB_STYLE + _SLIDER_STYLE)
        self.win.addToolBarBreak()
        self.win.addToolBar(tb2)
        fmax_avail = float(np.nanmax(self.fo)) if self.fo.size else 200.0
        self.freq_slider, self._lbl_freq = self._add_qslider(
            tb2, "Freq band", 10.0, fmax_avail, self._fmax, 1.0, "%.0f Hz",
            self._set_freq_band, 200)
        self.contrast_slider, self._lbl_contrast = self._add_qslider(
            tb2, "Depth", 0.5, 3.0, self._contrast, 10.0, "%.1f×",
            self._on_contrast, 160)
        tb2.addSeparator()
        cap = _QLabel("  Colormap ")
        cap.setStyleSheet("font-weight:600; padding-left:6px;")
        tb2.addWidget(cap)
        cmap_combo = QComboBox()
        cmap_combo.addItems([c for c in CMAP_CHOICES if c in matplotlib.colormaps])
        # explicit text colours everywhere: the forced white backgrounds would
        # otherwise render the names white-on-white under macOS dark mode
        cmap_combo.setStyleSheet(
            "QComboBox{background:#FFFFFF; color:#1D1D1F; border:1px solid #D1D1D6;"
            " border-radius:7px; padding:3px 10px; font-size:12px; min-width:110px;}"
            "QComboBox::drop-down{border:none; width:18px;}"
            "QComboBox QAbstractItemView{background:#FFFFFF; color:#1D1D1F;"
            " border:1px solid #E4E4E8;"
            " selection-background-color:#007AFF; selection-color:#fff; outline:none;}")
        cmap_combo.currentTextChanged.connect(lambda name: (self._on_cmap(name),
                                                            self.canvas.setFocus()))
        tb2.addWidget(cmap_combo)
        self._slider_guard = False

    def _add_qslider(self, tb, text, lo, hi, init, scale, fmt, fn, width):
        """Add a captioned horizontal QSlider (+ live value label) to a toolbar.
        Values are scaled to integers for QSlider and mapped back to floats."""
        cap = _QLabel(f"  {text} ")
        cap.setStyleSheet("font-weight:600; padding-left:6px;")
        tb.addWidget(cap)
        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setFixedWidth(width)
        imin, imax = int(round(lo * scale)), int(round(hi * scale))
        sld.setMinimum(imin)
        sld.setMaximum(max(imax, imin + 1))
        sld.setValue(int(round(min(max(init, lo), hi) * scale)))
        vlbl = _QLabel(fmt % init)
        vlbl.setStyleSheet("color:#333; min-width:46px;")
        sld.valueChanged.connect(
            lambda iv, s=scale, v=vlbl, f=fmt, g=fn: self._on_qslider(iv, s, v, f, g))
        tb.addWidget(sld)
        tb.addWidget(vlbl)
        return sld, vlbl

    def _on_qslider(self, iv, scale, vlbl, fmt, fn):
        val = iv / scale
        vlbl.setText(fmt % val)
        if self._slider_guard:
            return
        fn(val)

    def _set_qslider(self, slider, scale, value, vlbl, fmt):
        iv = int(round(value * scale))
        iv = max(slider.minimum(), min(slider.maximum(), iv))
        slider.setValue(iv)
        vlbl.setText(fmt % value)

    # -- spectrogram display controls -------------------------------------
    def _set_freq_band(self, fmax):
        """Show the spectrograms up to ``fmax`` Hz (re-slices the display data)."""
        mask = self.fo <= fmax
        if int(mask.sum()) < 2:
            return
        self._fmax = float(fmax)
        ylo, yhi = float(self.fo[mask][0]), float(self.fo[mask][-1])
        for i, img in enumerate(self.spec_imgs):
            img.set_data(self.spec_disp[i][mask, :])
            img.set_extent([self.to[0], self.to[-1], ylo, yhi])
            self.ax_spec[i].set_ylim(ylo, yhi)
        self.fig.canvas.draw_idle()

    def _on_contrast(self, k):
        """Colormap 'depth': compress/expand the colour limits about their centre
        (higher = more contrast / deeper colours)."""
        self._contrast = max(float(k), 1e-3)
        for img, (v0, v1) in zip(self.spec_imgs, self._spec_clim0):
            c = (v0 + v1) / 2.0
            half = (v1 - v0) / 2.0 / self._contrast
            img.set_clim(c - half, c + half)
        self.fig.canvas.draw_idle()

    def _on_cmap(self, name):
        for img in self.spec_imgs:
            img.set_cmap(name)
        self.fig.canvas.draw_idle()

    def _raw_centre(self):
        return float(np.mean(self.ax_eeg[0].get_xlim()))

    def _on_rawwin_slider(self, val):
        if self._slider_guard:
            return
        self.eeg_show = float(val)
        self._update_eeg(self._raw_centre())
        self.fig.canvas.draw_idle()

    def _on_rawgain_slider(self, val):
        if self._slider_guard:
            return
        self.eeg_scale = max(float(val), 1e-6)
        self._update_eeg(self._raw_centre())
        self.fig.canvas.draw_idle()

    def _on_win_slider(self, val):
        if self._slider_guard:
            return
        lo, hi = self._xlim_get()
        centre = (lo + hi) / 2
        half = float(val) / 2
        self._set_xlim(centre - half, centre + half)

    def _on_pos_slider(self, val):
        if self._slider_guard:
            return
        lo, hi = self._xlim_get()
        width = hi - lo                         # keep the current window size
        start = min(max(float(val), self.lims[0]), self.lims[1] - width)
        self._set_xlim(start, start + width)

    def _sync_sliders(self, lo, hi):
        """Reflect the current x-limits back onto the sliders (no feedback loop)."""
        if not hasattr(self, "win_slider"):
            return
        self._slider_guard = True
        try:
            self._set_qslider(self.win_slider, self._sc_win, hi - lo, self._lbl_win, "%.0f s")
            self._set_qslider(self.pos_slider, self._sc_pos, lo, self._lbl_pos, "%.0f s")
        finally:
            self._slider_guard = False

    # --------------------------------------------------------------- side panel
    def _build_side_panel(self):
        """Only the (hidden) full-figure help overlay now lives here — the state
        legend is replaced by the coloured toolbar buttons, and the live stats by
        the toolbar readout beside them."""
        lines = ["Keyboard / mouse controls", ""]
        lines += [f"{k:>10}   {d}" for k, d in HELP_LINES]
        self.help_overlay = self.fig.text(
            0.5, 0.5, "\n".join(lines), ha="center", va="center",
            fontsize=12, family="monospace", visible=False, zorder=100,
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#ffffe0",
                      edgecolor="0.3"))

    def _coverage(self):
        scored = int(np.count_nonzero(self.states))
        return scored, self.n_bins

    def _update_info(self):
        """Refresh the compact live stats shown beside the state buttons."""
        if not hasattr(self, "stats_lbl"):
            return
        scored, total = self._coverage()
        pct = 100.0 * scored / total if total else 0.0
        if self.current_state is not None:
            armed = f"{self.current_state} {STATE_NAMES[self.current_state]}"
        elif self.event_mode:
            armed = f"{self.event_mode} ev{self.event_num}"
        else:
            armed = "—"
        counts = "   ".join(
            f"{lab} {int(np.count_nonzero(self.states == s))}"
            for s, lab in ((1, "W"), (2, "L"), (3, "N"), (4, "I"), (5, "R")))
        self.stats_lbl.setText(
            f"armed: {armed}      scored {pct:.1f}%      {counts}      "
            f"{len(self.events)} ev")

    def _toggle_help(self):
        self.help_visible = not self.help_visible
        self.help_overlay.set_visible(self.help_visible)
        self.fig.canvas.draw_idle()

    def _confirm_close(self):
        """Auto-save the scoring to the results folder, then allow the close.

        Called from ``_EditorWindow.closeEvent``. If the save fails the user is
        asked whether to close anyway (No keeps the window open)."""
        if not _HAVE_QT or self.win is None:
            return True
        try:
            self.save_results()
        except Exception as exc:
            resp = QMessageBox.question(
                self.win, "Save failed",
                f"Could not save results:\n{exc}\n\nClose anyway (scoring will "
                f"be lost)?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            return resp == QMessageBox.StandardButton.Yes
        return True

    def _mark_dirty(self):
        self.dirty = True
        self._set_title()

    def _draw_state_bar(self):
        self.ax_state.clear()
        self._plot_hypnogram(self.ax_state, self.states)
        self.ax_state.set_yticks([1, 2, 3, 4, 5])
        self.ax_state.set_yticklabels(["W", "L", "N", "I", "R"], fontsize=7)
        self.ax_state.set_xticks([])
        self.ax_state.set_ylabel("Manual", fontsize=8.5, fontweight="bold",
                                 rotation=0, ha="right", va="center", labelpad=8)
        self.ax_state.set_xlim(getattr(self, "_xlim", self.lims))

    def _plot_hypnogram(self, ax, states):
        """Draw a stepped hypnogram (staircase) of ``states`` on ``ax``.

        Mirrors TheStateEditor's original look: a step line running between the
        state levels, with each horizontal run drawn in its state colour and
        unscored (0) bins left as gaps.
        """
        y = np.asarray(states, dtype=float).copy()
        y[y == 0] = np.nan                              # unscored -> gap
        t = np.asarray(self.to, dtype=float)
        # grey staircase outline (the vertical transitions between levels)
        ax.step(t, y, where="post", color="0.55", lw=0.8, zorder=1)
        # coloured horizontal segment per contiguous run
        for s, e, v in self._state_runs(states):
            if v == 0:
                continue
            x1 = t[e] if e < t.size else t[-1]
            ax.hlines(v, t[s], x1, color=STATE_COLORS[v], lw=2.4, zorder=2)
        ax.set_ylim(0.5, 5.5)

    @staticmethod
    def _state_runs(states):
        """Yield (start, end_exclusive, value) for each contiguous run."""
        states = np.asarray(states)
        n = states.size
        if n == 0:
            return
        start = 0
        for i in range(1, n + 1):
            if i == n or states[i] != states[start]:
                yield start, i, int(states[start])
                start = i

    def _draw_auto_bar(self):
        """Stepped hypnogram of the provided auto-scored (Buzsáki) labels."""
        if self.ax_auto is None or self.auto_states is None:
            return
        self.ax_auto.clear()
        self._plot_hypnogram(self.ax_auto, self.auto_states)
        self.ax_auto.set_yticks([1, 2, 3, 4, 5])
        self.ax_auto.set_yticklabels(["W", "L", "N", "I", "R"], fontsize=7)
        self.ax_auto.set_xticks([])
        self.ax_auto.set_ylabel(self.auto_label, fontsize=8.5, fontweight="bold",
                                rotation=0, ha="right", va="center", labelpad=8)
        self.ax_auto.set_xlim(getattr(self, "_xlim", self.lims))

    # ------------------------------------------------------------------ events
    def _connect(self):
        c = self.fig.canvas
        c.mpl_connect("key_press_event", self._on_key)
        c.mpl_connect("button_press_event", self._on_click)
        c.mpl_connect("scroll_event", self._on_scroll)

    def _set_title(self):
        extra = ""
        if self.current_state is not None:
            extra = f" - Add State {self.current_state} ({STATE_NAMES[self.current_state]})"
        elif self.event_mode == "add":
            extra = f" - Add Event {self.event_num} (click to place)"
        elif self.event_mode == "delete":
            extra = f" - Delete Event {self.event_num} (click near a mark)"
        star = "*" if getattr(self, "dirty", False) else ""
        self._set_window_title(f"States: {self.base_name}{star}{extra}")
        self._sync_state_buttons()
        self._update_match()
        self._update_statusbar()
        if hasattr(self, "stats_lbl"):
            self._update_info()
            self.fig.canvas.draw_idle()

    @staticmethod
    def _cohens_kappa(a, b):
        """Cohen's κ between two label vectors — agreement corrected for chance."""
        a = np.asarray(a)
        b = np.asarray(b)
        if a.size == 0:
            return np.nan
        po = float(np.mean(a == b))                         # observed agreement
        labels = np.unique(np.concatenate([a, b]))
        pe = float(sum(np.mean(a == c) * np.mean(b == c) for c in labels))  # chance
        if pe >= 1.0:                                       # single shared category
            return 1.0 if po >= 1.0 else np.nan
        return (po - pe) / (1.0 - pe)

    def _update_match(self):
        """Show how well the manual scoring agrees with the auto (Buzsáki) labels,
        over the bins scored so far — raw % and Cohen's κ — in the toolbar readout."""
        if not hasattr(self, "match_lbl"):
            return
        if self.auto_states is None:
            self.match_lbl.setText("")
            return
        scored = self.states != 0
        n = int(scored.sum())
        if n == 0:
            self.match_lbl.setText("Auto ↔ Manual:  —  ")
            return
        m, a = self.states[scored], self.auto_states[scored]
        agree = float(np.mean(m == a)) * 100.0
        kappa = self._cohens_kappa(m, a)
        ktxt = f"κ={kappa:.2f}" if np.isfinite(kappa) else "κ=n/a"
        self.match_lbl.setText(f"Auto ↔ Manual:  {agree:.0f}%   {ktxt}   (n={n})  ")

    def _update_statusbar(self):
        """Reflect scored coverage and dirty state in the Qt status bar."""
        if getattr(self, "win", None) is None:
            return
        scored, total = self._coverage()
        pct = 100.0 * scored / total if total else 0.0
        flag = "● unsaved changes" if self.dirty else "✓ saved"
        self.win.statusBar().showMessage(f"{flag}   ·   scored {pct:.1f}%   ·   "
                                         f"{len(self.events)} event(s)")

    def _xlim_get(self):
        return self.ax_spec[0].get_xlim()

    def _set_xlim(self, lo, hi):
        lo = max(lo, self.lims[0])
        hi = min(hi, self.lims[1])
        # enforce a minimum view window: each scored epoch spans >= 10 s
        min_win = min(MIN_VIEW_WINDOW, self.lims[1] - self.lims[0])
        if hi - lo < min_win:
            c = (lo + hi) / 2
            lo, hi = c - min_win / 2, c + min_win / 2
            if lo < self.lims[0]:
                lo, hi = self.lims[0], self.lims[0] + min_win
            elif hi > self.lims[1]:
                lo, hi = self.lims[1] - min_win, self.lims[1]
        self._xlim = (lo, hi)
        for ax in self.ax_spec:
            ax.set_xlim(lo, hi)
        self.ax_motion.set_xlim(lo, hi)
        self.ax_state.set_xlim(lo, hi)
        if self.ax_auto is not None:
            self.ax_auto.set_xlim(lo, hi)
        # keep the time cursor in view, then redraw the LFP at the cursor
        self.cursor_time = float(np.clip(self.cursor_time, lo, hi))
        self._update_eeg(self.cursor_time)
        self._sync_sliders(lo, hi)
        if hasattr(self, "stats_lbl"):
            self._update_info()
        self.fig.canvas.draw_idle()

    def _update_eeg(self, centre):
        low = centre - self.eeg_show / 2
        high = centre + self.eeg_show / 2
        if low < self.lims[0]:
            high += self.lims[0] - low
            low = self.lims[0]
        elif high > self.lims[1]:
            low -= high - self.lims[1]
            high = self.lims[1]
        m = (self.eeg_x >= low - 1) & (self.eeg_x <= high + 1)
        for i, ln in enumerate(self.eeg_lines):
            ln.set_data(self.eeg_x[m], self.eeg[i][m])
            self.ax_eeg[i].set_xlim(low, high)
            yl = self.eeg_yabs[i] / self.eeg_scale
            self.ax_eeg[i].set_ylim(-yl, yl)
            self.eeg_cursor[i].set_xdata([centre, centre])
        for ln in self.cursor_lines:
            ln.set_xdata([centre, centre])

    def _move_cursor(self, direction, steps=1):
        """Step the time cursor by ``steps`` bins (← →), scrolling to follow."""
        self.cursor_time = float(np.clip(self.cursor_time + direction * steps * self._dt,
                                         self.lims[0], self.lims[1]))
        lo, hi = self._xlim_get()
        w = hi - lo
        if self.cursor_time < lo:
            self._set_xlim(self.cursor_time, self.cursor_time + w)
        elif self.cursor_time > hi:
            self._set_xlim(self.cursor_time - w, self.cursor_time)
        else:
            self._update_eeg(self.cursor_time)
            if hasattr(self, "stats_lbl"):
                self._update_info()
            self.fig.canvas.draw_idle()

    def _confirm_boundary(self):
        """Space: confirm the cursor as an epoch bound. 1st = start, 2nd = apply.

        With a state armed, number → space → space brackets an epoch (a double
        space at one spot assigns a single ≥10 s epoch there)."""
        if self.current_state is None:
            return
        if self.pending_bound is None:
            self.pending_bound = self.cursor_time
            for ax in self.ax_spec + [self.ax_motion, self.ax_state]:
                self.pending_line.append(ax.axvline(self.cursor_time, color="r", lw=1.0))
            self.fig.canvas.draw_idle()
        else:
            self._apply_state(self.pending_bound, self.cursor_time, self.current_state)
            self.pending_bound = None
            self._clear_pending_line()

    def _on_key(self, event):
        k = event.key
        if k in "012345":
            self.current_state = int(k)
            self.event_mode = None
            self.pending_bound = None
            self._clear_pending_line()
            self._set_title()
        elif k == "c":
            self.current_state = None
            self.event_mode = None
            self.pending_bound = None
            self._clear_pending_line()
            self._set_title()
        elif k == "e":
            self.event_mode = None if self.event_mode == "add" else "add"
            self.current_state = None
            self.pending_bound = None
            self._clear_pending_line()
            self._set_title()
        elif k == "d":
            self.event_mode = None if self.event_mode == "delete" else "delete"
            self.current_state = None
            self.pending_bound = None
            self._clear_pending_line()
            self._set_title()
        elif k in ("[", "]"):
            step = 1 if k == "]" else -1
            self.event_num = (self.event_num - 1 + step) % 10 + 1
            self._refresh_events()
            self._set_title()
        elif k in ("n", "p"):
            self._jump_event(forward=(k == "n"))
        elif k in ("right", "left"):
            self._move_cursor(1 if k == "right" else -1, steps=ARROW_STEP)
        elif k in (" ", "space"):
            self._confirm_boundary()
        elif k in ("shift+right", "shift+left"):    # coarse pan (whole window)
            lo, hi = self._xlim_get()
            step = (hi - lo) * (1 if k == "shift+right" else -1)
            self._set_xlim(lo + step, hi + step)
        elif k == "home":
            lo, hi = self._xlim_get()
            self._set_xlim(self.lims[0], self.lims[0] + (hi - lo))
        elif k == "end":
            lo, hi = self._xlim_get()
            self._set_xlim(self.lims[1] - (hi - lo), self.lims[1])
        elif k == "r":
            self._set_xlim(*self.lims)
        elif k in ("up", "down"):
            delta = -0.1 if k == "up" else 0.1
            for img in self.spec_imgs:
                lo, hi = img.get_clim()
                img.set_clim(lo + delta, hi + delta)
            self.fig.canvas.draw_idle()
        elif k in ("-", "="):
            self._change_eeg_width(k)
        elif k == "u":
            self._undo()
        elif k == "s":
            self.save_states()
        elif k == "l":
            self.load_states()
        elif k == "h":
            self._toggle_help()

    def _change_eeg_width(self, k):
        cur = self.eeg_show
        if k == "=":                       # widen
            nxt = [s for s in EEG_STEPS if s > cur + 1e-9]
            self.eeg_show = nxt[0] if nxt else EEG_STEPS[-1]
        else:                              # narrow
            prv = [s for s in EEG_STEPS if s < cur - 1e-9]
            self.eeg_show = prv[-1] if prv else EEG_STEPS[0]
        if hasattr(self, "rawwin_slider"):     # keep the slider in sync
            self._slider_guard = True
            try:
                self._set_qslider(self.rawwin_slider, self._sc_raw, self.eeg_show,
                                  self._lbl_raw, "%.1f s")
            finally:
                self._slider_guard = False
        self._update_eeg(np.mean(self.ax_eeg[0].get_xlim()))
        self.fig.canvas.draw_idle()

    def _on_scroll(self, event):
        if event.inaxes is None:
            return
        lo, hi = self._xlim_get()
        centre = event.xdata if event.xdata is not None else (lo + hi) / 2
        factor = 0.8 if event.button == "up" else 1.25
        half = (hi - lo) * factor / 2
        self._set_xlim(centre - half, centre + half)

    def _on_click(self, event):
        if event.inaxes is None or event.xdata is None:
            return
        in_panel = event.inaxes in self.ax_spec or event.inaxes is self.ax_motion \
            or event.inaxes is self.ax_state
        if not in_panel:
            return

        # event marking takes precedence when armed
        if self.event_mode == "add":
            self._add_event(event.xdata)
            return
        if self.event_mode == "delete":
            self._delete_event(event.xdata)
            return

        # A click only moves the time cursor (and re-centres the LFP view).
        # Epoch boundaries are confirmed with Space only, never by clicking.
        self.cursor_time = float(event.xdata)
        self._update_eeg(self.cursor_time)
        if hasattr(self, "stats_lbl"):
            self._update_info()
        self.fig.canvas.draw_idle()

    def _clear_pending_line(self):
        # A pending marker drawn on the state bar is wiped by ax_state.clear() when
        # the hypnogram redraws, so the artist may already be gone — remove
        # defensively (newer matplotlib raises instead of ignoring).
        for ln in self.pending_line:
            try:
                ln.remove()
            except (NotImplementedError, ValueError):
                pass
        self.pending_line = []
        self.fig.canvas.draw_idle()

    def _apply_state(self, t0, t1, state):
        i0 = max(0, matlab_round(t0 - self.to[0]))
        i1 = min(self.n_bins - 1, matlab_round(t1 - self.to[0]))
        if i1 < i0:
            return
        # enforce a minimum scored epoch of MIN_EPOCH_S seconds (expand about the
        # selection centre if the user picked a shorter span)
        dt = np.median(np.diff(self.to)) if self.n_bins > 1 else 1.0
        min_bins = min(self.n_bins, max(1, int(round(MIN_EPOCH_S / dt))))
        if (i1 - i0 + 1) < min_bins:
            centre = (i0 + i1) // 2
            i0 = max(0, min(centre - min_bins // 2, self.n_bins - min_bins))
            i1 = i0 + min_bins - 1
        self.history.append((i0, i1, self.states[i0:i1 + 1].copy()))
        self.states[i0:i1 + 1] = state
        self._mark_dirty()
        self._refresh_state_bar()

    def _undo(self):
        if not self.history:
            return
        i0, i1, prev = self.history.pop()
        self.states[i0:i1 + 1] = prev
        self._mark_dirty()
        self._refresh_state_bar()

    def _refresh_state_bar(self):
        self._draw_state_bar()
        if hasattr(self, "stats_lbl"):
            self._update_info()
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------- events
    def _add_event(self, t):
        self.events.append([self.event_num, float(t)])
        self._mark_dirty()
        self._refresh_events()

    def _delete_event(self, t):
        """Delete the nearest event of the active number within 1% of the view."""
        lo, hi = self._xlim_get()
        tol = 0.01 * (hi - lo)
        candidates = [(abs(tm - t), i) for i, (en, tm) in enumerate(self.events)
                      if en == self.event_num]
        if not candidates:
            return
        dist, idx = min(candidates)
        if dist <= tol:
            self.events.pop(idx)
            self._mark_dirty()
            self._refresh_events()

    def _jump_event(self, forward=True):
        """Centre the view on the next/previous event of the active number."""
        times = sorted(tm for en, tm in self.events if en == self.event_num)
        if not times:
            return
        centre = np.mean(self._xlim_get())
        if forward:
            nxt = [t for t in times if t > centre + 1e-6]
            target = nxt[0] if nxt else times[-1]
        else:
            prv = [t for t in times if t < centre - 1e-6]
            target = prv[-1] if prv else times[0]
        lo, hi = self._xlim_get()
        half = (hi - lo) / 2
        self._set_xlim(target - half, target + half)

    def _refresh_events(self):
        """Redraw all event lines (active number bold, others faint)."""
        for a in self.event_artists:
            try:
                a.remove()
            except (NotImplementedError, ValueError):
                pass
        self.event_artists = []
        # not the state bar: it is cleared/redrawn whenever a state changes
        panels = self.ax_spec + [self.ax_motion]
        top_ax = self.ax_spec[0]
        for en, tm in self.events:
            active = (en == self.event_num)
            lw = 1.6 if active else 0.7
            alpha = 1.0 if active else 0.35
            for ax in panels:
                self.event_artists.append(
                    ax.axvline(tm, color=EVENT_COLOR, ls=":", lw=lw, alpha=alpha))
            self.event_artists.append(
                top_ax.text(tm, 0.98, str(en), transform=top_ax.get_xaxis_transform(),
                            color=EVENT_COLOR, fontsize=8, ha="center", va="top",
                            alpha=alpha, clip_on=True))
        if hasattr(self, "stats_lbl"):
            self._update_info()
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------ save/load
    def save_states(self, path=None):
        if path is None:
            path = os.path.join(self.out_folder, f"{self.base_name}-states.mat")
        transitions = self._compute_transitions()
        events = (np.array(self.events, dtype=float) if self.events
                  else np.zeros((0, 2)))
        savemat(path, {"states": self.states.astype(float).reshape(1, -1),
                       "events": events,
                       "transitions": transitions})
        print(f"Saved {path}  ({len(self.events)} events)")
        self.dirty = False
        self._set_title()
        return path

    def save_states_npz(self, path=None):
        """Save the scoring as a NumPy ``.npz`` (states, per-bin timestamps,
        events and transitions), a companion to the MATLAB ``-states.mat``.

        The ``.npz`` is what the setup GUI reloads to resume a partially scored
        session. Prompts for a path when run interactively with none given."""
        if path is None:
            default = os.path.join(self.out_folder, f"{self.base_name}-states.npz")
            path = default
            if _HAVE_QT and self.win is not None:
                chosen, _ = QFileDialog.getSaveFileName(
                    self.win, "Save scoring (NumPy)", default, "NumPy (*.npz)")
                if not chosen:
                    return None
                path = chosen
        if not path.endswith(".npz"):
            path += ".npz"
        events = (np.array(self.events, dtype=float) if self.events
                  else np.zeros((0, 2)))
        np.savez(path,
                 states=self.states.astype(int),
                 timestamps=self.to.astype(float),
                 events=events,
                 transitions=self._compute_transitions())
        print(f"Saved {path}  ({len(self.events)} events)")
        self.dirty = False
        self._set_title()
        return path

    def save_results(self):
        """Save the scoring to ``<results_folder>/results_<date>_<name>``.

        Written automatically when the editor window closes: both a NumPy
        ``.npz`` (resumable) and a MATLAB ``.mat``, stamped with the scoring
        date and the "Labeled by" name given on launch."""
        os.makedirs(self.results_folder, exist_ok=True)
        name = "_".join((self.labeled_by or "unknown").split()) or "unknown"
        stem = f"results_{_datetime.date.today().isoformat()}_{name}"
        events = (np.array(self.events, dtype=float) if self.events
                  else np.zeros((0, 2)))
        transitions = self._compute_transitions()
        npz_path = os.path.join(self.results_folder, stem + ".npz")
        np.savez(npz_path,
                 states=self.states.astype(int),
                 timestamps=self.to.astype(float),
                 events=events,
                 transitions=transitions,
                 labeled_by=np.array(self.labeled_by or "unknown"))
        mat_path = os.path.join(self.results_folder, stem + ".mat")
        savemat(mat_path, {"states": self.states.astype(float).reshape(1, -1),
                           "events": events,
                           "transitions": transitions,
                           "labeled_by": self.labeled_by or "unknown"})
        print(f"Saved results to {npz_path} (+ .mat)")
        self.dirty = False
        self._set_title()
        return npz_path

    def _compute_transitions(self):
        """Nx3 [state, start_s, end_s] from contiguous runs (MATLAB format)."""
        rows = []
        if self.n_bins == 0:
            return np.zeros((0, 3))
        start = 0
        for i in range(1, self.n_bins + 1):
            if i == self.n_bins or self.states[i] != self.states[start]:
                s = self.states[start]
                if s != 0:
                    rows.append([s, self.to[start], self.to[i - 1]])
                start = i
        return np.array(rows) if rows else np.zeros((0, 3))

    def load_states(self, path=None):
        """Load a previously saved scoring, from either a NumPy ``.npz`` or a
        MATLAB ``-states.mat`` (chosen by extension). Only applied when the bin
        count matches this session."""
        if path is None:
            path = os.path.join(self.out_folder, f"{self.base_name}-states.mat")
        if not os.path.isfile(path):
            print(f"No states file at {path}")
            return
        data = np.load(path) if path.endswith(".npz") else loadmat(path)
        if "states" in data:
            s = np.asarray(data["states"]).ravel().astype(int)
            if s.size == self.n_bins:
                self.states = s
                self.dirty = False
                self._refresh_state_bar()
                self._set_title()
                print(f"Loaded states from {path}")
            else:
                print(f"State length mismatch ({s.size} vs {self.n_bins}); ignored")
        if "events" in data:
            ev = np.asarray(data["events"], dtype=float)
            if ev.ndim == 2 and ev.shape[1] == 2:
                self.events = [[int(round(en)), float(tm)] for en, tm in ev]
                self._refresh_events()
                print(f"Loaded {len(self.events)} events")

    def _prompt_labeled_by(self):
        """Modal "Labeled by" prompt shown on launch. The name is required — it
        goes into the auto-saved results filename. Returns False on Cancel."""
        while True:
            name, ok = QInputDialog.getText(
                self.win, "Labeled by",
                "Who is scoring this session?\n"
                "(used in the saved results filename)")
            if not ok:
                return False
            name = name.strip()
            if name:
                self.labeled_by = name
                return True

    def show(self):
        """Show the editor window and block until it is closed.

        Asks for the scorer's name ("Labeled by") first — cancelling that
        prompt aborts opening the editor. Uses a nested Qt event loop so a
        caller (the setup GUI) can invoke this synchronously from within the
        already-running application loop, the same way ``QDialog.exec()``
        blocks. No-op when built headlessly."""
        if self.win is None:
            return
        owns_app = False
        if QApplication.instance() is None:         # standalone use
            self._app = QApplication([])
            owns_app = True
        if self.labeled_by is None and not self._prompt_labeled_by():
            return                                  # cancelled — don't open
        self.win.show()
        self.win.raise_()
        self.win.activateWindow()
        if _VIBRANCY:                     # opt-in real behind-window blur
            apply_vibrancy(self.win)
        self.canvas.setFocus()
        if owns_app:
            self._app.exec()
        else:
            loop = QEventLoop()
            self.win._loop = loop
            loop.exec()
