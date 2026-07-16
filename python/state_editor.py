"""Interactive sleep-state editor (Python port of ``TheStateEditor.m``).

Displays, for up to three LFP channels, a whitened multitaper spectrogram, a
motion/EMG trace and the raw LFP, plus a colour-coded state bar.  States are
scored at 1 s resolution by arming a state (keys 0-5) and clicking the two time
bounds.  Work is saved to a MATLAB-compatible ``<base>-states.mat`` file so it
interoperates with the original MATLAB toolkit.

State codes:  0 none, 1 awake, 2 light/drowsy, 3 NREM, 4 intermediate, 5 REM.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider
from scipy.io import loadmat, savemat
from scipy.signal.windows import hann

from processing import matlab_round

# The editor window is Qt (PyQt6). Importing Qt needs no running QApplication,
# so this is safe even in headless tests (which set MPLBACKEND=Agg and never
# build a window — see _setup_backend).
try:
    from PyQt6.QtCore import Qt, QEventLoop
    from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox
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

MAX_FREQ = 60.0       # default visible frequency extent (Hz)
HANNING_W = 10        # temporal smoothing window for the spectrogram
RESOLUTION = 0.5      # frequency binning resolution (Hz)
DOWNSAMPLE = 4        # plot every Nth LFP sample
PAN_FRAC = 0.15       # fraction of the window moved by arrow keys
MIN_VIEW_WINDOW = 10.0  # smallest main-view window (s)
MIN_EPOCH_S = 10.0      # smallest scored epoch (s) — manual assignments span >= 10 s
EEG_STEPS = [0.25, 0.5, 1, 2, 5, 15, 30, 60]   # '-'/'=' LFP width steps

HELP_LINES = [
    ("1-5", "arm a state (then Space Space to score an epoch)"),
    ("← →", "move the time cursor"),
    ("Space", "confirm epoch bound (1st = start, 2nd = apply)"),
    ("0", "arm 'no state' (erase)"),
    ("c", "cancel the armed state"),
    ("click", "move cursor here (or set a bound when armed)"),
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


if _HAVE_QT:
    class _EditorWindow(QMainWindow):
        """Top-level Qt window hosting the editor's matplotlib canvas.

        Closing it runs the editor's unsaved-changes prompt and stops the nested
        event loop that ``StateEditor.show()`` spins."""

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
                 out_folder=".", states=None, chs=None,
                 auto_states=None, auto_states_ts=None,
                 auto_label="Auto", overlays=None):
        self.base_name = base_name
        self.eeg_fs = float(eeg_fs)
        self.out_folder = out_folder
        self.n_ch = len(specs)
        self.chs = list(chs) if chs is not None else list(range(1, self.n_ch + 1))
        self.to = np.asarray(to, dtype=float)
        self.lims = (float(self.to[0]), float(self.to[-1]))
        self.n_bins = self.to.size

        self.states = (np.zeros(self.n_bins, dtype=int) if states is None
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

        # --- event marking state --------------------------------------------
        self.events = []            # list of [event_num, time_s]
        self.event_num = 1          # currently active event number (1-10)
        self.event_mode = None      # None | 'add' | 'delete'
        self.event_artists = []     # drawn vertical lines + labels

        self._setup_backend()
        self._build_figure()
        self._build_side_panel()
        self._connect()

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
            canvas = FigureCanvasQTAgg(self.fig)    # sets self.fig.canvas
            self.win.setCentralWidget(canvas)
            self.win.resize(1500, 900)
            canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.canvas = self.fig.canvas

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
        left, width = 0.065, 0.80
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
        self.ax_motion = self.fig.add_axes([left, 0.235, width, 0.07])
        # compact LFP stack anchored just below motion, leaving room at the very
        # bottom (< ~0.09) for the window/position sliders.
        eeg_h = 0.04
        eeg_top = 0.23 - eeg_h
        self.ax_eeg = [self.fig.add_axes([left, eeg_top - i * (eeg_h + 0.006),
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
            ax.set_ylabel(f"Ch {self.chs[i]}\nFreq (Hz)")
            ax.set_xlim(self.lims)
            if i != n - 1:
                ax.set_xticklabels([])
            self.cursor_lines.append(ax.axvline(mid, color="w", ls="--", lw=0.8))

        traces = [("Motion", self.motion, "#000000")] + [
            (label, a, c) for (label, a), c in zip(
                self.overlays, ["#c0392b", "#2980b9", "#16a085", "#8e44ad"])]
        for label, a, c in traces:
            self.ax_motion.plot(self.to, a, "-", color=c, lw=0.6, label=label)
        allv = np.concatenate([a[np.isfinite(a)] for _, a, _ in traces
                               if np.isfinite(a).any()]) if traces else np.array([0.0])
        if allv.size:
            self.ax_motion.set_ylim(np.percentile(allv, 1), np.percentile(allv, 99))
        self.ax_motion.set_ylabel("Motion · EMG\n" + r"$\theta/\delta$  (z)",
                                  fontsize=8)
        if self.overlays:
            self.ax_motion.legend(loc="upper right", fontsize=6, ncol=len(traces),
                                  framealpha=0.4, handlelength=1.0, columnspacing=1.0)
        self.ax_motion.set_xlim(self.lims)
        self.ax_motion.set_xticklabels([])
        self.cursor_lines.append(self.ax_motion.axvline(mid, color="k", ls="--", lw=0.8))

        self.eeg_lines, self.eeg_cursor, self.eeg_yabs = [], [], []
        self.eeg_scale = 1.0                       # raw-trace amplitude gain
        for i, ax in enumerate(self.ax_eeg):
            (ln,) = ax.plot([], [], color="y", lw=0.5)
            self.eeg_lines.append(ln)
            ax.set_facecolor("black")
            ax.set_ylabel(f"Ch {self.chs[i]}", fontsize=8.5, rotation=0,
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
        self._build_sliders()
        self._set_title()

    def _build_sliders(self):
        """Bottom sliders: LEFT = main time view (window size + sliding position);
        RIGHT = raw LFP trace (window size + amplitude gain)."""
        span = self.lims[1] - self.lims[0]
        self._win_min = min(10.0, span)
        self._slider_guard = False
        # left column — main spectrogram/state time view
        ax_win = self.fig.add_axes([0.11, 0.05, 0.30, 0.016])
        ax_pos = self.fig.add_axes([0.11, 0.02, 0.30, 0.016])
        self.win_slider = Slider(ax_win, "Window (s)", self._win_min, span,
                                 valinit=span, valfmt="%.0f")
        self.pos_slider = Slider(ax_pos, "Position (s)", self.lims[0], self.lims[1],
                                 valinit=self.lims[0], valfmt="%.0f")
        self.win_slider.on_changed(self._on_win_slider)
        self.pos_slider.on_changed(self._on_pos_slider)
        # right column — raw LFP trace controls
        ax_rw = self.fig.add_axes([0.60, 0.05, 0.22, 0.016])
        ax_rg = self.fig.add_axes([0.60, 0.02, 0.22, 0.016])
        self.rawwin_slider = Slider(ax_rw, "Raw win (s)", 0.5, min(30.0, span),
                                    valinit=self.eeg_show, valfmt="%.1f")
        self.rawgain_slider = Slider(ax_rg, "Raw gain", 0.2, 10.0,
                                     valinit=self.eeg_scale, valfmt="%.1f")
        self.rawwin_slider.on_changed(self._on_rawwin_slider)
        self.rawgain_slider.on_changed(self._on_rawgain_slider)
        for s in (self.win_slider, self.pos_slider, self.rawwin_slider,
                  self.rawgain_slider):
            s.label.set_fontsize(7)

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
            self.win_slider.set_val(np.clip(hi - lo, self.win_slider.valmin,
                                            self.win_slider.valmax))
            self.pos_slider.set_val(np.clip(lo, self.pos_slider.valmin,
                                            self.pos_slider.valmax))
        finally:
            self._slider_guard = False

    # --------------------------------------------------------------- side panel
    def _build_side_panel(self):
        """Right-margin legend + live info + (hidden) help overlay."""
        ax = self.fig.add_axes([0.865, 0.34, 0.125, 0.59])
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0.0, 0.99, "States", fontsize=11, fontweight="bold", va="top")
        for i, s in enumerate([1, 2, 3, 4, 5, 0]):
            y = 0.92 - i * 0.058
            ax.add_patch(Rectangle((0.0, y - 0.035), 0.16, 0.045,
                                   facecolor=STATE_COLORS[s],
                                   edgecolor="0.4", lw=0.6))
            ax.text(0.21, y - 0.013, f"{s}  {STATE_NAMES[s]}", fontsize=9, va="center")
        ax.text(0.0, 0.50, "Press 'h' for help", fontsize=8.5,
                style="italic", color="0.35", va="top")
        self.ax_side = ax

        # live info text (armed state, position, coverage)
        self.info_text = self.fig.text(0.865, 0.30, "", fontsize=9,
                                       va="top", family="monospace")
        self._update_info()

        # full-figure help overlay, hidden until toggled
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
        centre = np.mean(self.ax_spec[0].get_xlim()) if hasattr(self, "ax_spec") \
            else self.lims[0]
        scored, total = self._coverage()
        pct = 100.0 * scored / total if total else 0.0
        if self.current_state is not None:
            armed = f"state {self.current_state} ({STATE_NAMES[self.current_state]})"
        elif self.event_mode:
            armed = f"{self.event_mode} event {self.event_num}"
        else:
            armed = "none"
        dur = self.lims[1] - self.lims[0]
        n_ev = len(self.events)
        n_ev_active = sum(1 for en, _ in self.events if en == self.event_num)
        lines = [
            f"armed : {armed}",
            f"time  : {centre:7.1f}s",
            f"length: {dur:7.1f}s ({dur/60:.1f}m)",
            f"scored: {pct:5.1f}%",
            f"events: {n_ev}  (#{self.event_num}: {n_ev_active})",
            "",
            "per-state bins:",
        ]
        for s in range(1, 6):
            c = int(np.count_nonzero(self.states == s))
            lines.append(f"  {s} {STATE_NAMES[s][:6]:6} {c:6d}")
        self.info_text.set_text("\n".join(lines))

    def _toggle_help(self):
        self.help_visible = not self.help_visible
        self.help_overlay.set_visible(self.help_visible)
        self.fig.canvas.draw_idle()

    def _confirm_close(self):
        """Return True if the window may close, prompting to save when dirty.

        Called from ``_EditorWindow.closeEvent``. Cancel keeps the window open."""
        if not self.dirty or not _HAVE_QT or self.win is None:
            return True
        resp = QMessageBox.question(
            self.win, "Unsaved changes", "Save state scoring before closing?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel)
        if resp == QMessageBox.StandardButton.Cancel:
            return False
        if resp == QMessageBox.StandardButton.Yes:
            self.save_states()
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
        if hasattr(self, "info_text"):
            self._update_info()
            self.fig.canvas.draw_idle()

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
        if hasattr(self, "info_text"):
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

    def _move_cursor(self, direction):
        """Step the time cursor by one bin (← →), scrolling the view to follow."""
        self.cursor_time = float(np.clip(self.cursor_time + direction * self._dt,
                                         self.lims[0], self.lims[1]))
        lo, hi = self._xlim_get()
        w = hi - lo
        if self.cursor_time < lo:
            self._set_xlim(self.cursor_time, self.cursor_time + w)
        elif self.cursor_time > hi:
            self._set_xlim(self.cursor_time - w, self.cursor_time)
        else:
            self._update_eeg(self.cursor_time)
            if hasattr(self, "info_text"):
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
            self._move_cursor(1 if k == "right" else -1)
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
                self.rawwin_slider.set_val(np.clip(self.eeg_show,
                                                   self.rawwin_slider.valmin,
                                                   self.rawwin_slider.valmax))
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

        # a click always moves the time cursor to that spot
        self.cursor_time = float(event.xdata)
        if self.current_state is None:
            # browse: centre LFP view on the click
            self._update_eeg(self.cursor_time)
            if hasattr(self, "info_text"):
                self._update_info()
            self.fig.canvas.draw_idle()
            return

        # scoring: first click sets a bound, second applies the state
        if self.pending_bound is None:
            self.pending_bound = self.cursor_time
            for ax in self.ax_spec + [self.ax_motion, self.ax_state]:
                self.pending_line.append(ax.axvline(self.cursor_time, color="r", lw=1.0))
            self._update_eeg(self.cursor_time)
            self.fig.canvas.draw_idle()
        else:
            self._apply_state(self.pending_bound, self.cursor_time, self.current_state)
            self.pending_bound = None
            self._clear_pending_line()

    def _clear_pending_line(self):
        for ln in self.pending_line:
            ln.remove()
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
        if hasattr(self, "info_text"):
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
            a.remove()
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
        if hasattr(self, "info_text"):
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
        if path is None:
            path = os.path.join(self.out_folder, f"{self.base_name}-states.mat")
        if not os.path.isfile(path):
            print(f"No states file at {path}")
            return
        data = loadmat(path)
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

    def show(self):
        """Show the editor window and block until it is closed.

        Uses a nested Qt event loop so a caller (the setup GUI) can invoke this
        synchronously from within the already-running application loop, the same
        way ``QDialog.exec()`` blocks. No-op when built headlessly."""
        if self.win is None:
            return
        owns_app = False
        if QApplication.instance() is None:         # standalone use
            self._app = QApplication([])
            owns_app = True
        self.win.show()
        self.win.raise_()
        self.win.activateWindow()
        self.canvas.setFocus()
        if owns_app:
            self._app.exec()
        else:
            loop = QEventLoop()
            self.win._loop = loop
            loop.exec()
