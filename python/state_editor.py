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

# Use an interactive Tk backend by default, but honour an explicit MPLBACKEND
# (e.g. "Agg" for headless testing).
if "MPLBACKEND" not in os.environ:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.io import loadmat, savemat
from scipy.signal.windows import hann

from processing import matlab_round

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
EEG_STEPS = [0.25, 0.5, 1, 2, 5, 15, 30, 60]   # '-'/'=' LFP width steps

HELP_LINES = [
    ("1-5", "arm a state, then click its two time bounds"),
    ("0", "arm 'no state' (erase) the same way"),
    ("c", "cancel the armed state"),
    ("click", "(no state armed) centre the LFP view here"),
    ("← →", "pan left / right"),
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


class StateEditor:
    def __init__(self, base_name, specs, fos, to, motion, raw_eeg, eeg_fs,
                 out_folder=".", states=None, chs=None):
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
        valid = ~np.isnan(m)
        if valid.any():
            m[valid] = (m[valid] - m[valid].mean()) / (m[valid].std() + 1e-12)
        self.motion = m

        # --- scoring interaction state --------------------------------------
        self.current_state = None
        self.pending_bound = None
        self.pending_line = []
        self.dirty = False          # unsaved changes?
        self.help_visible = False

        # --- event marking state --------------------------------------------
        self.events = []            # list of [event_num, time_s]
        self.event_num = 1          # currently active event number (1-10)
        self.event_mode = None      # None | 'add' | 'delete'
        self.event_artists = []     # drawn vertical lines + labels

        self._build_figure()
        self._build_side_panel()
        self._connect()
        self._install_close_handler()

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

    def _build_figure(self):
        self.fig = plt.figure(figsize=(15, 9))
        self.fig.canvas.manager.set_window_title(f"States: {self.base_name}")

        n = self.n_ch
        left, width = 0.05, 0.80
        # vertical layout: state bar / spectrograms / motion / LFP traces
        self.ax_state = self.fig.add_axes([left, 0.945, width, 0.045])
        spec_top, spec_h = 0.93, (0.93 - 0.34) / n
        self.ax_spec, self.spec_imgs = [], []
        for i in range(n):
            y = spec_top - (i + 1) * spec_h
            self.ax_spec.append(self.fig.add_axes([left, y, width, spec_h - 0.005]))
        self.ax_motion = self.fig.add_axes([left, 0.235, width, 0.07])
        eeg_h = 0.05
        self.ax_eeg = [self.fig.add_axes([left, 0.18 - i * (eeg_h + 0.005), width, eeg_h])
                       for i in range(n)]

        self._draw_state_bar()

        cmap = "jet"
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

        self.ax_motion.plot(self.to, self.motion, "-k", lw=0.6)
        valid = ~np.isnan(self.motion)
        if valid.any():
            self.ax_motion.set_ylim(np.percentile(self.motion[valid], 1),
                                    np.percentile(self.motion[valid], 99))
        self.ax_motion.set_ylabel("Motion (z)")
        self.ax_motion.set_xlim(self.lims)
        self.ax_motion.set_xticklabels([])
        self.cursor_lines.append(self.ax_motion.axvline(mid, color="k", ls="--", lw=0.8))

        self.eeg_lines = []
        for i, ax in enumerate(self.ax_eeg):
            (ln,) = ax.plot([], [], color="y", lw=0.5)
            self.eeg_lines.append(ln)
            ax.set_facecolor("black")
            ax.set_ylabel(f"Ch {self.chs[i]}")
            yabs = np.percentile(np.abs(self.eeg[i]), 99.5) or 1.0
            ax.set_ylim(-yabs, yabs)
            ax.yaxis.set_major_locator(MaxNLocator(3))
            if i != n - 1:
                ax.set_xticklabels([])
        self.ax_eeg[-1].set_xlabel("Time (s)")

        self._update_eeg(mid)
        self._set_title()

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
            ax.add_patch(plt.Rectangle((0.0, y - 0.035), 0.16, 0.045,
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

    def _install_close_handler(self):
        """Warn about unsaved work when closing (TkAgg only; no-op elsewhere)."""
        try:
            win = self.fig.canvas.manager.window
            win.protocol("WM_DELETE_WINDOW", self._on_close_request)
        except Exception:
            pass

    def _on_close_request(self):
        if self.dirty:
            try:
                from tkinter import messagebox
                resp = messagebox.askyesnocancel(
                    "Unsaved changes",
                    "Save state scoring before closing?")
                if resp is None:           # cancel -> stay open
                    return
                if resp:                   # yes -> save then close
                    self.save_states()
            except Exception:
                pass
        plt.close(self.fig)

    def _mark_dirty(self):
        self.dirty = True
        self._set_title()

    def _draw_state_bar(self):
        rgb = np.ones((5, self.n_bins, 3))
        for s in range(1, 6):
            idx = self.states == s
            if idx.any():
                rgb[:, idx, :] = STATE_COLORS[s]
        self.ax_state.clear()
        self.ax_state.imshow(rgb, origin="lower", aspect="auto",
                             extent=[self.to[0], self.to[-1], 0.5, 5.5])
        self.ax_state.set_yticks([1, 2, 3, 4, 5])
        self.ax_state.set_xticks([])
        self.ax_state.set_ylabel("State")
        self.ax_state.set_xlim(getattr(self, "_xlim", self.lims))

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
        self.fig.canvas.manager.set_window_title(f"States: {self.base_name}{star}{extra}")
        if hasattr(self, "info_text"):
            self._update_info()
            self.fig.canvas.draw_idle()

    def _xlim_get(self):
        return self.ax_spec[0].get_xlim()

    def _set_xlim(self, lo, hi):
        lo = max(lo, self.lims[0])
        hi = min(hi, self.lims[1])
        self._xlim = (lo, hi)
        for ax in self.ax_spec:
            ax.set_xlim(lo, hi)
        self.ax_motion.set_xlim(lo, hi)
        self.ax_state.set_xlim(lo, hi)
        self._update_eeg((lo + hi) / 2)
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
        for ln in self.cursor_lines:
            ln.set_xdata([centre, centre])

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
            lo, hi = self._xlim_get()
            step = PAN_FRAC * (hi - lo) * (1 if k == "right" else -1)
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

        if self.current_state is None:
            # browse: centre LFP view on the click
            self._update_eeg(event.xdata)
            self.fig.canvas.draw_idle()
            return

        # scoring: first click sets a bound, second applies the state
        if self.pending_bound is None:
            self.pending_bound = event.xdata
            for ax in self.ax_spec + [self.ax_motion, self.ax_state]:
                self.pending_line.append(ax.axvline(event.xdata, color="r", lw=1.0))
            self.fig.canvas.draw_idle()
        else:
            t0, t1 = sorted((self.pending_bound, event.xdata))
            self._apply_state(t0, t1, self.current_state)
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
        plt.show()
