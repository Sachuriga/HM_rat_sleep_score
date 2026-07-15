"""Setup GUI - Python port of ``Sleep_score_HM_neuron.m``.

Pick an LFP folder, enter three channel numbers, auto-detect the motion/EMG
file, choose an output folder and parameters, then launch the state editor.
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np

from processing import (compute_channel_spectrogram, process_motion,
                        detect_sampling_rate, cache_path, save_cache, load_cache,
                        find_lfp_source, load_lfp_channel, find_output)
from state_editor import StateEditor

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

# LFP for sleep scoring is ~250–2000 Hz. Anything above this is the raw
# acquisition rate leaking in via lfp_timestamps.npy, not a real LFP rate.
LFP_FS_MAX = 5000


class SetupGUI:
    def __init__(self):
        self.lfp_folder = ""
        self.emg_file = ""
        self.out_folder = ""
        self.lfp_source = None      # dict from find_lfp_source, set on folder select

        self.root = tk.Tk()
        self.root.title("Sleep Score Setup")
        self.root.geometry("790x560")
        self.root.minsize(680, 540)
        self.root.resizable(True, True)
        self._bring_to_front()
        self._build()

    def _bring_to_front(self):
        """Force the window (and its dialogs) to the foreground on macOS."""
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after_idle(self.root.attributes, "-topmost", False)
        self.root.focus_force()

    # ------------------------------------------------------------------ layout
    def _build(self):
        # Column 0 (entries) stretches; column 1 (Browse buttons) stays fixed.
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=0)

        def header(text, row):
            tk.Label(self.root, text=text, font=("Helvetica", 11, "bold"),
                     anchor="w").grid(row=row, column=0, columnspan=2,
                                      sticky="w", padx=12, pady=(8, 2))

        # LFP folder
        header("LFP Output Folder", 0)
        self.lfp_var = tk.StringVar()
        tk.Entry(self.root, textvariable=self.lfp_var, state="readonly").grid(
            row=1, column=0, sticky="we", padx=(12, 4))
        tk.Button(self.root, text="Browse...", command=self._sel_lfp).grid(
            row=1, column=1, padx=(0, 12), sticky="e")

        # Channels
        header("Channel Numbers (1-N)", 2)
        self.ch_vars = []
        chframe = tk.Frame(self.root)
        chframe.grid(row=3, column=0, columnspan=2, sticky="w", padx=12)
        for k in range(3):
            tk.Label(chframe, text=f"Ch {k + 1}:").grid(row=0, column=2 * k, padx=(0, 2))
            v = tk.StringVar(value=str(k + 1))
            self.ch_vars.append(v)
            tk.Entry(chframe, textvariable=v, width=6).grid(row=0, column=2 * k + 1, padx=(0, 14))

        # Motion / EMG
        header("Motion / EMG File", 4)
        self.emg_var = tk.StringVar()
        tk.Entry(self.root, textvariable=self.emg_var, state="readonly").grid(
            row=5, column=0, sticky="we", padx=(12, 4))
        tk.Button(self.root, text="Browse...", command=self._sel_emg).grid(
            row=5, column=1, padx=(0, 12), sticky="e")
        self.emg_auto = tk.Label(self.root, text="", fg="#007000", anchor="w")
        self.emg_auto.grid(row=6, column=0, columnspan=2, sticky="w", padx=12)

        # Output folder
        header("Output / Save Folder", 7)
        self.out_var = tk.StringVar()
        tk.Entry(self.root, textvariable=self.out_var, state="readonly").grid(
            row=8, column=0, sticky="we", padx=(12, 4))
        tk.Button(self.root, text="Browse...", command=self._sel_out).grid(
            row=8, column=1, padx=(0, 12), sticky="e")

        # Recording info (channels / duration, filled in after folder select)
        self.info_label = tk.Label(self.root, text="", fg="#444444", anchor="w")
        self.info_label.grid(row=9, column=0, columnspan=2, sticky="w", padx=12)

        # Parameters
        params = tk.Frame(self.root)
        params.grid(row=10, column=0, columnspan=2, sticky="w", padx=12, pady=6)
        tk.Label(params, text="Sampling Rate (Hz):").grid(row=0, column=0)
        self.fs_var = tk.StringVar(value="1500")
        tk.Entry(params, textvariable=self.fs_var, width=8).grid(row=0, column=1, padx=(2, 20))
        tk.Label(params, text="Session Name:").grid(row=0, column=2)
        self.name_var = tk.StringVar(value="HM_neurons")
        tk.Entry(params, textvariable=self.name_var, width=14).grid(row=0, column=3, padx=2)
        tk.Label(params, text="Motion type:").grid(row=1, column=0, pady=(6, 0), sticky="w")
        self.motion_mode_var = tk.StringVar(value="Accelerometer (case 3)")
        ttk.Combobox(params, textvariable=self.motion_mode_var, state="readonly",
                     values=list(MOTION_MODES.keys()), width=26).grid(
            row=1, column=1, columnspan=3, sticky="w", padx=(2, 0), pady=(6, 0))

        self.recompute_var = tk.BooleanVar(value=False)
        tk.Checkbutton(params, text="Ignore cache (recompute)",
                       variable=self.recompute_var).grid(
            row=2, column=0, columnspan=4, sticky="w", pady=(4, 0))

        # Buzsáki auto-scoring: shown as an extra panel in the editor. Loads a
        # saved buzsaki_states.npz if present, or computes one when ticked.
        self.buzsaki_var = tk.BooleanVar(value=True)
        tk.Checkbutton(params, text="Show Buzsáki auto-score (recompute on open)",
                       variable=self.buzsaki_var).grid(
            row=3, column=0, columnspan=4, sticky="w", pady=(2, 0))

        # Editable scoring thresholds — multipliers on the auto (bimodal) thresholds
        # (1.0 = auto; lower = more permissive), plus drowsy-band width & min epoch.
        import buzsaki_score as _bz
        thr = tk.Frame(params)
        thr.grid(row=4, column=0, columnspan=4, sticky="w", pady=(4, 0))
        tk.Label(thr, text="Thresholds:", font=("Helvetica", 9, "bold")).grid(
            row=0, column=0, sticky="w")
        self.swf_var = tk.StringVar(value="1.0")
        self.thf_var = tk.StringVar(value=str(_bz.TH_THRESH_FACTOR))
        self.emgf_var = tk.StringVar(value="1.0")
        self.drowsy_var = tk.StringVar(value=str(_bz.DROWSY_FRAC))
        self.minsec_var = tk.StringVar(value="10")
        fields = [("SW× (NREM)", self.swf_var), ("θ× (REM)", self.thf_var),
                  ("EMG× (wake)", self.emgf_var), ("drowsy", self.drowsy_var),
                  ("min ep (s)", self.minsec_var)]
        for i, (lbl, var) in enumerate(fields):
            tk.Label(thr, text=lbl).grid(row=0, column=1 + 2 * i, padx=(8, 1))
            tk.Entry(thr, textvariable=var, width=5).grid(row=0, column=2 + 2 * i)
        tk.Label(thr, text="1.0 = auto  ·  ↓SW = more sleep  ·  ↓θ = more REM  ·  "
                 "↓EMG = more wake", fg="#777777").grid(
            row=1, column=1, columnspan=10, sticky="w", pady=(1, 0))

        # Status + launch
        self.status = tk.Label(self.root, text="Ready. Select an LFP folder to begin.",
                               fg="#333333", anchor="w", wraplength=560, justify="left")
        self.status.grid(row=11, column=0, columnspan=2, sticky="w", padx=12, pady=(8, 4))

        tk.Button(self.root, text="Launch State Editor", font=("Helvetica", 12, "bold"),
                  bg="#2e8b2e", fg="white", command=self._launch).grid(
            row=12, column=0, columnspan=2, pady=10, ipadx=20, ipady=8)

    # ------------------------------------------------------------------ helpers
    def _set_status(self, msg, color="#333333"):
        self.status.config(text=msg, fg=color)
        self.root.update()

    def _show_recording_info(self, folder):
        """Display channel count + duration and auto-fill the sampling rate."""
        src = self.lfp_source
        if src is None:
            self.info_label.config(text="")
            return
        n_samples = src["n_samples"]
        chans = src["channels"]

        fs = detect_sampling_rate(find_output(folder, "lfp_timestamps.npy"))
        rate_note = ""
        # LFP for sleep scoring is ~250–2000 Hz. A detected rate this high means
        # lfp_timestamps.npy holds the RAW acquisition rate (e.g. 30 kHz), not the
        # LFP rate — auto-setting it would make each spectrogram bin ~fs samples
        # (20× too wide) and score only ~1/20 of the recording. Keep the LFP
        # default and warn instead of silently adopting the raw rate.
        if fs and fs > LFP_FS_MAX:
            rate_note = (f"  (⚠ lfp_timestamps.npy implies {int(fs)} Hz = the raw "
                         f"rate, not the LFP rate — keeping {self.fs_var.get()} Hz. "
                         f"Re-export the LFP or set the rate manually.)")
        elif fs:
            self.fs_var.set(str(int(fs)))
            rate_note = f"  (sampling rate auto-set to {int(fs)} Hz)"
        try:
            fs_val = float(self.fs_var.get())
        except ValueError:
            fs_val = fs or 1000.0
        dur = n_samples / fs_val if fs_val else 0
        layout = "lfp_data.npy" if src["kind"] == "matrix" else "channels_npy/"
        rng = f"{chans[0]}-{chans[-1]}" if chans else "none"
        self.info_label.config(
            text=f"{len(chans)} channels ({rng}) via {layout}, {n_samples:,} samples, "
                 f"{dur:.1f} s ({dur / 60:.1f} min){rate_note}")

    def _sel_lfp(self):
        folder = filedialog.askdirectory(title="Select LFP Output Folder",
                                         parent=self.root)
        if not folder:
            return
        self.lfp_folder = folder
        self.lfp_var.set(folder)
        self.lfp_source = find_lfp_source(folder)
        if self.lfp_source is None:
            self._set_status("Warning: no lfp_data.npy or channels_npy/ found here.",
                             "#cc6600")
            self.info_label.config(text="")
        else:
            self._set_status("Folder loaded. Enter channels, then Launch.", "#000000")
            self._show_recording_info(folder)
            # Auto-detect the rat_sessiondate_ prefix from the folder's files and
            # use it as the session name, so saved files share the session naming.
            from processing import output_prefix
            pfx = output_prefix(folder)
            if pfx:
                self.name_var.set(pfx.rstrip("_"))
                self._set_status(f"Folder loaded. Session: {pfx.rstrip('_')}. "
                                 f"Enter channels, then Launch.", "#000000")

        self.emg_file = ""
        for cand in EMG_CANDIDATES:
            p = find_output(folder, cand)          # prefixed (rat_sessiondate_) or not
            if p is not None:
                self.emg_file = str(p)
                self.emg_var.set(str(p))
                self.emg_auto.config(text=f"Auto-detected: {os.path.basename(p)}",
                                     fg="#007000")
                break
        if not self.emg_file:
            self.emg_auto.config(text="No EMG file auto-detected - browse manually.",
                                 fg="#cc6600")
        if not self.out_folder:
            self.out_folder = folder
            self.out_var.set(folder)

    def _sel_emg(self):
        start = self.lfp_folder or os.getcwd()
        f = filedialog.askopenfilename(title="Select Motion / EMG File",
                                       initialdir=start, parent=self.root,
                                       filetypes=[("NumPy", "*.npy")])
        if not f:
            return
        self.emg_file = f
        self.emg_var.set(f)
        self.emg_auto.config(text="Motion file set manually.", fg="#000099")

    def _sel_out(self):
        folder = filedialog.askdirectory(title="Select Output / Save Folder",
                                         parent=self.root)
        if folder:
            self.out_folder = folder
            self.out_var.set(folder)

    # ------------------------------------------------------------------ launch
    def _launch(self):
        if not self.lfp_folder:
            return self._set_status("Error: select an LFP folder.", "#cc0000")
        if not self.emg_file:
            return self._set_status("Error: select a motion/EMG file.", "#cc0000")
        if not self.out_folder:
            return self._set_status("Error: select an output folder.", "#cc0000")
        try:
            eeg_fs = float(self.fs_var.get())
            assert eeg_fs > 0
        except (ValueError, AssertionError):
            return self._set_status("Error: sampling rate must be positive.", "#cc0000")

        chs = []
        for k, v in enumerate(self.ch_vars):
            try:
                val = int(v.get().strip())
                assert val >= 1
            except (ValueError, AssertionError):
                return self._set_status(f"Error: Ch {k + 1} must be a whole number >= 1.",
                                        "#cc0000")
            chs.append(val)
        if len(set(chs)) < 3:
            return self._set_status("Error: all 3 channels must differ.", "#cc0000")

        base = self.name_var.get().strip() or "session"
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
            self._set_status(f"Error: {exc}", "#cc0000")
            raise

    def _run(self, chs, eeg_fs, base):
        source = self.lfp_source or find_lfp_source(self.lfp_folder)
        if source is None:
            return self._set_status(
                f"Error: no lfp_data.npy or channels_npy/ found in {self.lfp_folder}",
                "#cc0000")

        cpath = cache_path(self.out_folder, base)
        cached = None
        if not self.recompute_var.get() and os.path.isfile(cpath):
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
                        f"(have {source['channels'][0]}-{source['channels'][-1]}).",
                        "#cc0000")

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

            mode = MOTION_MODES.get(self.motion_mode_var.get(), "accelerometer")
            self._set_status(f"Loading + processing motion ({mode}) ...", "#0000aa")
            motion_raw = np.load(self.emg_file, mmap_mode="r")
            motion = process_motion(motion_raw, raw_eeg[0].size, eeg_fs, mode=mode)
            if motion.size != to.size:
                motion = np.interp(np.linspace(0, 1, to.size),
                                   np.linspace(0, 1, motion.size), motion.ravel())

            self._set_status("Caching spectrograms for fast reload ...", "#0000aa")
            try:
                save_cache(cpath, chs, eeg_fs, specs, fos, to, raw_eeg, motion)
            except Exception as exc:
                print(f"Warning: could not write cache: {exc}")

        auto_states, auto_ts = self._buzsaki_labels(chs)
        overlays = self._load_overlays()

        self._set_status("Launching state editor ...", "#007000")
        editor = StateEditor(base, specs, fos, to, motion, raw_eeg, eeg_fs,
                             out_folder=self.out_folder, chs=chs,
                             auto_states=auto_states, auto_states_ts=auto_ts,
                             overlays=overlays)
        self.root.withdraw()
        editor.show()
        self.root.deiconify()
        self._set_status("State editor closed. Results saved to output folder.", "#006600")

    def _buzsaki_labels(self, chs):
        """Return (states, timestamps) Buzsáki auto-labels to show, or (None, None).

        Recomputes fresh from the LFP folder every time (so scorer changes always
        take effect) and overwrites buzsaki_states.npz. Only falls back to a saved
        npz if the recompute fails.
        """
        if not self.buzsaki_var.get():
            return None, None
        import buzsaki_score as bz

        def _f(var, default):
            try:
                return float(var.get())
            except (ValueError, tk.TclError):
                return default
        kw = dict(sw_factor=_f(self.swf_var, 1.0),
                  th_factor=_f(self.thf_var, bz.TH_THRESH_FACTOR),
                  emg_factor=_f(self.emgf_var, 1.0),
                  drowsy_frac=_f(self.drowsy_var, bz.DROWSY_FRAC),
                  min_secs=_f(self.minsec_var, 10.0))
        try:
            self._set_status("Computing Buzsáki auto-score ...", "#0000aa")
            res, ch = bz.score_from_lfp_output(self.lfp_folder, channel=chs[0], **kw)
            from processing import output_prefix
            pfx = output_prefix(self.lfp_folder)
            bz.save(res, os.path.join(self.lfp_folder, f"{pfx}{bz.DEFAULT_OUT}"))
            self._set_status("Buzsáki auto-score recomputed.", "#006600")
            return res["states"], res["timestamps"]
        except Exception as exc:
            for folder in (self.lfp_folder, self.out_folder):
                f = find_output(folder, bz.DEFAULT_OUT)   # prefixed or not
                if f is not None:
                    self._set_status(f"Compute failed; loaded saved labels ({exc})",
                                     "#cc6600")
                    return bz.load_states(f)
            self._set_status(f"Buzsáki auto-score skipped: {exc}", "#cc6600")
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
        # EMG-from-LFP (5 Hz, has its own timestamps)
        ef = find_output(self.lfp_folder, "emg_from_lfp_5hz.npy") or \
            find_output(self.lfp_folder, "emg_from_lfp.npy")
        if ef is not None:
            try:
                emg = np.load(ef).ravel()
                tsf = find_output(self.lfp_folder, "emg_from_lfp_timestamps.npy")
                ets = np.load(tsf).ravel() if tsf is not None else None
                overlays.append(("EMG", emg, ets))
            except Exception as exc:
                print(f"Warning: could not load EMG overlay: {exc}")
        # theta/delta ratio (full LFP rate) — log-scaled (heavily skewed),
        # decimated, and smoothed (~15 s) so the NREM/REM trend is legible.
        tf = find_output(self.lfp_folder, "theta_delta_ratio.npy")
        if tf is not None:
            try:
                td = np.asarray(np.load(tf, mmap_mode="r")[::100], dtype=float)
                td = np.log10(np.clip(td, 1e-6, None))
                from scipy.ndimage import uniform_filter1d
                td = uniform_filter1d(td, size=225, mode="nearest")  # ~15 s @ 15 Hz
                overlays.append(("theta/delta", td, None))
            except Exception as exc:
                print(f"Warning: could not load theta/delta overlay: {exc}")
        return overlays

    def run(self):
        self.root.mainloop()


def main():
    SetupGUI().run()


if __name__ == "__main__":
    main()
