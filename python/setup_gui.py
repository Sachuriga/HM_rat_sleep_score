"""Setup GUI - Python port of ``Sleep_score_HM_neuron.m``.

Pick an LFP folder, enter three channel numbers, auto-detect the motion/EMG
file, choose an output folder and parameters, then launch the state editor.
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np

from processing import compute_channel_spectrogram, downsample_motion
from state_editor import StateEditor

EMG_CANDIDATES = ["emg_rms.npy", "emg_data.npy",
                  "theta_delta_ratio.npy", "awakeness.npy"]


class SetupGUI:
    def __init__(self):
        self.lfp_folder = ""
        self.emg_file = ""
        self.out_folder = ""

        self.root = tk.Tk()
        self.root.title("Sleep Score Setup")
        self.root.geometry("620x470")
        self.root.minsize(560, 470)
        self.root.resizable(True, False)
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

        # Parameters
        params = tk.Frame(self.root)
        params.grid(row=9, column=0, columnspan=2, sticky="w", padx=12, pady=6)
        tk.Label(params, text="Sampling Rate (Hz):").grid(row=0, column=0)
        self.fs_var = tk.StringVar(value="1000")
        tk.Entry(params, textvariable=self.fs_var, width=8).grid(row=0, column=1, padx=(2, 20))
        tk.Label(params, text="Session Name:").grid(row=0, column=2)
        self.name_var = tk.StringVar(value="HM_neurons")
        tk.Entry(params, textvariable=self.name_var, width=14).grid(row=0, column=3, padx=2)

        # Status + launch
        self.status = tk.Label(self.root, text="Ready. Select an LFP folder to begin.",
                               fg="#333333", anchor="w", wraplength=560, justify="left")
        self.status.grid(row=10, column=0, columnspan=2, sticky="w", padx=12, pady=(8, 4))

        tk.Button(self.root, text="Launch State Editor", font=("Helvetica", 12, "bold"),
                  bg="#2e8b2e", fg="white", command=self._launch).grid(
            row=11, column=0, columnspan=2, pady=10, ipadx=20, ipady=8)

    # ------------------------------------------------------------------ helpers
    def _set_status(self, msg, color="#333333"):
        self.status.config(text=msg, fg=color)
        self.root.update()

    def _sel_lfp(self):
        folder = filedialog.askdirectory(title="Select LFP Output Folder",
                                         parent=self.root)
        if not folder:
            return
        self.lfp_folder = folder
        self.lfp_var.set(folder)
        if not os.path.isfile(os.path.join(folder, "lfp_data.npy")):
            self._set_status("Warning: lfp_data.npy not found in this folder.", "#cc6600")
        else:
            self._set_status("Folder loaded. Enter channels, then Launch.", "#000000")

        self.emg_file = ""
        for cand in EMG_CANDIDATES:
            p = os.path.join(folder, cand)
            if os.path.isfile(p):
                self.emg_file = p
                self.emg_var.set(p)
                self.emg_auto.config(text=f"Auto-detected: {cand}", fg="#007000")
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
        try:
            self._run(chs, eeg_fs, base)
        except Exception as exc:  # surface any load/compute failure in the GUI
            self._set_status(f"Error: {exc}", "#cc0000")
            raise

    def _run(self, chs, eeg_fs, base):
        lfp_file = os.path.join(self.lfp_folder, "lfp_data.npy")
        if not os.path.isfile(lfp_file):
            return self._set_status(f"Error: lfp_data.npy not found in {self.lfp_folder}",
                                    "#cc0000")

        self._set_status("Loading lfp_data.npy ...", "#0000aa")
        eeg = np.load(lfp_file, mmap_mode="r")          # [samples, channels]
        n_ch = eeg.shape[1]
        for c in chs:
            if c > n_ch:
                return self._set_status(
                    f"Error: channel {c} does not exist (file has {n_ch}).", "#cc0000")

        self._set_status("Preprocessing + computing spectrograms ...", "#0000aa")
        specs, fos, raw_eeg = [], [], []
        to = None
        for c in chs:
            sig = np.asarray(eeg[:, c - 1], dtype=np.float64)
            spec, fo, to, cleaned = compute_channel_spectrogram(sig, eeg_fs)
            specs.append(spec)
            fos.append(fo)
            raw_eeg.append(cleaned)

        self._set_status("Loading motion file ...", "#0000aa")
        motion_raw = np.load(self.emg_file)
        motion = downsample_motion(motion_raw, raw_eeg[0].size, eeg_fs)
        if motion.size != to.size:
            motion = np.interp(np.linspace(0, 1, to.size),
                               np.linspace(0, 1, motion.size), motion.ravel())

        self._set_status("Launching state editor ...", "#007000")
        editor = StateEditor(base, specs, fos, to, motion, raw_eeg, eeg_fs,
                             out_folder=self.out_folder)
        self.root.withdraw()
        editor.show()
        self.root.deiconify()
        self._set_status("State editor closed. Results saved to output folder.", "#006600")

    def run(self):
        self.root.mainloop()


def main():
    SetupGUI().run()


if __name__ == "__main__":
    main()
