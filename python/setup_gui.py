"""Setup GUI - Python port of ``Sleep_score_HM_neuron.m`` (PyQt6).

Pick an LFP folder, enter three channel numbers, auto-detect the motion/EMG
file, choose an output folder and parameters, then launch the state editor.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox, QGroupBox,
    QFileDialog, QSizePolicy,
)

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


class SetupGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.lfp_folder = ""
        self.emg_file = ""
        self.out_folder = ""
        self.lfp_source = None      # dict from find_lfp_source, set on folder select

        self.setWindowTitle("Sleep Score Setup")
        self.resize(820, 620)
        self.setMinimumSize(700, 560)
        self._build()

    # ------------------------------------------------------------------ layout
    def _build(self):
        root = QWidget()
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(6)

        def header(text):
            lbl = QLabel(text)
            lbl.setStyleSheet("font-weight:700;")
            v.addWidget(lbl)

        def path_row(edit, browse_cb):
            row = QHBoxLayout()
            edit.setReadOnly(True)
            row.addWidget(edit, 1)
            btn = QPushButton("Browse…")
            btn.clicked.connect(browse_cb)
            row.addWidget(btn)
            v.addLayout(row)

        # LFP folder
        header("LFP Output Folder")
        self.lfp_edit = QLineEdit()
        path_row(self.lfp_edit, self._sel_lfp)

        # Channels
        header("Channel Numbers (1-N)")
        chrow = QHBoxLayout()
        self.ch_edits = []
        for k in range(3):
            chrow.addWidget(QLabel(f"Ch {k + 1}:"))
            e = QLineEdit(str(k + 1))
            e.setFixedWidth(56)
            self.ch_edits.append(e)
            chrow.addWidget(e)
            chrow.addSpacing(12)
        chrow.addStretch(1)
        v.addLayout(chrow)

        # Motion / EMG
        header("Motion / EMG File")
        self.emg_edit = QLineEdit()
        path_row(self.emg_edit, self._sel_emg)
        self.emg_auto = QLabel("")
        v.addWidget(self.emg_auto)

        # Output folder
        header("Output / Save Folder")
        self.out_edit = QLineEdit()
        path_row(self.out_edit, self._sel_out)

        # Recording info (channels / duration, filled in after folder select)
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color:#444444;")
        self.info_label.setWordWrap(True)
        v.addWidget(self.info_label)

        # Parameters
        params = QGroupBox("Parameters")
        pg = QGridLayout(params)
        pg.addWidget(QLabel("Sampling Rate (Hz):"), 0, 0)
        self.fs_edit = QLineEdit("1500")
        self.fs_edit.setFixedWidth(80)
        pg.addWidget(self.fs_edit, 0, 1)
        pg.addWidget(QLabel("Session Name:"), 0, 2)
        self.name_edit = QLineEdit("HM_neurons")
        pg.addWidget(self.name_edit, 0, 3)
        pg.addWidget(QLabel("Motion type:"), 1, 0)
        self.motion_combo = QComboBox()
        self.motion_combo.addItems(list(MOTION_MODES.keys()))
        pg.addWidget(self.motion_combo, 1, 1, 1, 3)

        self.recompute_chk = QCheckBox("Ignore cache (recompute)")
        pg.addWidget(self.recompute_chk, 2, 0, 1, 4)

        # Buzsáki auto-scoring: shown as an extra panel in the editor. Loads a
        # saved buzsaki_states.npz if present, or computes one when ticked.
        self.buzsaki_chk = QCheckBox("Show Buzsáki auto-score (recompute on open)")
        self.buzsaki_chk.setChecked(True)
        pg.addWidget(self.buzsaki_chk, 3, 0, 1, 4)

        # Editable scoring thresholds — multipliers on the auto (bimodal) thresholds
        # (1.0 = auto; lower = more permissive), plus drowsy-band width & min epoch.
        import buzsaki_score as _bz
        thr = QHBoxLayout()
        lab = QLabel("Thresholds:")
        lab.setStyleSheet("font-weight:700;")
        thr.addWidget(lab)
        self.swf_edit = QLineEdit("1.0")
        self.thf_edit = QLineEdit(str(_bz.TH_THRESH_FACTOR))
        self.emgf_edit = QLineEdit("1.0")
        self.drowsy_edit = QLineEdit(str(_bz.DROWSY_FRAC))
        self.minsec_edit = QLineEdit("10")
        for lbl, edit in [("SW× (NREM)", self.swf_edit), ("θ× (REM)", self.thf_edit),
                          ("EMG× (wake)", self.emgf_edit), ("drowsy", self.drowsy_edit),
                          ("min ep (s)", self.minsec_edit)]:
            thr.addSpacing(8)
            thr.addWidget(QLabel(lbl))
            edit.setFixedWidth(48)
            thr.addWidget(edit)
        thr.addStretch(1)
        pg.addLayout(thr, 4, 0, 1, 4)
        hint = QLabel("1.0 = auto  ·  ↓SW = more sleep  ·  ↓θ = more REM  ·  ↓EMG = more wake")
        hint.setStyleSheet("color:#777777;")
        pg.addWidget(hint, 5, 0, 1, 4)
        v.addWidget(params)

        # Status + launch
        self.status = QLabel("Ready. Select an LFP folder to begin.")
        self.status.setStyleSheet("color:#333333;")
        self.status.setWordWrap(True)
        v.addWidget(self.status)

        v.addStretch(1)
        launch = QPushButton("Launch State Editor")
        launch.setStyleSheet(
            "QPushButton{background:#2e8b2e; color:white; font-size:14px; "
            "font-weight:700; padding:10px;} QPushButton:hover{background:#37a337;}")
        launch.clicked.connect(self._launch)
        launch.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        v.addWidget(launch)

    # ------------------------------------------------------------------ helpers
    def _set_status(self, msg, color="#333333"):
        self.status.setText(msg)
        self.status.setStyleSheet(f"color:{color};")
        QApplication.processEvents()

    def _show_recording_info(self, folder):
        """Display channel count + duration and auto-fill the sampling rate."""
        src = self.lfp_source
        if src is None:
            self.info_label.setText("")
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

    def _sel_lfp(self):
        folder = QFileDialog.getExistingDirectory(self, "Select LFP Output Folder")
        if not folder:
            return
        self.lfp_folder = folder
        self.lfp_edit.setText(folder)
        self.lfp_source = find_lfp_source(folder)
        if self.lfp_source is None:
            self._set_status("Warning: no lfp_data.npy or channels_npy/ found here.",
                             "#cc6600")
            self.info_label.setText("")
        else:
            self._set_status("Folder loaded. Enter channels, then Launch.", "#000000")
            self._show_recording_info(folder)
            # Auto-detect the rat_sessiondate_ prefix from the folder's files and
            # use it as the session name, so saved files share the session naming.
            from processing import output_prefix
            pfx = output_prefix(folder)
            if pfx:
                self.name_edit.setText(pfx.rstrip("_"))
                self._set_status(f"Folder loaded. Session: {pfx.rstrip('_')}. "
                                 f"Enter channels, then Launch.", "#000000")

        self.emg_file = ""
        for cand in EMG_CANDIDATES:
            p = find_output(folder, cand)          # prefixed (rat_sessiondate_) or not
            if p is not None:
                self.emg_file = str(p)
                self.emg_edit.setText(str(p))
                self.emg_auto.setText(f"Auto-detected: {os.path.basename(p)}")
                self.emg_auto.setStyleSheet("color:#007000;")
                break
        if not self.emg_file:
            self.emg_auto.setText("No EMG file auto-detected - browse manually.")
            self.emg_auto.setStyleSheet("color:#cc6600;")
        if not self.out_folder:
            self.out_folder = folder
            self.out_edit.setText(folder)

    def _sel_emg(self):
        start = self.lfp_folder or os.getcwd()
        f, _ = QFileDialog.getOpenFileName(self, "Select Motion / EMG File", start,
                                           "NumPy (*.npy)")
        if not f:
            return
        self.emg_file = f
        self.emg_edit.setText(f)
        self.emg_auto.setText("Motion file set manually.")
        self.emg_auto.setStyleSheet("color:#000099;")

    def _sel_out(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output / Save Folder")
        if folder:
            self.out_folder = folder
            self.out_edit.setText(folder)

    # ------------------------------------------------------------------ launch
    def _launch(self):
        if not self.lfp_folder:
            return self._set_status("Error: select an LFP folder.", "#cc0000")
        if not self.emg_file:
            return self._set_status("Error: select a motion/EMG file.", "#cc0000")
        if not self.out_folder:
            return self._set_status("Error: select an output folder.", "#cc0000")
        try:
            eeg_fs = float(self.fs_edit.text())
            assert eeg_fs > 0
        except (ValueError, AssertionError):
            return self._set_status("Error: sampling rate must be positive.", "#cc0000")

        chs = []
        for k, e in enumerate(self.ch_edits):
            try:
                val = int(e.text().strip())
                assert val >= 1
            except (ValueError, AssertionError):
                return self._set_status(f"Error: Ch {k + 1} must be a whole number >= 1.",
                                        "#cc0000")
            chs.append(val)
        if len(set(chs)) < 3:
            return self._set_status("Error: all 3 channels must differ.", "#cc0000")

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

            mode = MOTION_MODES.get(self.motion_combo.currentText(), "accelerometer")
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
        self.hide()
        editor.show()
        self.show()
        self._set_status("State editor closed. Results saved to output folder.", "#006600")

    def _buzsaki_labels(self, chs):
        """Return (states, timestamps) Buzsáki auto-labels to show, or (None, None).

        Recomputes fresh from the LFP folder every time (so scorer changes always
        take effect) and overwrites buzsaki_states.npz. Only falls back to a saved
        npz if the recompute fails.
        """
        if not self.buzsaki_chk.isChecked():
            return None, None
        import buzsaki_score as bz

        def _f(edit, default):
            try:
                return float(edit.text())
            except ValueError:
                return default
        kw = dict(sw_factor=_f(self.swf_edit, 1.0),
                  th_factor=_f(self.thf_edit, bz.TH_THRESH_FACTOR),
                  emg_factor=_f(self.emgf_edit, 1.0),
                  drowsy_frac=_f(self.drowsy_edit, bz.DROWSY_FRAC),
                  min_secs=_f(self.minsec_edit, 10.0))
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


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    gui = SetupGUI()
    gui.show()
    gui.raise_()
    gui.activateWindow()
    app.exec()


if __name__ == "__main__":
    main()
