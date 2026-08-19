# Rat Sleep Scoring Toolkit (Python)

A point-and-click PyQt6 toolkit for manual sleep-state scoring of rodent LFP
recordings. Load LFP saved as NumPy `.npy`, pick channels and a motion/EMG
signal, and score sleep states in an interactive spectrogram editor.

Sleep states:

| Code | State |
|------|-------|
| 0 | No state |
| 1 | Awake |
| 2 | Light / Drowsy |
| 3 | NREM |
| 4 | Intermediate |
| 5 | REM |

The state editor is a Python reimplementation of `TheStateEditor` (originally by
Dr. Andres Grosmark and Dr. Abdel Rayan), modified and generalised by Sachuriga.

## Install

Two one-time steps per machine.

**1. Install the package** into a conda environment (any name — it differs per
PC and that's fine). Run this **from the project root folder** (the folder
containing `pyproject.toml`) — the `.` means "this folder":

```bash
conda activate <your-env>
cd <where-you-cloned>/HM_rat_sleep_score
pip install -e .
```

Because it's an editable install, edits to the code under `python/` take
effect immediately (no reinstall needed). If you later **move the repo
folder**, rerun `pip install -e .` from the new location.

**2. Put the launcher on your PATH** so `sleepscore` works from any folder
*without* activating conda first.

*macOS / Linux* — add to `~/.zshrc` (or `~/.bashrc`), using your machine's
actual repo path:

```bash
export PATH="$PATH:$HOME/Desktop/code/HM_rat_sleep_score/bin"
```

*Windows* — add the repo's `bin` folder to your user Path: press Win, search
"environment variables" → *Edit environment variables for your account* →
select **Path** → *Edit* → *New* → add e.g.
`C:\Users\<you>\Desktop\code\HM_rat_sleep_score\bin`, then open a new
terminal. (`bin\sleepscore.bat` is the Windows launcher.)

The launcher auto-discovers whichever conda env has the package installed
(checking, in order: `SLEEPSCORE_CONDA_ENV` if set, the active env, then all
envs of every conda/mamba install) and runs the GUI with that env's Python —
so nothing is hardcoded to one machine's env name, and stale installs left
behind after moving the repo are skipped. `sleepscore --which` prints the env
it would use.

**Requirements:** Python ≥ 3.9, `PyQt6`, `numpy`, `scipy`, `matplotlib`
(installed automatically by `pip install -e .`).

## Usage

From any folder:

```bash
sleepscore
```

The setup GUI opens. Steps:

1. **LFP output folder** — Browse to the folder holding your LFP data. It reads
   either a `channels_npy/` subfolder of per-channel files (`lfp_ntXX_ch01.npy`)
   or a single `lfp_data.npy` matrix. Channel count, duration and sampling rate
   are shown, and the session name is auto-filled from the folder's prefix.
2. **Channels** — enter three distinct 1-based channel numbers to score.
3. **Motion / EMG file** — auto-detected from the LFP folder (in priority order
   `motion.npy`, `emg_rms.npy`, `emg_data.npy`, `theta_delta_ratio.npy`), or
   Browse for any `.npy`.
4. **Output / save folder** — where results and the spectrogram cache are
   written. Defaults to the LFP folder.
5. **Resume from previous scoring** (optional) — auto-detects a saved
   `*-states.npz` / `*-states.mat` so you can continue an earlier session.
6. **Parameters** — sampling rate, session name, motion type, and optional
   Buzsáki auto-scoring with adjustable thresholds.

Click **Launch State Editor**. Press `h` in the editor for the full list of
keyboard/mouse controls.

### Scoring in the editor

- **Arm a state** with the coloured toolbar buttons (or keys `1`–`5`, `0` to
  erase); click again / press `c` to un-arm.
- **Score an epoch**: with a state armed, click two time points (or press
  `Space` twice) to assign it to that span (minimum 10 s).
- **Navigate**: `← →` move the time cursor (hold to accelerate), `Shift+← →`
  pan, scroll to zoom, `Home`/`End` jump to the ends, `r` resets the view.
- **Save / load**: the toolbar has **Save .npy** (NumPy `.npz`), **Save .mat**
  (`s`), and **Load** (reads either format, `l`). `u` undoes the last change.

## Output

Saved scoring contains:

| Field | Description |
|-------|-------------|
| `states` | Length-N vector (1 s bins), values 0–5 per bin |
| `events` | N×2 array of event numbers and timestamps (s) |
| `transitions` | N×3 array `[state, start_s, end_s]` |
| `timestamps` | (`.npz` only) per-bin time in seconds |

`.npz` is the native NumPy format; `.mat` is written via SciPy for
compatibility with other tools. A `<SessionName>.eegstates` cache of the
whitened spectrograms is created on first run to speed up subsequent loads.

## Repository layout

```text
python/
├── sleepscore.py       # entry point (the `sleepscore` command)
├── setup_gui.py        # setup GUI (folder/channel/parameter selection)
├── state_editor.py     # interactive spectrogram + scoring editor
├── processing.py       # LFP preprocessing, spectrograms, motion processing
├── buzsaki_score.py    # optional Buzsáki auto-scoring
└── test_pipeline.py    # headless smoke test
LFP_Output/             # example data folder
```

## License

See [LICENSE](LICENSE).
