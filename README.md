# Rat Sleep Scoring Toolkit (Python)

A point-and-click PyQt6 toolkit for manual sleep-state scoring of rodent LFP
recordings. Load LFP saved as NumPy `.npy`, pick channels and a motion/EMG
signal, and score sleep states in an interactive spectrogram editor.

Sleep states:

| Code | State | Key |
|------|-------|-----|
| 0 | No state | `0` |
| 1 | Awake | `1` |
| 3 | NREM | `2` |
| 5 | REM | `3` |
| 4 | Intermediate | `4` |

The three states of Watson et al. (2016, *Neuron*) plus **intermediate** sleep,
the NREM→REM transition. Keys are sequential (`1`–`4`) while the stored codes
stay 1/3/4/5, so files remain MATLAB-compatible. The legacy code 2
(light/drowsy) is not scored — that paper folds drowsy periods and
microarousals into WAKE — and old files containing it are mapped on load
(2 → awake).

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
5. **Resume from previous scoring** (optional) — empty by default, which
   starts a fresh scoring. Browse for a saved `*-states.npz` / `*-states.mat`
   (or a `results/` file) to continue it: its labels are loaded, its scorer
   name is kept — no "Labeled by" prompt — and saving updates that same file
   instead of writing a new one. **Clear** empties the field again.
6. **Parameters** — sampling rate, session name, motion type, and optional
   auto-scoring with adjustable thresholds. A `sleep_score_model.npz` in the
   LFP folder — see
   [fitting the auto-scorer](python/README.md#fitting-the-auto-scorer-to-your-own-scoring)
   — scores in place of the thresholds.

Click **Launch State Editor**. Press `h` in the editor for the full list of
keyboard/mouse controls.

### Scoring in the editor

- **Arm a state** with the coloured toolbar buttons (or keys `1` awake, `2`
  NREM, `3` REM, `4` intermediate, `5` erase); click again / press `c` to
  un-arm.
- **Score an epoch**: with a state armed, press `Space` at the start and again
  at the end to label that span (minimum 10 s). Clicking never scores.
- **No stranded scraps**: scoring inside an existing epoch can leave a sliver
  of the old label behind. Any run left **shorter than 5 s** is relabelled to
  match whichever adjacent epoch lasts longer (the earlier one if they tie), so
  the hypnogram never carries unusable fragments. `u` undoes the whole edit,
  absorbed bins included.
- **Navigate**: the time cursor is a fixed playhead that holds the **middle**
  of the view — the data moves past it, and it is drawn on every panel,
  spectrograms, motion and the hypnogram bars alike. `← →` step it, a click
  brings that moment to the centre, and **dragging** pans: on a spectrogram /
  motion / state panel it moves the window like the Position slider, on a raw
  LFP trace it scrubs finely (one panel width = the raw window's few seconds).
  Scroll and the **Window** slider zoom about the cursor, `Shift+← →` pans a
  whole window, `Home`/`End` jump to the ends, and **`0`** (or `r`) snaps back
  to the whole recording. (Near either end of the recording the view runs out
  of room, so the cursor sits off-centre.)
- **Saving is automatic**: closing the editor writes the scoring (both `.npz`
  and `.mat`) — to the file you resumed from, or a new dated one — so there is
  no Save button and nothing to remember. `s` forces that same save mid-session
  if you want a checkpoint, **Load** (`l`) opens an earlier scoring, and `u`
  undoes the last change.

## Output

Saved scoring contains:

| Field | Description |
|-------|-------------|
| `states` | Length-N vector (1 s bins), values 0/1/3/4/5 per bin |
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
├── buzsaki_score.py    # optional Buzsáki auto-scoring (WAKE/NREM/REM)
├── fit_auto_score.py   # fit the auto-scorer to hand-scored sessions (+ intermediate)
└── test_pipeline.py    # headless smoke test
LFP_Output/             # example data folder
```

## License

See [LICENSE](LICENSE).
