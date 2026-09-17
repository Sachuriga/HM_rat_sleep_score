# Rat Sleep Scoring Toolkit — Python port

A Python reimplementation of the MATLAB sleep-scoring GUI in [`../scr/`](../scr).
Load LFP `.npy` recordings, pick three channels and a motion signal, view
whitened multitaper spectrograms, and score sleep states by hand.

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

## Requirements

- Python ≥ 3.11
- `numpy`, `scipy`, `matplotlib` (`pip install -r requirements.txt`)
- `tkinter` — bundled with CPython on Windows/macOS; on Linux install the
  `python3-tk` system package.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python sleepscore.py
```

This opens the **setup GUI** (port of `Sleep_score_HM_neuron.m`):

1. **Browse** to the LFP folder. Two data layouts are supported, checked in
   this order:
   - `lfp_data.npy` — a single `[samples, channels]` matrix (as in the MATLAB
     `Sleep_score_HM_neuron.m`); or
   - `channels_npy/lfp_ntNN_ch01.npy` — one file per tetrode/channel (the layout
     the example `../LFP_Output/` ships), where channel number `N` maps to
     `lfp_ntNN_ch01.npy`.

   The GUI then shows the **available channel range and recording duration**, and
   **auto-detects the sampling rate** from `lfp_timestamps.npy` when present.
2. Enter three **channel numbers** (1-based).
3. The **motion** file is auto-detected in priority order
   `motion.npy → emg_rms.npy → emg_data.npy → theta_delta_ratio.npy →
   awakeness.npy`, or browse to any `.npy` manually.
4. Choose the **motion type** (see below).
5. Choose an **output folder** (defaults to the LFP folder).
6. Set **sampling rate** and **session name**.
7. Click **Launch State Editor**.

### Motion type

The dropdown selects how the motion file is turned into the panel signal,
mirroring `TheStateEditor.m`'s motion branches:

| Mode | Port | Processing |
|------|------|-----------|
| **Accelerometer (case 3)** — *default* | `MotionType='Channels (accelerometer)'` | per-channel `\|z-score\|` → sum across channels → 0.1–1 Hz FIR band-pass → 1 s bins |
| MEG (case 4) | `MotionType='Channels (MEG)'` | z-score + sum → 100–600 Hz band-pass → square → 0.1–1 Hz band-pass → 1 s bins |
| File / precomputed (case 5) | `MotionType='File'` | no processing — just downsample a precomputed signal to 1 s bins |

Accelerometer is the default so the result matches the original
`sleep_scorer_andres.m` workflow (which passes raw accelerometer channels).
Multi-channel motion (`motion.npy`, shape `[samples, channels]` or
`[channels, samples]`) is aligned to the LFP length automatically, so very large
motion files stay cheap to load. Use **File** mode for an already-computed 1-D
signal such as `emg_rms.npy`.

The first launch computes and **caches the spectrograms** to
`<SessionName>.eegstates.npz`; subsequent launches load it instantly. Tick
**Ignore cache (recompute)** to force a fresh computation (e.g. after changing
channels or the sampling rate).

### State editor keyboard shortcuts

| Key | Action |
|-----|--------|
| `1`–`4` (awake/NREM/REM/intermediate, `5` = erase) | Arm a state, then mark the two time bounds with `Space` `Space` |
| `c` | Cancel the current state action |
| Left / Right | Step the time cursor; the view scrolls to keep it centred |
| Shift + Left / Right | Pan by a whole window |
| Home / End | Jump to the start / end of the recording |
| **Drag** a spectrogram/motion/state panel | Pan the window along time, like the Position slider |
| **Drag** a raw LFP trace | Scrub the cursor at that panel's own (much finer) scale |
| Scroll wheel | Zoom in / out (about the centred cursor) |
| Up / Down | Increase / decrease spectrogram contrast |
| `-` / `=` | Decrease / increase the LFP display width |
| Single click (no armed state) | Move the cursor there — the view re-centres on it |
| `0` / `r` | Reset the time axis to the whole recording |
| `u` | Undo the last state change |
| `e` / `d` | Toggle add / delete **event** mode, then click to place/remove a mark |
| `[` / `]` | Select the previous / next event number (1–10) |
| `n` / `p` | Jump to the next / previous event of the active number |
| `s` | Save now (closing the window saves by itself) |
| `l` | Load a saved scoring |
| `h` | Toggle the on-screen help overlay |

Scoring inside an existing epoch can strand a sliver of the old label: any run
left **shorter than 5 s** (`MIN_RUN_S`) is relabelled to match whichever
adjacent epoch is longer — ties going to the earlier one — so the hypnogram
holds no unusable fragments. `u` undoes the edit together with everything it
absorbed. The time cursor is drawn on every panel, the hypnogram bars included.

The editor also shows an always-visible **state-colour legend** and a **live
info panel** (armed state, current time, recording length, percent scored, and
per-state bin counts). Each spectrogram and LFP trace is labelled with its
channel number, and the status bar names the file the scoring will be written
to. There is no Save button and no unsaved marker: closing the window saves the
scoring automatically (you are only prompted if that write fails).

## Output

Saving writes `<SessionName>-states.mat` to the output folder in the **same
MATLAB-compatible format** as the original tool (via `scipy.io.savemat`), so the
files are interchangeable with the MATLAB toolkit:

| Field | Description |
|-------|-------------|
| `states` | `1×N` vector (N = number of 1 s bins), values 0/1/3/4/5 |
| `events` | `M×2` matrix of `[event_number, time_s]` (empty when no events placed) |
| `transitions` | `N×3` `[state, start_s, end_s]` for each contiguous scored run |

A `<SessionName>.eegstates.npz` spectrogram cache is also written to the output
folder on first run (the Python equivalent of the MATLAB `.eegstates.mat`),
speeding up subsequent loads of the same session.

## Processing pipeline

Each channel is processed identically to the MATLAB source
([`processing.py`](processing.py)):

```
raw LFP → MAD artifact clip (±5σ) → 50 Hz notch (Q=35, zero-phase)
        → AR(1) whitening → multitaper spectrogram (DPSS, NW=3, 5 tapers,
          nFFT=3072, 1 s windows, 0–200 Hz)
```

For display the spectrogram is frequency-binned (~0.5 Hz), smoothed across time
with a 10-point Hann window, log-scaled, and channels 2–3 are normalised to the
dynamic range of channel 1 — matching `TheStateEditor.m`.

## Buzsáki automatic scoring (optional)

`buzsaki_score.py` is a Python port of the brain-state segregation in Watson et
al. (2016, *Neuron* 90:839–852). It labels each 1 s bin **WAKE / NREM / REM**
from three metrics — all 0–1 normalised, each split at its bimodal-histogram dip
(the paper's per-session bimodal cutoffs):

| Metric | How | Separates |
|--------|-----|-----------|
| broadband slow wave | delta (0.5–4 Hz) minus gamma (40–100 Hz) z-scored log power — the paper's PC1 axis (low freqs weighted opposite gamma) | NREM (high mode) |
| theta ratio | narrow-band power ratio 5–10 Hz / 2–16 Hz | REM |
| EMG | EMG-from-LFP (tracker step 8), and/or a provided motion signal | movement gate on REM |

Classified in the paper's order: `NREM = SW>thr`; `REM = ~NREM & theta>thr &
EMG<thr`; `WAKE` = the rest (movement, microarousals and quiet wake all count
as WAKE). This threshold scorer produces only these three states — it has no
notion of intermediate sleep. A model fitted to your own scoring does score
intermediate: see [Fitting the auto-scorer](#fitting-the-auto-scorer-to-your-own-scoring).

The REM movement gate uses the EMG-from-LFP (`emg_from_lfp*.npy`) when the
folder has it, and otherwise the scored channel's own **275–500 Hz power** —
muscle tone read off the LFP itself, which separates WAKE from sleep as well as
a recorded EMG does. So every session gets a gate, including those with no
accelerometer. Accelerometer motion is added only with `--use_motion`: not
every session records it, and a threshold that leans on it does not transfer to
the sessions that don't.

Two thresholds were re-fitted against a hand-scored session (4 h, all states):
`SW_THRESH_FACTOR = 1.25`, because the slow-wave histogram dip sits *below* the
REM mode and the raw cutoff therefore labels most REM as NREM; and the movement
ceiling, now the 60th percentile of the movement signal rather than its bimodal
dip, which lands in very different places on the EMG and high-frequency traces.
Together they take agreement with the hand scoring from 80% (κ 0.57) to 87%
(κ 0.75), mostly by recovering REM (recall 0.37 → 0.81).

Generate labels for a folder:

```bash
python buzsaki_score.py --lfp_folder /path/to/LFP_Output   # writes buzsaki_states.npz
python buzsaki_score.py --lfp_folder ... --model none      # thresholds, ignoring any fitted model
```

### Seeing the labels in the editor

The state editor shows an **extra colour bar** (`W`/`N`/`I`/`R`) above the manual state bar
whenever auto-labels are supplied — it pans and zooms in lock-step with everything else,
so you can score by hand while comparing against the automatic labels. The setup GUI's
**"Show Buzsáki auto-score"** checkbox loads `buzsaki_states.npz` from the LFP/output
folder if present, or computes it on launch. Programmatically:

```python
StateEditor(..., auto_states=states, auto_states_ts=timestamps)
```

## Fitting the auto-scorer to your own scoring

`fit_auto_score.py` replaces those hand-set cutoffs with boundaries *fitted* to
sessions you scored yourself — a Gaussian model per state plus the transition
matrix from your own hypnogram, decoded with Viterbi:

```bash
python fit_auto_score.py --lfp_folder /path/to/LFP_Output \
                         --labels /path/to/results/results_2026-09-17_you.npz
```

It prints held-out agreement (5 contiguous time blocks — never shuffled bins,
since neighbouring seconds are nearly identical) and writes
`<prefix>sleep_score_model.npz` into the LFP folder. From then on
`buzsaki_score.py` and the setup GUI use it automatically and the threshold
multipliers no longer apply; delete the file to go back. Repeat
`--lfp_folder`/`--labels` to fit several sessions at once, which is the better
way to use it — one session teaches it that session's electrodes.

Seven features per 1 s bin, each smoothed 15 s then z-scored within the session
(so electrode gain cancels), **all from the LFP** — no motion, no EMG file:

| Feature | What | Separates |
|---------|------|-----------|
| `sw` | z(log δ 0.5–4) − z(log γ 40–100), cortex | NREM |
| `thdelta` | log θ 5–10 − log δ 0.5–4, cortex | REM (AUC 0.99 vs NREM, against 0.89 for the 5–10/2–16 ratio) |
| `hf` | log 275–500 Hz, cortex | WAKE (AUC 0.96, vs 0.96 for recorded EMG and 0.89 for the accelerometer) |
| `spindle` | log 10–16 Hz − log 1–100 Hz, stratum radiatum | **intermediate** (AUC 0.97 vs everything else) |
| `spindle_hi` | log 13–18 Hz, stratum radiatum | intermediate vs REM (AUC 0.88) |
| `sr_thdelta` | log θ 5–10 − log δ 0.5–4, stratum radiatum | REM |
| `amp_cv` | CV of the 1–30 Hz envelope within each second, cortex | REM, which holds a steadier amplitude than any other state (AUC 0.88 vs intermediate); lifts REM precision 0.81 → 0.90 |

Intermediate sleep is fitted whenever the labels contain at least
`MIN_INTER_BINS` (60) of it; `--no_intermediate` scores WAKE/NREM/REM only.
Held-out agreement on our hand-scored session, 4 states against 3:

| | accuracy | κ | WAKE | NREM | INTER | REM |
|---|---|---|---|---|---|---|
| 4 states | 0.906 | 0.817 | 0.90 / 0.86 | 0.92 / 0.94 | **0.71 / 0.56** | 0.85 / 0.90 |
| 3 states | 0.912 | 0.825 | 0.90 / 0.86 | 0.92 / 0.94 | — | 0.96 / 0.92 |

*(recall / precision per state)*. For reference, two passes of hand scoring over
that same session differ on 3.7% of bins, so much of what is left is boundary
disagreement.

Scoring intermediate costs a little REM recall (0.96 → 0.85) and buys all of it
back as intermediate. Read its bin-level precision (0.56) with care: the state
covers ~1% of the recording, so a single boundary disagreement dominates it.
Epoch-wise, the model **found all 6 hand-scored intermediate epochs** and
proposed 3 extra — two of them at the onset of the two REM epochs scored as
going straight from NREM to REM (so it is proposing a transition the hand
scoring did not mark, not inventing one out of nowhere), and one 13 s epoch
inside NREM. That is why the report prints the epoch count next to the
bin-level numbers.

Two decode rules keep intermediate honest, since a plain HMM enters it at REM
onset and stays there for the whole REM epoch: its epochs are capped at
`INTER_MAX_S` (30 s — the hand-scored ones run 10–46 s) and it carries a small
penalty `INTER_BIAS`. Both are stored in the model file and adjustable
(`--inter_max_s`, `--inter_bias`).

## Files

| File | Purpose |
|------|---------|
| `sleepscore.py` | Entry point (opens the setup GUI) |
| `setup_gui.py` | Tkinter setup launcher — port of `Sleep_score_HM_neuron.m` |
| `state_editor.py` | Matplotlib state editor — port of `TheStateEditor.m` (+ auto-label panel) |
| `processing.py` | Preprocessing + multitaper spectrogram |
| `buzsaki_score.py` | Buzsáki auto sleep scoring (WAKE/NREM/REM) → `buzsaki_states.npz` |
| `fit_auto_score.py` | Fit the auto-scorer to hand-scored sessions → `sleep_score_model.npz` |
| `test_pipeline.py` | Headless smoke test (`python test_pipeline.py`) |

## Differences from the MATLAB version

- Scoring, navigation, events (add/delete/next/prev), save/load and transitions
  are ported; the right-hand info/legend panel replaces the MATLAB control panel.
- The exact-frequency (`F`) resize mode is replaced by scroll/zoom.
- Spectrograms render with matplotlib's `jet` colormap as in the original.
- Data loads from either `lfp_data.npy` or a `channels_npy/` folder of
  per-tetrode files; the MATLAB GUI reads only `lfp_data.npy`.
