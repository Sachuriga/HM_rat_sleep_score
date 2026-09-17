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
| `1`–`4` (awake/NREM/REM/intermediate, `0` = erase) | Arm a state, then mark the two time bounds with `Space` `Space` |
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
| `r` | Reset the time axis to the full extent |
| `u` | Undo the last state change |
| `e` / `d` | Toggle add / delete **event** mode, then click to place/remove a mark |
| `[` / `]` | Select the previous / next event number (1–10) |
| `n` / `p` | Jump to the next / previous event of the active number |
| `s` | Save states |
| `l` | Load states |
| `h` | Toggle the on-screen help overlay |

Scoring inside an existing epoch can strand a sliver of the old label: any run
left **shorter than 5 s** (`MIN_RUN_S`) is relabelled to match whichever
adjacent epoch is longer — ties going to the earlier one — so the hypnogram
holds no unusable fragments. `u` undoes the edit together with everything it
absorbed. The time cursor is drawn on every panel, the hypnogram bars included.

The editor also shows an always-visible **state-colour legend** and a **live
info panel** (armed state, current time, recording length, percent scored, and
per-state bin counts). Each spectrogram and LFP trace is labelled with its
channel number, the window title shows a `*` when there are unsaved changes, and
closing with unsaved work prompts you to save first.

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
as WAKE). The automatic labels use only these three states — intermediate
sleep is a manual-only label, so it never appears in the `Auto` bar. Without an
EMG/motion signal the REM gate
uses theta only (REM tends to be over-called — supply `emg_from_lfp*.npy` for
a proper split).

Generate labels for a folder:

```bash
python buzsaki_score.py --lfp_folder /path/to/LFP_Output   # writes buzsaki_states.npz
```

### Seeing the labels in the editor

The state editor shows an **extra colour bar** (`W`/`N`/`R`) above the manual state bar
whenever auto-labels are supplied — it pans and zooms in lock-step with everything else,
so you can score by hand while comparing against the automatic labels. The setup GUI's
**"Show Buzsáki auto-score"** checkbox loads `buzsaki_states.npz` from the LFP/output
folder if present, or computes it on launch. Programmatically:

```python
StateEditor(..., auto_states=states, auto_states_ts=timestamps)
```

## Files

| File | Purpose |
|------|---------|
| `sleepscore.py` | Entry point (opens the setup GUI) |
| `setup_gui.py` | Tkinter setup launcher — port of `Sleep_score_HM_neuron.m` |
| `state_editor.py` | Matplotlib state editor — port of `TheStateEditor.m` (+ auto-label panel) |
| `processing.py` | Preprocessing + multitaper spectrogram |
| `buzsaki_score.py` | Buzsáki auto sleep scoring (WAKE/NREM/REM) → `buzsaki_states.npz` |
| `test_pipeline.py` | Headless smoke test (`python test_pipeline.py`) |

## Differences from the MATLAB version

- Scoring, navigation, events (add/delete/next/prev), save/load and transitions
  are ported; the right-hand info/legend panel replaces the MATLAB control panel.
- The exact-frequency (`F`) resize mode is replaced by scroll/zoom.
- Spectrograms render with matplotlib's `jet` colormap as in the original.
- Data loads from either `lfp_data.npy` or a `channels_npy/` folder of
  per-tetrode files; the MATLAB GUI reads only `lfp_data.npy`.
