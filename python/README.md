# Rat Sleep Scoring Toolkit — Python port

A Python reimplementation of the MATLAB sleep-scoring GUI in [`../scr/`](../scr).
Load LFP `.npy` recordings, pick three channels and a motion signal, view
whitened multitaper spectrograms, and score sleep states by hand.

| Code | State |
|------|-------|
| 0 | No state |
| 1 | Awake |
| 2 | Light / Drowsy |
| 3 | NREM |
| 4 | Intermediate |
| 5 | REM |

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

1. **Browse** to the LFP folder containing `lfp_data.npy` (shape
   `[samples, channels]`).
2. Enter three **channel numbers** (1-based).
3. The **motion/EMG** file is auto-detected in priority order
   `emg_rms.npy → emg_data.npy → theta_delta_ratio.npy → awakeness.npy`, or
   browse to any `.npy` manually.
4. Choose an **output folder** (defaults to the LFP folder).
5. Set **sampling rate** (default 1000 Hz) and **session name**.
6. Click **Launch State Editor**.

### State editor keyboard shortcuts

| Key | Action |
|-----|--------|
| `0`–`5` | Arm a state, then click the two time bounds on any spectrogram/motion/state panel |
| `c` | Cancel the current state action |
| Left / Right | Pan the view |
| Scroll wheel | Zoom in / out (around the cursor) |
| Up / Down | Increase / decrease spectrogram contrast |
| `-` / `=` | Decrease / increase the LFP display width |
| Single click (no armed state) | Centre the LFP view on the click |
| `r` | Reset the time axis to the full extent |
| `u` | Undo the last state change |
| `s` | Save states |
| `l` | Load states |
| `h` | Print help to the console |

## Output

Saving writes `<SessionName>-states.mat` to the output folder in the **same
MATLAB-compatible format** as the original tool (via `scipy.io.savemat`), so the
files are interchangeable with the MATLAB toolkit:

| Field | Description |
|-------|-------------|
| `states` | `1×N` vector (N = number of 1 s bins), values 0–5 |
| `events` | `N×2` event matrix (events not yet implemented in the Python UI; saved empty) |
| `transitions` | `N×3` `[state, start_s, end_s]` for each contiguous scored run |

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

## Files

| File | Purpose |
|------|---------|
| `sleepscore.py` | Entry point (opens the setup GUI) |
| `setup_gui.py` | Tkinter setup launcher — port of `Sleep_score_HM_neuron.m` |
| `state_editor.py` | Matplotlib state editor — port of `TheStateEditor.m` |
| `processing.py` | Preprocessing + multitaper spectrogram |
| `test_pipeline.py` | Headless smoke test (`python test_pipeline.py`) |

## Differences from the MATLAB version

- Events (`E`/`D`/`N`/`P`) and the right-hand control panel are not ported; the
  core scoring, navigation, save/load and transitions are.
- The exact-frequency (`F`) resize mode is replaced by scroll/zoom.
- Spectrograms render with matplotlib's `jet` colormap as in the original.
