# Rat Sleep Scoring Toolkit v1

MATLAB GUI for manual sleep state scoring of rodent LFP recordings using an interactive state editor.

## Overview

This toolkit provides a point-and-click workflow to load LFP (local field potential) recordings saved as NumPy `.npy` files, select channels and a motion signal, and manually score sleep states using `TheStateEditor`. Sleep states are classified into:

| Code | State |
|------|-------|
| 0 | No state |
| 1 | Awake |
| 2 | Light / Drowsy |
| 3 | NREM |
| 4 | Intermediate |
| 5 | REM |

The original `TheStateEditor` was written by Dr. Andres Grosmark and Dr. Abdel Rayan. It was modified and generalized by Sachuriga.

## Repository Structure

```
scr/
├── Sleep_score_HM_neuron.m      # Main GUI — run this to start
├── TheStateEditor.m             # Interactive GUI for manual sleep scoring
├── HM_neurons.eegstates.mat     # Example cached spectrogram/channel state file
└── tool/
    └── npy-matlab-master/       # Third-party library for reading .npy files in MATLAB

LFP_Output/                      # Example data folder
├── channels_npy/
│   ├── lfp_nt01_ch01.npy        # Per-tetrode LFP channel files (nt01–nt32)
│   └── ...
├── emg_rms.npy                  # EMG RMS (auto-detected as motion signal)
├── emg_data.npy                 # Raw EMG data
├── theta_delta_ratio.npy        # Theta/delta ratio (fallback motion signal)
├── lfp_data.npy                 # Full LFP matrix (all channels)
└── lfp_timestamps.npy           # Timestamps
```

## Requirements

- MATLAB (any recent version)
- [`npy-matlab`](https://github.com/kwikteam/npy-matlab) — included under `scr/tool/npy-matlab-master/`

## Usage

Run the setup GUI from MATLAB:

```matlab
cd scr
Sleep_score_HM_neuron()
```

The GUI will open with the following steps:

### 1. Select LFP Output Folder
Click **Browse** and select the folder containing your LFP data (e.g. `LFP_Output/`). The GUI will automatically scan for channel files named `lfp_ntXX_ch01.npy` and populate the channel list.

### 2. Select 3 Channels
The channel list shows all available channels found in the `channels_npy/` subfolder. Hold **Ctrl** (Windows/Linux) or **Cmd** (Mac) and click to select exactly 3 channels.

### 3. Motion / EMG File
The GUI auto-detects a motion file from the LFP folder in priority order:

| Priority | File | Description |
|----------|------|-------------|
| 1st | `emg_rms.npy` | EMG root-mean-square (preferred) |
| 2nd | `emg_data.npy` | Raw EMG data |
| 3rd | `theta_delta_ratio.npy` | Theta/delta ratio |
| 4th | `awakeness.npy` | Awakeness signal |

You can also click **Browse** to select any `.npy` file manually.

### 4. Select Output Folder
Choose where `TheStateEditor` will save its output files. Defaults to the LFP folder.

### 5. Set Parameters
- **Sampling Rate (Hz):** LFP sampling rate (default: `1000`)
- **Session Name:** Base name for output files (default: `HM_neurons`)

### 6. Launch
Click **Launch TheStateEditor**. The GUI loads the 3 selected channels and motion signal, then opens the state editor. Press `H` inside the editor for a full list of keyboard shortcuts.

## Output

Pressing `S` in the state editor saves a `<SessionName>-states.mat` file to the output folder containing:

| Field | Description |
|-------|-------------|
| `states` | Vector of length N (seconds), values 0–5 indicating sleep state per bin |
| `events` | N×2 matrix of event numbers and timestamps (seconds) |
| `transitions` | N×3 matrix of exact state transitions: [state, start\_time, end\_time] |

A `<SessionName>.eegstates.mat` cache file is also created on first run to store whitened spectrograms, speeding up subsequent loads.

## Motion Input Options

`TheStateEditor` supports the following `MotionType` values:

| Value | Description |
|-------|-------------|
| `'none'` | No motion signal |
| `'File'` | Pre-computed motion signal passed directly (used by this GUI) |
| `'Whl'` | Wheel data |
| `'Channels (accelerometer)'` | Raw accelerometer channels |
| `'Channels (MEG)'` | MEG channels |

## License

See [LICENSE](LICENSE).
