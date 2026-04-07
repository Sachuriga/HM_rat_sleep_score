# Rat Sleep Scoring Toolkit

MATLAB scripts for manual sleep state scoring of rodent LFP recordings using a GUI-based state editor.

## Overview

This toolkit provides a workflow to load LFP (local field potential) recordings saved as NumPy `.npy` files and manually score sleep states using `TheStateEditor` — an interactive MATLAB GUI. Sleep states are classified into:

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
├── Sleep_score_HM_neuron.m      # Entry-point script: loads data and launches the editor
├── TheStateEditor.m             # Interactive GUI for manual sleep scoring
├── HM_neurons.eegstates.mat     # Example cached spectrogram/channel state file
└── tool/
    └── npy-matlab-master/       # Third-party library for reading .npy files in MATLAB
```

## Requirements

- MATLAB (any recent version)
- [`npy-matlab`](https://github.com/kwikteam/npy-matlab) — included under `scr/tool/npy-matlab-master/`

## Setup

1. Add the `npy-matlab` functions to your MATLAB path:
   ```matlab
   addpath(genpath('scr/tool/npy-matlab-master/npy-matlab-master/npy-matlab'))
   ```

2. Add `TheStateEditor.m` to your MATLAB path or navigate to the `scr/` directory.

## Usage

Edit `Sleep_score_HM_neuron.m` to point to your data files and configure your recording parameters, then run it from MATLAB.

### Key parameters to configure

```matlab
% Paths to your .npy files
eegData  = readNPY('path/to/lfp_data.npy');          % [channels x samples]
timeVec  = readNPY('path/to/lfp_timestamps.npy');
motion   = readNPY('path/to/theta_delta_ratio.npy'); % motion proxy signal

% Recording parameters
eegFS      = 1000;          % LFP sampling rate in Hz
chsToUse   = [7, 16, 19];  % Channel indices to display (max 3)
baseName   = 'HM_neurons';  % Base name for output files
```

### Launching the editor

```matlab
TheStateEditor(baseName, inputData);
```

The GUI will open. Press `H` inside the editor for a full list of keyboard shortcuts.

## Output

Pressing `S` in the editor saves a `baseName-states.mat` file containing:

| Field | Description |
|-------|-------------|
| `states` | Vector of length N (seconds), values 0–5 indicating sleep state per bin |
| `events` | N×2 matrix of event numbers and timestamps (seconds) |
| `transitions` | N×3 matrix of exact state transitions: [state, start_time, end_time] |

A `baseName.eegstates.mat` cache file is also created on first run to store whitened spectrograms, speeding up subsequent loads.

## Motion Input Options

The `MotionType` field in `inputData` controls how motion is handled:

| Value | Description |
|-------|-------------|
| `'none'` | No motion signal |
| `'File'` | Pre-computed motion signal passed directly via `inputData.motion` |
| `'Whl'` | Wheel data |
| `'Channels (accelerometer)'` | Raw accelerometer channels |
| `'Channels (MEG)'` | MEG channels |

## License

See [LICENSE](LICENSE).