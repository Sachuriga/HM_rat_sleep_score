"""Headless smoke test of the processing + editor pipeline (no GUI window)."""
import os
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
from processing import (mad_clip, notch_filter, whiten_ar1,
                        multitaper_spectrogram, compute_channel_spectrogram,
                        downsample_motion)
from state_editor import StateEditor

fs = 1000
dur = 120  # seconds
t = np.arange(dur * fs) / fs
rng = np.random.default_rng(0)

# Synthetic 3-channel LFP: 1/f-ish noise + theta + a 50 Hz line component + artifacts
def make_channel(seed):
    r = np.random.default_rng(seed)
    x = np.cumsum(r.standard_normal(t.size)) * 0.3        # brownian -> 1/f
    x += 2 * np.sin(2 * np.pi * 7 * t)                    # theta
    x += 1.5 * np.sin(2 * np.pi * 50 * t)                 # line noise
    x[10000:10005] = 80                                   # artifact spike
    return x

chs = [make_channel(s) for s in (1, 2, 3)]

specs, fos, raw_eeg, to = [], [], [], None
for c in chs:
    spec, fo, to, cleaned = compute_channel_spectrogram(c, fs)
    specs.append(spec); fos.append(fo); raw_eeg.append(cleaned)
    print(f"spec shape {spec.shape}, fo[{fo[0]:.2f}..{fo[-1]:.2f}], to[{to[0]:.0f}..{to[-1]:.0f}]")

# Check the 50 Hz notch knocked down the line component in the cleaned trace
import numpy as _np
def line_power(x, f0, fs):
    X = _np.abs(_np.fft.rfft(x)); fr = _np.fft.rfftfreq(x.size, 1/fs)
    return X[_np.argmin(_np.abs(fr - f0))]   # the exact 50 Hz bin
raw50 = line_power(chs[0], 50, fs); clean50 = line_power(raw_eeg[0].astype(float), 50, fs)
print(f"50 Hz line power raw={raw50:.1f} cleaned={clean50:.1f}  (notch ratio {clean50/raw50:.4f})")
assert clean50 < raw50 * 0.05, "notch did not attenuate 50 Hz"
spiky = chs[0].copy(); spiky[:] = rng.standard_normal(spiky.size); spiky[123] = 500.0
clipped = mad_clip(spiky)
print(f"MAD clip: raw max {_np.abs(spiky).max():.1f} -> clipped max {_np.abs(clipped).max():.2f}")
assert _np.abs(clipped).max() < 20, "MAD clip failed to remove outlier"

motion = downsample_motion(rng.standard_normal(dur * fs), raw_eeg[0].size, fs)
print(f"motion bins {motion.size}, spectrogram bins {to.size}")

# Build the editor headlessly and exercise scoring/save/load/transitions
ed = StateEditor("test", specs, fos, to, motion, raw_eeg, fs, out_folder="/tmp")
ed._apply_state(10, 25, 3)     # NREM 10-25 s
ed._apply_state(40, 55, 5)     # REM  40-55 s
ed._apply_state(60, 70, 1)     # awake
assert (ed.states[10:26] == 3).all()
assert (ed.states[40:56] == 5).all()
ed._undo()                     # undoes the awake block
assert (ed.states[60:71] == 0).all()

path = "/tmp/test-states.mat"
ed.save_states(path)
from scipy.io import loadmat
d = loadmat(path)
print("saved keys:", [k for k in d if not k.startswith("__")])
print("transitions:\n", d["transitions"])
assert d["states"].ravel().size == to.size

ed.states[:] = 0
ed.load_states(path)
assert (ed.states[10:26] == 3).all(), "load round-trip failed"

print("\nALL CHECKS PASSED")
