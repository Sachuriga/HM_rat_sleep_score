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
assert (ed.states == 3).all(), "bins should default to NREM"
ed._apply_state(10, 25, 3)     # NREM 10-25 s
ed._apply_state(40, 55, 5)     # REM  40-55 s
ed._apply_state(60, 70, 1)     # awake
assert (ed.states[10:26] == 3).all()
assert (ed.states[40:56] == 5).all()
ed._undo()                     # undoes the awake block
assert (ed.states[60:71] == 3).all()   # back to the NREM default

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

# Auto-saved results file (what closing the editor window writes) for a FRESH
# session — a dated results_ file. (The load above made this a resumed session,
# whose save target is the loaded file; the resume case is tested below.)
assert ed.results_path == path, "loading a scoring should adopt it as save target"
ed.results_path = None
ed.labeled_by = "Test User"
rpath = ed.save_results()
assert os.path.isfile(rpath) and os.path.basename(os.path.dirname(rpath)) == "results"
assert "_Test_User" in os.path.basename(rpath)
d = np.load(rpath)
assert str(d["labeled_by"]) == "Test User"
assert d["states"].size == to.size
print(f"results auto-save ok: {rpath}")

# Buzsáki 3-state tree (Watson et al. 2016): NREM by slow waves first, REM by
# theta among quiet non-NREM bins, everything else (incl. movement) WAKE.
import buzsaki_score as bz
sw_m    = np.array([0.9, 0.9, 0.1, 0.1, 0.1, 0.9])
theta_m = np.array([0.1, 0.1, 0.9, 0.9, 0.1, 0.1])
emg_m   = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.1])
st, thr = bz.cluster_states(sw_m, theta_m, emg_m,
                            swthresh=0.5, ththresh=0.5, emgthresh=0.5)
#            NREM      NREM      REM     WAKE(mov)  WAKE     NREM(SW wins)
assert st.tolist() == [bz.NREM, bz.NREM, bz.REM, bz.WAKE, bz.WAKE, bz.NREM]
# legacy light/drowsy (2) -> awake; intermediate (4) is scored, so it survives
from state_editor import sanitize_states, KEY_TO_STATE
assert sanitize_states([0, 1, 2, 3, 4, 5]).tolist() == [0, 1, 1, 3, 4, 5]
assert KEY_TO_STATE == {"1": 1, "2": 3, "3": 5, "4": 4, "5": 0}   # 0 = reset view
print("Buzsáki 3-state clustering + legacy-code mapping ok")

# Resuming a scored file: the scorer name comes from the file (no prompt) and
# saving updates that same file instead of writing a new one.
import glob, tempfile
rdir = tempfile.mkdtemp()
ed.results_folder = rdir
ed.results_path = None                     # a fresh session
ed.labeled_by = "Test User"
first = ed.save_results()
assert len(glob.glob(os.path.join(rdir, "*.npz"))) == 1

ed2 = StateEditor("test", specs, fos, to, motion, raw_eeg, fs, out_folder="/tmp")
ed2.results_folder = rdir
assert ed2.labeled_by is None              # would prompt on launch...
ed2.load_states(first)
assert ed2.labeled_by == "Test User", ed2.labeled_by   # ...but the file names the scorer
assert ed2.results_path == first
assert (ed2.states[40:56] == 5).all(), "resumed labels not shown"
ed2._apply_state(80, 95, 1)                # edit, then auto-save as on close
again = ed2.save_results()
assert again == first, f"made a new file: {again} != {first}"
assert len(glob.glob(os.path.join(rdir, "*.npz"))) == 1, "a second results file appeared"
assert (np.load(first)["states"][80:96] == 1).all(), "edit not written back"

# a file with no labeled_by field falls back to the name in its filename
from state_editor import _labeled_by_of
assert _labeled_by_of({}, "results_2026-09-17_Sachuriga_R.npz") == "Sachuriga R"
assert _labeled_by_of({}, "session-states.npz") is None
print("resume round-trip ok: same scorer, same file")

# Intermediate sleep (4) is scoreable and round-trips through a save/load
ed3 = StateEditor("test", specs, fos, to, motion, raw_eeg, fs, out_folder="/tmp")
ed3._apply_state(30, 45, 4)
assert (ed3.states[30:46] == 4).all(), "intermediate not applied"
ip = ed3.save_states_npz("/tmp/test-inter.npz")
ed3.states[:] = 0
ed3.load_states(ip)
assert (ed3.states[30:46] == 4).all(), "intermediate lost in round trip"
print("intermediate state ok")

# Navigation: the cursor holds the middle of the view, and dragging pans it
class _Ev:                       # minimal stand-in for a matplotlib MouseEvent
    def __init__(self, **kw): self.__dict__.update(kw)

span = ed3.lims[1] - ed3.lims[0]
ed3._set_xlim(ed3.lims[0], ed3.lims[0] + span / 4)     # zoom in so panning has room
ed3.cursor_time = float(ed3.lims[0] + span / 2)
ed3._centre_view()
lo, hi = ed3._xlim_get()
assert abs((lo + hi) / 2 - ed3.cursor_time) < 1e-6, "cursor should be centred"

ed3._move_cursor(1, steps=5)                            # arrows keep it centred
lo, hi = ed3._xlim_get()
assert abs((lo + hi) / 2 - ed3.cursor_time) < 1e-6, "cursor drifted off centre"

ax = ed3.ax_spec[0]
before = ed3._xlim_get()
ed3._on_press(_Ev(button=1, x=400.0, xdata=before[0], inaxes=ax))
ed3._on_motion(_Ev(button=1, x=300.0, xdata=None, inaxes=ax))   # drag 100 px left
ed3._on_release(_Ev(button=1, x=300.0, xdata=None, inaxes=ax))
after = ed3._xlim_get()
assert after[0] > before[0], f"drag did not pan forward: {before} -> {after}"
assert abs((after[1] - after[0]) - (before[1] - before[0])) < 1e-6, "pan changed zoom"
assert abs(sum(after) / 2 - ed3.cursor_time) < 1e-6, "cursor not centred after drag"

held = ed3._xlim_get()                                  # a click must not pan
ed3._on_press(_Ev(button=1, x=300.0, xdata=held[0] + 5, inaxes=ax))
ed3._on_release(_Ev(button=1, x=301.0, xdata=held[0] + 5, inaxes=ax))
assert abs(ed3.cursor_time - (held[0] + 5)) < 1e-6, "click did not move the cursor"
assert ed3._drag is None
print("centred cursor + drag-pan ok")

# Short runs (< MIN_RUN_S) are absorbed by whichever neighbour lasts longer
from state_editor import absorb_short_runs, state_runs, MIN_RUN_S
assert absorb_short_runs([1]*10 + [5]*3 + [3]*6, 5).tolist() == [1]*13 + [3]*6
assert absorb_short_runs([1]*6 + [5]*2 + [3]*6, 5).tolist() == [1]*8 + [3]*6  # tie -> earlier
assert absorb_short_runs([1]*8 + [3]*2 + [5]*2 + [1]*8, 5).tolist() == [1]*20  # chain
assert absorb_short_runs([3]*4, 5).tolist() == [3]*4          # one run: untouched
assert absorb_short_runs([], 5).tolist() == []

ed4 = StateEditor("test", specs, fos, to, motion, raw_eeg, fs, out_folder="/tmp")
ed4._apply_state(40, 55, 5)
snapshot = ed4.states.copy()
ed4._apply_state(60, 70, 1)        # leaves a 4 s scrap of NREM at 56-60 s
runs = [(b - a, v) for a, b, v in state_runs(ed4.states)]
assert all(n >= 5 for n, _ in runs), f"a run under 5 s survived: {runs}"
assert (ed4.states[56:60] == 5).all(), "scrap should join the longer REM epoch"
ed4._undo()                        # undo restores the absorbed bins too
assert (ed4.states == snapshot).all(), "undo did not restore the absorbed scrap"
print(f"short-run absorption ok (< {MIN_RUN_S:.0f} s)")

# The cursor is drawn on the state bar(s) and tracks the others
assert ed4._state_cursor is not None, "no cursor on the state view"
ed4.cursor_time = float(ed4.lims[0] + (ed4.lims[1] - ed4.lims[0]) / 3)
ed4._centre_view()
assert abs(float(ed4._state_cursor.get_xdata()[0]) - ed4.cursor_time) < 1e-6
ed4._refresh_state_bar()           # redrawing the bar keeps the cursor
assert abs(float(ed4._state_cursor.get_xdata()[0]) - ed4.cursor_time) < 1e-6
print("state-view cursor ok")

# Dragging a raw-LFP trace scrubs the cursor at that panel's finer scale
raw_ax = ed4.ax_eeg[0]
sp = ed4.lims[1] - ed4.lims[0]
ed4._set_xlim(ed4.lims[0], ed4.lims[0] + sp / 4)   # zoomed in, so the view can follow
ed4._centre_view()
c0 = ed4.cursor_time
ed4._on_press(_Ev(button=1, x=400.0, xdata=c0, inaxes=raw_ax))
ed4._on_motion(_Ev(button=1, x=380.0, xdata=None, inaxes=raw_ax))  # drag 20 px left
ed4._on_release(_Ev(button=1, x=380.0, xdata=None, inaxes=raw_ax))
moved = ed4.cursor_time - c0
assert 0 < moved < ed4.eeg_show, f"raw drag should scrub forward a little: {moved}"
assert abs(sum(ed4._xlim_get()) / 2 - ed4.cursor_time) < 1e-6, "view lost the cursor"
print(f"raw-trace drag ok (scrubbed {moved:.3f} s per 20 px)")

# Zooming keeps the width it is asked for: a full-span window centred near an
# end slides into range instead of being trimmed (which used to strand the
# Window slider below full width, with no way back).
full = ed4.lims[1] - ed4.lims[0]
ed4.cursor_time = ed4.lims[0] + 0.1 * full          # well off-centre
ed4._zoom_to(30)
for _ in range(3):                                   # repeated asks must not decay
    ed4._on_win_slider(full)
    lo, hi = ed4._xlim_get()
    assert abs((hi - lo) - full) < 1e-6, f"window stuck at {hi - lo:.0f} of {full:.0f} s"

ed4._set_xlim(ed4.lims[1] - 20, ed4.lims[1])         # zoomed in at the far end
ed4.cursor_time = ed4.lims[1] - 10
for _ in range(25):
    ed4._on_scroll(_Ev(button="down", inaxes=ed4.ax_spec[0], xdata=ed4.cursor_time))
lo, hi = ed4._xlim_get()
assert abs((hi - lo) - full) < 1e-6, f"scrolling out reached only {hi - lo:.0f} s"
ed4._zoom_to(1e9)                                    # absurd width clamps to the span
assert abs(np.diff(ed4._xlim_get())[0] - full) < 1e-6
print("zoom reaches full width from any cursor position")

# `0` resets the view to the whole recording; `5` arms erase
ed4._zoom_to(25)
ed4._on_key(_Ev(key="0"))
lo, hi = ed4._xlim_get()
assert abs((hi - lo) - full) < 1e-6, "key 0 did not reset the view"
assert ed4.current_state is None, "key 0 must not arm a state"
ed4._on_key(_Ev(key="5"))
assert ed4.current_state == 0, "key 5 should arm erase"
ed4._on_key(_Ev(key="c"))
print("key 0 = reset view, key 5 = erase")

print("\nALL CHECKS PASSED")
