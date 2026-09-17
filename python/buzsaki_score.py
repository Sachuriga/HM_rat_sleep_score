"""Buzsáki automatic sleep scoring (WAKE / NREM / REM) from LFP + EMG.

A Python port of the brain-state segregation in Watson, Levenstein, Greene,
Gelinas & Buzsáki (2016, Neuron 90:839–852, "Network Homeostasis and State
Dynamics of Neocortical Sleep"), using the EMG-from-LFP signal produced by the
tracker's step 8.

Three metrics, one value per 1 s bin (each smoothed, 0–1 normalised, and
thresholded at the dip of its bimodal histogram, per session — the paper's
"cutoffs at the minima of bimodal distributions"):
  * broadbandSlowWave = delta-vs-gamma contrast of the log spectrogram
    (z-scored delta power minus z-scored gamma power — the axis the paper's
    spectrogram PC1 captures: low frequencies weighted opposite in sign to
    gamma; its high mode is NREM).
  * thratio           = narrow-band theta power ratio, 5–10 Hz / 2–16 Hz.
  * EMG               = EMG-from-LFP (muscle tone), and/or accelerometer motion.

Classified in the paper's order — slow waves first, then theta/EMG:
  NREM  = SW > swthresh                            (high-PC1 mode)
  REM   = ~NREM & theta > ththresh & EMG/motion quiescent
  WAKE  = everything else (movement, microarousals, quiet wake)
Only these three states are produced; the paper's microarousals and any
transitional/intermediate substates are folded into WAKE.

States use the HM codes: 1 = WAKE, 3 = NREM, 5 = REM (0 = unscored).

Two scorers live here, and **this threshold scorer is the default**: it needs
no training and runs on any session. A model fitted to hand-scored sessions
(see ``fit_auto_score.py``) reached kappa 0.82 on the session it was fitted to,
but a model fitted to one session of one rat did not transfer to other
recordings, so it is opt-in per folder (``model="auto"`` or a path) rather than
picked up because a file happens to be there.

The movement gate on REM prefers the EMG-from-LFP file, falling back to the
LFP's own high-frequency power (``HF_BAND``) so sessions with neither an
accelerometer nor an EMG file still get a gate. Accelerometer motion is used
only when ``use_motion=True``: not every session records it, and thresholds
that depend on it don't transfer to the ones that don't.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import spectrogram
from scipy.ndimage import uniform_filter1d

# HM state codes (match TheStateEditor / the Python state editor).
WAKE, NREM, REM = 1, 3, 5

DELTA_BAND = (0.5, 4.0)      # slow-wave (delta) power band (Hz)
GAMMA_BAND = (40.0, 100.0)   # gamma band, weighted opposite to delta (Hz)
TH_BAND = (5.0, 10.0)        # theta band (Hz)
TH_DENOM = (2.0, 16.0)       # theta-ratio denominator band (Hz)
HF_BAND = (275.0, 500.0)     # muscle tone from the LFP's high-frequency tail (Hz)
WINDOW_S = 2.0              # spectrogram window (s)
DT_S = 1.0                 # spectrogram step / bin size (s)
SMOOTH_S = 15.0            # metric smoothing window (s)
TH_THRESH_FACTOR = 0.85    # lower the theta threshold (<1) to accept more REM
# Every cutoff is the paper's own bimodal-histogram dip, unscaled. A x1.25
# slow-wave multiplier and a 60th-percentile movement gate were fitted to one
# hand-scored session here and did not carry over to other recordings, so the
# defaults stay at the published behaviour; pass --sw_factor / --emg_factor (or
# fit a model with fit_auto_score.py) to override per session.
SW_THRESH_FACTOR = 1.0
MOVE_GATE_PCT = None       # None = threshold at the bimodal dip, as published


# ---------------------------------------------------------------------------- #
#  Metrics
# ---------------------------------------------------------------------------- #
def _spectrogram(lfp, fs):
    """Power spectrogram at 1 s steps. Returns (freqs, times, power[F, T])."""
    nperseg = int(round(WINDOW_S * fs))
    noverlap = int(round((WINDOW_S - DT_S) * fs))
    f, t, Sxx = spectrogram(np.asarray(lfp, dtype=np.float64), fs=fs,
                            nperseg=nperseg, noverlap=noverlap,
                            scaling="density", mode="psd")
    return f, t, Sxx


def _smooth(x, fs_bins):
    """Moving-average smooth over SMOOTH_S seconds (fs_bins = bins per second)."""
    win = max(1, int(round(SMOOTH_S * fs_bins)))
    return uniform_filter1d(x, win, mode="nearest")


def _norm01(x):
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo + 1e-12)


def _zscore(x):
    """Z-score, ignoring NaNs (returns a copy)."""
    x = np.asarray(x, dtype=np.float64).copy()
    v = x[~np.isnan(x)]
    if v.size:
        x = (x - v.mean()) / (v.std() + 1e-12)
    return x


def _channel_metrics(lfp, fs):
    """All per-bin metrics of one channel from a single spectrogram.

    Returns ``(times, {"sw", "thratio", "hf"})``, each 0–1 per 1 s bin:

    * ``sw`` — broadbandSlowWave, the delta-vs-gamma spectrogram contrast.
      Implements the axis the paper's spectrogram PC1 captures — power at low
      frequencies weighted opposite in sign to the gamma range (Watson et al.
      2016, Fig. 1) — as z-scored log delta power minus z-scored log gamma
      power. Computing the contrast directly instead of a per-session PCA keeps
      the metric's sign and meaning fixed even when movement/broadband
      artifacts dominate the spectrogram's variance (a broadband power rise
      lifts both terms and cancels). High in NREM; low in theta-dominated REM
      and in desynchronised, gamma-rich WAKE. Cleanest from a CORTEX channel.
    * ``thratio`` — narrow-band theta power ratio, power(5–10 Hz) /
      power(2–16 Hz) (Watson et al. 2016). Strongest from a STRATUM RADIATUM
      channel (theta peaks there in REM).
    * ``hf`` — log power in ``HF_BAND``, muscle tone read off the LFP's own
      high-frequency tail. Separates WAKE from sleep as well as a recorded EMG
      does, and needs no extra file.
    """
    f, t, Sxx = _spectrogram(lfp, fs)
    fs_bins = 1.0 / (np.median(np.diff(t)) if t.size > 1 else DT_S)
    logP = np.log10(Sxx + 1e-12)

    def band(lo, hi):
        return logP[(f >= lo) & (f <= hi), :].mean(axis=0)

    sw = _zscore(band(*DELTA_BAND)) - _zscore(band(*GAMMA_BAND))
    num = (f >= TH_BAND[0]) & (f <= TH_BAND[1])
    den = (f >= TH_DENOM[0]) & (f <= TH_DENOM[1])
    thratio = Sxx[num, :].sum(axis=0) / (Sxx[den, :].sum(axis=0) + 1e-24)
    hf_hi = min(HF_BAND[1], 0.95 * fs / 2)          # stay under Nyquist
    hf = band(min(HF_BAND[0], 0.6 * hf_hi), hf_hi)
    return t, {k: _norm01(_smooth(v, fs_bins))
               for k, v in (("sw", sw), ("thratio", thratio), ("hf", hf))}


def compute_metrics(lfp, fs, emg=None, emg_ts=None, theta_lfp=None):
    """Compute (times, broadbandSlowWave, thratio, emg_aligned, hf), 0–1 per 1 s bin.

    ``lfp`` provides the slow-wave and high-frequency metrics — use a **cortex**
    channel. ``theta_lfp`` (optional) provides the theta metric — use a
    **stratum radiatum** channel where theta peaks; if omitted, theta is taken
    from ``lfp`` too (single-channel mode). ``emg`` + ``emg_ts`` is the
    EMG-from-LFP; when it is missing, ``hf`` stands in as the movement signal.
    """
    t, m = _channel_metrics(lfp, fs)
    sw, thratio, hf = m["sw"], m["thratio"], m["hf"]
    if theta_lfp is not None:
        t_th, m_th = _channel_metrics(theta_lfp, fs)
        thratio = m_th["thratio"]
        # align theta bins onto the SW time base if the two differ in length
        if thratio.size != t.size:
            thratio = np.interp(t, t_th, thratio)

    emg_aligned = None
    if emg is not None:
        emg = np.asarray(emg, dtype=np.float64).ravel()
        if emg_ts is None:
            emg_ts = np.linspace(t[0], t[-1], emg.size)
        emg_aligned = _norm01(np.interp(t, np.asarray(emg_ts).ravel(), emg))

    return t, sw, thratio, emg_aligned, hf


# ---------------------------------------------------------------------------- #
#  Bimodal threshold + clustering
# ---------------------------------------------------------------------------- #
def bimodal_threshold(x, nbins=60, default=0.5, mode="dip"):
    """Threshold at the histogram dip between the two largest modes.

    Mirrors the intent of ``bz_BimodalThresh``: smooth the histogram, find the two
    tallest peaks, and put the threshold at the lowest trough between them. Falls
    back to ``default`` (on the 0–1 metric) when the distribution isn't bimodal.

    ``mode="high"`` bounds the HIGH mode instead: when a significant third mode
    sits between the two tallest peaks (e.g. REM on the slow-wave metric, between
    the WAKE-low and NREM-high modes), the threshold moves to the trough directly
    below the high mode so the middle mode is not swallowed into it. On a cleanly
    bimodal distribution it is identical to ``"dip"``.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if x.size < 10:
        return default
    counts, edges = np.histogram(x, bins=nbins)
    centres = (edges[:-1] + edges[1:]) / 2
    h = uniform_filter1d(counts.astype(float), 3, mode="nearest")

    peaks = [i for i in range(1, len(h) - 1) if h[i] > h[i - 1] and h[i] >= h[i + 1]]
    if len(peaks) < 2:
        return default
    tallest = sorted(peaks, key=lambda i: h[i], reverse=True)
    p1, p2 = sorted(tallest[:2])
    if mode == "high":
        # significant intermediate modes (≥25% of the smaller top peak; tiny
        # noise bumps in an empty gap don't count) pull the threshold up to
        # the trough adjacent to the high mode
        sig = 0.25 * min(h[p1], h[p2])
        mid = [i for i in peaks if p1 < i < p2 and h[i] >= sig]
        if mid:
            p1 = max(mid)
    trough = p1 + int(np.argmin(h[p1:p2 + 1]))
    return float(centres[trough])


def cluster_states(sw, thratio, emg, motion=None, swthresh=None, ththresh=None,
                   emgthresh=None, sw_factor=SW_THRESH_FACTOR,
                   th_factor=TH_THRESH_FACTOR, emg_factor=1.0):
    """Classify each 1 s bin into WAKE/NREM/REM (Watson et al. 2016 order).

    The slow-wave metric's bimodal split labels NREM first; among the remaining
    bins, high theta with quiescent EMG/motion is REM; everything else —
    movement, microarousals, quiet wake — is WAKE:
      NREM = SW > swthresh
      REM  = ~NREM & theta > ththresh & every movement signal below threshold
      WAKE = the rest
    Without any movement signal, theta alone gates REM (over-calls REM — supply
    EMG or motion for a proper split).

    Every cutoff is the dip of its own bimodal histogram, as published, scaled
    by ``sw_factor`` / ``th_factor`` / ``emg_factor`` (th_factor <1 = more REM;
    emg_factor >1 = laxer movement gate on REM). Setting ``MOVE_GATE_PCT``
    puts the movement ceiling at that percentile of the movement signal's own
    distribution instead of at its dip.
    Returns ``(states, thresholds)``.
    """
    n = len(sw)
    # NREM = the HIGH mode of the slow-wave metric: bound it from directly
    # below so a middle (REM) mode is not swallowed into NREM
    swt = (bimodal_threshold(sw, mode="high") * sw_factor
           if swthresh is None else swthresh)
    # th_factor (<1) lowers the theta threshold so more borderline bins are REM
    tht = bimodal_threshold(thratio) * th_factor if ththresh is None else ththresh
    nrem = sw > swt
    hightheta = thratio > tht

    movesigs = [np.asarray(s, dtype=np.float64) for s in (emg, motion) if s is not None]
    quiet = np.ones(n, dtype=bool)
    movt = None
    if movesigs:
        thrs = [(bimodal_threshold(s) if MOVE_GATE_PCT is None
                 else float(np.nanpercentile(s, MOVE_GATE_PCT))) * emg_factor
                if emgthresh is None else emgthresh for s in movesigs]
        for s, thv in zip(movesigs, thrs):
            quiet &= s < thv                  # REM needs EVERY signal quiescent
        movt = thrs[0]

    states = np.full(n, WAKE, dtype=int)
    states[nrem] = NREM
    states[(~nrem) & hightheta & quiet] = REM
    return states, {"swthresh": swt, "ththresh": tht, "emgthresh": movt}


# ---------------------------------------------------------------------------- #
#  Minimum-duration smoothing
# ---------------------------------------------------------------------------- #
def post_rem_arousal(states):
    """Mask of WAKE runs that begin the moment a REM epoch ends.

    REM terminates in an arousal — in our hand scoring every one of the six REM
    epochs is followed by WAKE, and REM never runs straight into NREM. Those
    arousals are often only seconds long, so ordinary min-duration smoothing
    absorbs them back into the REM epoch, and the hypnogram is then left with an
    impossible REM->NREM transition. Marking them keeps that from happening.
    """
    states = np.asarray(states, dtype=int)
    keep = np.zeros(states.size, dtype=bool)
    if states.size == 0:
        return keep
    edges = np.flatnonzero(np.diff(states)) + 1
    for a, b in zip(np.r_[0, edges], np.r_[edges, states.size]):
        if states[a] == WAKE and a > 0 and states[a - 1] == REM:
            keep[a:b] = True
    return keep


def enforce_min_duration(states, min_secs=6, dt=1.0, protect_post_rem=True):
    """Remove state runs shorter than ``min_secs`` by merging into a neighbour.

    A light-weight stand-in for the Buzsáki min-window rules: the shortest blip
    is absorbed into whichever neighbouring epoch is longer (ties to the
    earlier one, as the state editor does), repeated until every run is long
    enough.

    ``protect_post_rem`` exempts the arousal that ends a REM epoch, however
    brief — see :func:`post_rem_arousal`. On our hand-scored session it takes
    the share of REM epochs that correctly end in WAKE from 45% to 96%, and
    costs nothing elsewhere.
    """
    states = np.asarray(states, dtype=int).copy()
    min_bins = max(1, int(round(min_secs / dt)))
    keep = (post_rem_arousal(states) if protect_post_rem
            else np.zeros(states.size, dtype=bool))
    while states.size > 1:
        edges = np.flatnonzero(np.diff(states)) + 1
        starts = np.r_[0, edges]
        ends = np.r_[edges, states.size]
        lens = ends - starts
        short = np.flatnonzero((lens < min_bins) & (states[starts] != 0)
                               & ~np.array([keep[a:b].any()
                                            for a, b in zip(starts, ends)]))
        if short.size == 0 or lens.size < 2:      # nothing short, or a single run
            break
        i = short[np.argmin(lens[short])]         # shortest blip first
        prev_len = lens[i - 1] if i > 0 else -1
        next_len = lens[i + 1] if i + 1 < lens.size else -1
        states[starts[i]:ends[i]] = (states[starts[i - 1]] if prev_len >= next_len
                                     else states[starts[i + 1]])
    return states


# ---------------------------------------------------------------------------- #
#  Full pipeline
# ---------------------------------------------------------------------------- #
def score(lfp, fs, emg=None, emg_ts=None, motion=None, motion_ts=None, min_secs=10,
          swthresh=None, ththresh=None, emgthresh=None, sw_factor=SW_THRESH_FACTOR,
          th_factor=TH_THRESH_FACTOR, emg_factor=1.0, theta_lfp=None):
    """Full Buzsáki auto-scoring. Returns a dict with states, timestamps, metrics.

    ``lfp`` = slow-wave (cortex) channel; ``theta_lfp`` = optional theta
    (stratum radiatum) channel — pass both for layer-specific scoring. REM is
    gated on movement quiescence, measured from ``emg`` (EMG-from-LFP) and/or
    ``motion`` (accelerometer); when neither is given the channel's own
    high-frequency power stands in, so the gate is never simply absent.
    ``sw_factor`` / ``th_factor`` / ``emg_factor`` scale the auto thresholds
    (1.0 = auto), ``min_secs`` the minimum epoch. ``states`` is one HM code
    (1 WAKE / 3 NREM / 5 REM) per 1 s bin; ``timestamps`` bin centres (s).
    """
    t, sw, thratio, emg_a, hf = compute_metrics(lfp, fs, emg=emg, emg_ts=emg_ts,
                                                theta_lfp=theta_lfp)
    motion_a = None
    if motion is not None:
        motion = np.asarray(motion, dtype=np.float64).ravel()
        if motion_ts is None:
            motion_ts = np.linspace(t[0], t[-1], motion.size)
        motion_a = _norm01(np.interp(t, np.asarray(motion_ts).ravel(), motion))
    gate = emg_a if emg_a is not None else (None if motion_a is not None else hf)
    states, thr = cluster_states(sw, thratio, gate, motion=motion_a,
                                 swthresh=swthresh, ththresh=ththresh,
                                 emgthresh=emgthresh, sw_factor=sw_factor,
                                 th_factor=th_factor, emg_factor=emg_factor)
    dt = np.median(np.diff(t)) if t.size > 1 else DT_S
    if min_secs:
        states = enforce_min_duration(states, min_secs=min_secs, dt=dt)
    return {
        "states": states,
        "timestamps": t,
        "metrics": {"broadbandSlowWave": sw, "thratio": thratio, "emg": emg_a,
                    "motion": motion_a, "hf": hf},
        "thresholds": thr,
    }


def save(result, path):
    """Save an auto-scoring result to ``path`` (.npz), readable by the GUI panel."""
    fields = {"states": result["states"].astype(np.int16),
              "timestamps": result["timestamps"].astype(np.float64)}
    for name, values in result["metrics"].items():
        fields[name] = np.array([]) if values is None else np.asarray(values)
    for name, value in result.get("thresholds", {}).items():
        fields[name] = np.nan if value is None else value
    np.savez(path, **fields)
    return path


def load_states(path):
    """Load a saved auto-scoring result: returns (states, timestamps)."""
    d = np.load(path, allow_pickle=False)
    return np.asarray(d["states"]).ravel().astype(int), np.asarray(d["timestamps"]).ravel()


DEFAULT_OUT = "buzsaki_states.npz"


def _nwb_signals(lfp_dir):
    """``(emg, emg_ts, motion, motion_ts)`` from the session NWB, all None when
    there is no NWB. Cached per folder — each call would otherwise reopen it."""
    key = str(lfp_dir)
    if key in _NWB_CACHE:
        return _NWB_CACHE[key]
    out = (None, None, None, None)
    try:
        import sleep_nwb as snwb
        nwb = snwb.find_session_nwb(lfp_dir)
        if nwb is not None:
            inputs = snwb.read_sleep_inputs(nwb, lazy=False)
            out = (inputs.get("emg"), inputs.get("emg_timestamps"),
                   inputs.get("motion"), inputs.get("motion_timestamps"))
    except Exception as exc:
        print(f"  warning: could not read signals from the session NWB: {exc}")
    _NWB_CACHE[key] = out
    return out


_NWB_CACHE = {}


def _load_emg(lfp_dir, fs):
    """EMG-from-LFP for a session: from the NWB, else the legacy ``.npy``."""
    emg, ts, _, _ = _nwb_signals(lfp_dir)
    if emg is not None:
        return np.asarray(emg).ravel(), (None if ts is None else np.asarray(ts).ravel())

    from processing import find_output
    f5 = find_output(lfp_dir, "emg_from_lfp_5hz.npy")   # prefixed or not
    if f5 is not None:
        emg = np.load(f5).ravel()
        ts_file = find_output(lfp_dir, "emg_from_lfp_timestamps.npy")
        ts = np.load(ts_file).ravel() if ts_file is not None else None
        return emg, ts
    fper = find_output(lfp_dir, "emg_from_lfp.npy")
    if fper is not None:
        emg = np.load(fper).ravel()
        return emg, np.arange(emg.size) / fs
    return None, None


def _load_motion(lfp_dir, fs):
    """Accelerometer movement magnitude + timestamps, else (None, None).

    Read from the session NWB when present, else the legacy ``.npy``. Drives
    the wake/sleep split in preference to EMG-from-LFP. Decimated by 100 —
    the trace is smooth at 1500 Hz and the scorer bins it to 1 s anyway.
    """
    _, _, motion, ts = _nwb_signals(lfp_dir)
    if motion is not None:
        m = np.asarray(motion, dtype=np.float64).ravel()[::100]
        if ts is not None:
            t = np.asarray(ts, dtype=np.float64).ravel()[::100]
            n = min(m.size, t.size)
            return m[:n], t[:n]
        return m, np.arange(m.size) * (100.0 / fs)

    from processing import find_output
    f = find_output(lfp_dir, "motion_accel.npy")     # 1-D magnitude, prefixed or not
    if f is None:
        return None, None
    m = np.asarray(np.load(f, mmap_mode="r")[::100], dtype=np.float64).ravel()
    tsf = find_output(lfp_dir, "motion_timestamps.npy")
    if tsf is not None:
        ts = np.asarray(np.load(tsf, mmap_mode="r")[::100], dtype=np.float64).ravel()
        n = min(m.size, ts.size)
        return m[:n], ts[:n]
    return m, np.arange(m.size) * (100.0 / fs)


def _resolve_model(model, lfp_dir):
    """Turn the ``model`` argument into ``(model dict, name)``, or ``(None, None)``."""
    if model is None:
        return None, None
    import fit_auto_score as fa
    if isinstance(model, dict):
        return model, "fitted model"
    path = fa.find_model(lfp_dir) if model == "auto" else model
    if path is None:
        return None, None
    return fa.load_model(path), Path(path).name


def score_from_lfp_output(lfp_dir, channel=None, ctx_channel=None,
                          sr_channel=None, fs=None, model=None,
                          use_motion=False, **kw):
    """Run the pipeline on an LFP_Output folder. Returns (result, channel_used).

    Layer-specific channels (recommended, per-rat): ``ctx_channel`` (cortex) drives
    the slow-wave/NREM metric, ``sr_channel`` (stratum radiatum) drives the
    theta/REM metric. If only ``channel`` (or none) is given, a single channel
    drives both (legacy behaviour). Channel numbers are 1-based tetrode numbers
    (channels_npy) or 1-based columns (lfp_data.npy), per find_lfp_source.

    ``model`` selects the scorer. The default ``None`` is the threshold scorer
    above — the published parameters, nothing fitted. ``"auto"`` opts in to a
    model fitted to hand-scored sessions if one is found in the folder (see
    ``fit_auto_score.py``); a path or a loaded model forces a specific one.
    A fitted model is deliberately *not* the default: one fitted to a single
    session of one rat scored well on that session and did not transfer, so it
    has to be asked for, per folder, rather than picked up because a file
    happens to sit there.

    ``use_motion`` adds the accelerometer to the REM movement gate. Off by
    default: not every session records motion, so leaving it out keeps one
    session's scoring comparable with the next.

    ``fs`` overrides the sampling rate detected from lfp_timestamps.npy — the
    setup GUI passes its validated LFP rate. When detecting, a rate above
    LFP_FS_MAX is the raw acquisition rate leaking in (e.g. 30 kHz on 1500 Hz
    LFP data), which would stretch every metric/timestamp ~20×, so it is
    rejected in favour of the 1500 Hz default.
    """
    from processing import (find_lfp_source, load_lfp_channel,
                            detect_sampling_rate, find_output, LFP_FS_MAX)

    lfp_dir = Path(lfp_dir)
    src = find_lfp_source(str(lfp_dir))
    if src is None:
        raise FileNotFoundError(f"no lfp_data.npy or channels_npy/ in {lfp_dir}")
    if fs is None:
        # an NWB source carries its own rate; else fall back to lfp_timestamps.npy
        fs = src.get("fs") or detect_sampling_rate(
            find_output(lfp_dir, "lfp_timestamps.npy"))
        if fs and fs > LFP_FS_MAX:
            print(f"  warning: the LFP timebase implies {int(fs)} Hz (the raw "
                  f"acquisition rate, not the LFP rate) — using 1500 Hz instead")
            fs = None
        fs = fs or 1500.0

    # Auto-load per-rat cortex/sr tetrodes saved by the tracker (SLEEP_CHANNELS_<rat>)
    # unless the caller passed them explicitly.
    if ctx_channel is None and sr_channel is None:
        from processing import load_sleep_channels
        sc = load_sleep_channels(lfp_dir)
        if sc:
            ctx_channel = sc.get("cortex")
            sr_channel = sc.get("sr")
            print(f"  using SLEEP_CHANNELS: cortex={ctx_channel} sr={sr_channel} "
                  f"pyr={sc.get('pyr')}")

    sw_ch = ctx_channel if ctx_channel is not None else (
        channel if channel is not None else src["channels"][0])
    lfp = load_lfp_channel(src, sw_ch)
    theta_lfp = load_lfp_channel(src, sr_channel) if sr_channel is not None else None
    if sr_channel is not None:
        print(f"  slow-wave from cortex ch {sw_ch}, theta from SR ch {sr_channel}")

    fitted, model_name = _resolve_model(model, lfp_dir)
    if fitted is not None:
        import fit_auto_score as fa
        t, X = fa.extract_features(lfp, fs, sr_lfp=theta_lfp)
        states = fa.predict(fitted, X)
        if kw.get("min_secs"):
            states = enforce_min_duration(states, min_secs=kw["min_secs"])
        names = [str(s) for s in fitted["features"]]
        print(f"  scored with {model_name} ({', '.join(names)}; "
              f"{len(fitted['codes'])} states)")
        return {"states": states, "timestamps": t, "model": model_name,
                "metrics": dict(zip(names, X.T)), "thresholds": {}}, sw_ch

    emg, emg_ts = _load_emg(lfp_dir, fs)
    motion, motion_ts = (_load_motion(lfp_dir, fs) if use_motion else (None, None))
    src_name = ("motion+EMG" if (motion is not None and emg is not None)
                else "motion" if motion is not None
                else "EMG-from-LFP" if emg is not None
                else f"LFP {int(HF_BAND[0])}-{int(HF_BAND[1])} Hz power")
    print(f"  REM movement gate from: {src_name}")
    return score(lfp, fs, emg=emg, emg_ts=emg_ts, motion=motion, motion_ts=motion_ts,
                 theta_lfp=theta_lfp, **kw), sw_ch


def main():
    ap = argparse.ArgumentParser(
        description="Buzsáki auto sleep scoring (WAKE/NREM/REM) from an LFP_Output "
                    "folder. Writes buzsaki_states.npz for the state-editor panel.")
    ap.add_argument("--lfp_folder", required=True,
                    help="LFP_Output folder (with lfp_data.npy or channels_npy/).")
    ap.add_argument("--channel", type=int, default=None,
                    help="LFP channel to score (default: first available).")
    ap.add_argument("--out", default=None,
                    help=f"Output .npz (default: <lfp_folder>/{DEFAULT_OUT}).")
    ap.add_argument("--min_secs", type=float, default=10.0,
                    help="Minimum state-run duration (s).")
    ap.add_argument("--model", default="none",
                    help="Fitted model to score with: a path, or 'auto' to use one "
                         "found in the folder. Default 'none' = the threshold "
                         "scorer with the published parameters.")
    ap.add_argument("--sw_factor", type=float, default=SW_THRESH_FACTOR,
                    help="Scale the slow-wave (NREM) cutoff; 1.0 = the bimodal dip, "
                         ">1 = less NREM.")
    ap.add_argument("--th_factor", type=float, default=TH_THRESH_FACTOR,
                    help="Scale the theta (REM) cutoff; <1 = more REM.")
    ap.add_argument("--emg_factor", type=float, default=1.0,
                    help="Scale the movement ceiling on REM; >1 = laxer gate.")
    ap.add_argument("--use_motion", action="store_true",
                    help="Add the accelerometer to the REM movement gate.")
    args = ap.parse_args()

    res, ch = score_from_lfp_output(args.lfp_folder, channel=args.channel,
                                    min_secs=args.min_secs,
                                    model=None if args.model == "none" else args.model,
                                    sw_factor=args.sw_factor,
                                    th_factor=args.th_factor,
                                    emg_factor=args.emg_factor,
                                    use_motion=args.use_motion)
    if args.out:
        out = args.out
    else:
        from processing import output_prefix
        pfx = output_prefix(args.lfp_folder)          # match the session's naming
        out = str(Path(args.lfp_folder) / f"{pfx}{DEFAULT_OUT}")
    save(res, out)

    st = res["states"]
    total = st.size or 1
    names = {WAKE: "WAKE", NREM: "NREM", 4: "INTER", REM: "REM"}
    print(f"Scored channel {ch}: {st.size} bins")
    for code, name in names.items():
        n = np.count_nonzero(st == code)
        if n:
            print(f"  {name:5}: {100.0 * n / total:5.1f}%")
    thr = res["thresholds"]
    if thr:
        print(f"  thresholds: SW={thr['swthresh']:.3f} theta={thr['ththresh']:.3f} "
              f"EMG={thr['emgthresh']}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
