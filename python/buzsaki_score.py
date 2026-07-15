"""Buzsáki automatic sleep scoring (WAKE / NREM / REM) from LFP + EMG.

A Python port of the Buzsáki lab ``SleepScoreMaster`` / ``ClusterStates`` pipeline
(Watson et al. 2016), using the power-spectrum-slope (PSS) metrics from
``anjal_sleepscore`` (``compute_delta_/compute_theta_buzsakiMethod.m``) and the
EMG-from-LFP signal produced by the tracker's step 8.

Three metrics, one value per 1 s bin:
  * broadbandSlowWave = log delta-band (0.5–4 Hz) power (NREM has strong delta,
    so higher = more NREM; drops in theta-dominated REM and desynchronised WAKE).
  * thratio           = peak oscillatory residual over 5–10 Hz (theta).
  * EMG               = EMG-from-LFP (muscle tone).

Each is smoothed, 0–1 normalised, thresholded at its bimodal-histogram dip, then
classified by an EMG-first decision tree (movement splits wake/sleep first, then
the slow-wave/theta metrics split sleep into NREM/REM):
  WAKE  = EMG > emgthresh                         (movement)
  REM   = ~WAKE & theta > ththresh & SW <= swthresh
  NREM  = ~WAKE & ~REM                            (all other sleep)
(without EMG it falls back to slow-wave-first: NREM = SW > swthresh.)

States use the HM codes: 1 = WAKE, 3 = NREM, 5 = REM (0 = unscored).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import spectrogram
from scipy.ndimage import uniform_filter1d

# HM state codes (match TheStateEditor / the Python state editor).
WAKE, LIGHT, NREM, INTER, REM = 1, 2, 3, 4, 5

DELTA_BAND = (0.5, 4.0)      # slow-wave (delta) power band (Hz)
TH_FRANGE = (2.0, 20.0)      # theta 1/f fit range (Hz)
TH_BAND = (5.0, 10.0)        # theta band (Hz)
WINDOW_S = 2.0              # spectrogram window (s)
DT_S = 1.0                 # spectrogram step / bin size (s)
SMOOTH_S = 15.0            # metric smoothing window (s)
TH_THRESH_FACTOR = 0.85    # lower the theta threshold (<1) to accept more REM
DROWSY_FRAC = 0.7          # movement in [DROWSY_FRAC*wake_thr, wake_thr) -> drowsy


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


def _loglog_fit(logf, logP):
    """Per-time-bin OLS line fit of logP (F×T) vs logf (F). Returns slope, resid.

    slope[t], and resid[F, t] = logP - (intercept + slope*logf). Vectorised over t.
    """
    x = logf - logf.mean()
    denom = np.sum(x ** 2)
    ybar = logP.mean(axis=0, keepdims=True)
    slope = (x[:, None] * (logP - ybar)).sum(axis=0) / denom
    intercept = ybar[0] - slope * logf.mean()
    fit = intercept[None, :] + slope[None, :] * logf[:, None]
    return slope, logP - fit


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


def _sw_metric(lfp, fs):
    """broadbandSlowWave (0–1 per 1 s bin) = log delta-band (0.5–4 Hz) power.

    Delta-only rather than the 4–90 Hz power-spectrum slope: the slope band
    includes theta, so REM theta inflated it and REM epochs stayed above the NREM
    threshold (never dropping into the low-SW cluster). Delta power drops in REM
    (theta-dominated) and in WAKE (desynchronised), so it cleanly separates NREM.
    Cleanest from a CORTEX channel. Returns (times, sw, fs_bins)."""
    f, t, Sxx = _spectrogram(lfp, fs)
    fs_bins = 1.0 / (np.median(np.diff(t)) if t.size > 1 else DT_S)
    m = (f >= DELTA_BAND[0]) & (f <= DELTA_BAND[1])
    delta = np.log10(Sxx[m, :].mean(axis=0) + 1e-12)
    return t, _norm01(_smooth(delta, fs_bins)), fs_bins


def _theta_metric(lfp, fs):
    """thratio (0–1 per 1 s bin) = peak oscillatory residual over 5–10 Hz.
    Strongest from a STRATUM RADIATUM channel (theta peaks there in REM)."""
    f, t, Sxx = _spectrogram(lfp, fs)
    logf = np.log10(f + 1e-12)
    logP = np.log10(Sxx + 1e-12)
    fs_bins = 1.0 / (np.median(np.diff(t)) if t.size > 1 else DT_S)
    m = (f >= TH_FRANGE[0]) & (f <= TH_FRANGE[1])
    _, resid = _loglog_fit(logf[m], logP[m, :])
    fth = f[m]
    tband = (fth >= TH_BAND[0]) & (fth <= TH_BAND[1])
    thratio = np.clip(resid[tband, :], 0, None).max(axis=0)
    return t, _norm01(_smooth(thratio, fs_bins))


def compute_metrics(lfp, fs, emg=None, emg_ts=None, theta_lfp=None):
    """Compute (times, broadbandSlowWave, thratio, emg_aligned), all 0–1 per 1 s bin.

    ``lfp`` provides the slow-wave metric — use a **cortex** channel. ``theta_lfp``
    (optional) provides the theta metric — use a **stratum radiatum** channel where
    theta peaks; if omitted, theta is taken from ``lfp`` too (single-channel mode).
    ``emg`` + ``emg_ts`` is the EMG-from-LFP; if not given, REM/WAKE fall back to
    theta only.
    """
    t, sw, _ = _sw_metric(lfp, fs)
    if theta_lfp is None:
        thratio = _theta_metric(lfp, fs)[1]
    else:
        t_th, thratio = _theta_metric(theta_lfp, fs)
        # align theta bins onto the SW time base if the two differ in length
        if thratio.size != t.size:
            thratio = np.interp(t, t_th, thratio)

    emg_aligned = None
    if emg is not None:
        emg = np.asarray(emg, dtype=np.float64).ravel()
        if emg_ts is None:
            emg_ts = np.linspace(t[0], t[-1], emg.size)
        emg_aligned = _norm01(np.interp(t, np.asarray(emg_ts).ravel(), emg))

    return t, sw, thratio, emg_aligned


# ---------------------------------------------------------------------------- #
#  Bimodal threshold + clustering
# ---------------------------------------------------------------------------- #
def bimodal_threshold(x, nbins=60, default=0.5):
    """Threshold at the histogram dip between the two largest modes.

    Mirrors the intent of ``bz_BimodalThresh``: smooth the histogram, find the two
    tallest peaks, and put the threshold at the lowest trough between them. Falls
    back to ``default`` (on the 0–1 metric) when the distribution isn't bimodal.
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
    peaks.sort(key=lambda i: h[i], reverse=True)
    p1, p2 = sorted(peaks[:2])
    trough = p1 + int(np.argmin(h[p1:p2 + 1]))
    return float(centres[trough])


def cluster_states(sw, thratio, emg, motion=None, swthresh=None, ththresh=None,
                   emgthresh=None, sw_factor=1.0, th_factor=TH_THRESH_FACTOR,
                   emg_factor=1.0, drowsy_frac=DROWSY_FRAC, sticky=True):
    """Classify each 1 s bin into WAKE/NREM/INTER/REM by a movement-first tree.

    Movement (accelerometer ``motion`` when given, else EMG-from-LFP ``emg``)
    is 3-level: high -> WAKE, moderate -> LIGHT (drowsy), quiescent -> SLEEP.
    Within SLEEP the slow-wave/theta metrics split it:
      * REM   = theta, no slow waves
      * INTER = theta AND slow waves (intermediate / transition sleep)
      * NREM  = slow waves / quiescent (no theta)

    Each auto (bimodal) threshold is scaled by its ``*_factor`` (1.0 = auto;
    <1 = more permissive). ``drowsy_frac`` sets the moderate-movement band width.
    Returns ``(states, thresholds)``.
    """
    n = len(sw)
    swt = bimodal_threshold(sw) * sw_factor if swthresh is None else swthresh
    # th_factor (<1) lowers the theta threshold so more borderline bins are REM
    tht = bimodal_threshold(thratio) * th_factor if ththresh is None else ththresh
    highsw = sw > swt
    hightheta = thratio > tht

    states = np.zeros(n, dtype=int)
    # Three-level movement: WAKE = high (EITHER EMG or motion in its own high mode,
    # so high-EMG can't be masked); LIGHT/DROWSY = moderate combined movement;
    # SLEEP = quiescent. Within SLEEP the (slow-wave × theta) 2×2 gives NREM/REM/INTER.
    movesigs = [np.asarray(s, dtype=np.float64) for s in (emg, motion) if s is not None]
    if movesigs:
        thrs = [bimodal_threshold(s) * emg_factor if emgthresh is None else emgthresh
                for s in movesigs]
        highmotion = np.zeros(n, dtype=bool)   # WAKE: any signal at wake level
        somemove = np.zeros(n, dtype=bool)     # any signal in the drowsy band+
        for s, thv in zip(movesigs, thrs):
            highmotion |= s > thv
            somemove |= s > drowsy_frac * thv
        movt = thrs[0]
        drowsy = somemove & (~highmotion)             # moderate movement -> drowsy
        sleep = ~somemove                             # quiescent -> asleep
        rem = sleep & hightheta & (~highsw)           # theta, no slow waves
        inter = sleep & hightheta & highsw            # theta + slow waves
        nrem = sleep & (~hightheta)                   # slow-wave / quiescent NREM
        states[highmotion] = WAKE
        states[drowsy] = LIGHT
        states[nrem] = NREM
        states[inter] = INTER
        states[rem] = REM
    else:
        # no movement signal at all: fall back to a slow-wave/theta 2×2
        movt = None
        states[highsw & ~hightheta] = NREM
        states[highsw & hightheta] = INTER
        states[~highsw & hightheta] = REM
        states[~highsw & ~hightheta] = WAKE
    return states, {"swthresh": swt, "ththresh": tht, "emgthresh": movt}


# ---------------------------------------------------------------------------- #
#  Minimum-duration smoothing
# ---------------------------------------------------------------------------- #
def _runs(states):
    """Yield (start, end_exclusive, value) for each contiguous run."""
    if len(states) == 0:
        return
    start = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            yield start, i, states[start]
            start = i


def enforce_min_duration(states, min_secs=6, dt=1.0):
    """Remove state runs shorter than ``min_secs`` by merging into the previous run.

    A light-weight stand-in for the Buzsáki min-window rules: short blips are
    absorbed into their preceding state, iterated until stable.
    """
    states = np.asarray(states, dtype=int).copy()
    min_bins = max(1, int(round(min_secs / dt)))
    changed = True
    while changed:
        changed = False
        for s, e, v in list(_runs(states)):
            if v != 0 and (e - s) < min_bins:
                fill = states[s - 1] if s > 0 else (states[e] if e < len(states) else v)
                states[s:e] = fill
                changed = True
                break
    return states


# ---------------------------------------------------------------------------- #
#  Full pipeline
# ---------------------------------------------------------------------------- #
def score(lfp, fs, emg=None, emg_ts=None, motion=None, motion_ts=None, min_secs=10,
          swthresh=None, ththresh=None, emgthresh=None, sw_factor=1.0,
          th_factor=TH_THRESH_FACTOR, emg_factor=1.0, drowsy_frac=DROWSY_FRAC,
          theta_lfp=None):
    """Full Buzsáki auto-scoring. Returns a dict with states, timestamps, metrics.

    ``lfp`` = slow-wave (cortex) channel; ``theta_lfp`` = optional theta
    (stratum radiatum) channel — pass both for layer-specific scoring. ``motion``
    (accelerometer) drives the wake/sleep split when given, else ``emg`` does.
    ``sw_factor`` / ``th_factor`` / ``emg_factor`` scale the auto thresholds
    (1.0 = auto), ``drowsy_frac`` sets the drowsy band, ``min_secs`` the minimum
    epoch. ``states`` is one HM code per 1 s bin; ``timestamps`` bin centres (s).
    """
    t, sw, thratio, emg_a = compute_metrics(lfp, fs, emg=emg, emg_ts=emg_ts,
                                            theta_lfp=theta_lfp)
    motion_a = None
    if motion is not None:
        motion = np.asarray(motion, dtype=np.float64).ravel()
        if motion_ts is None:
            motion_ts = np.linspace(t[0], t[-1], motion.size)
        motion_a = _norm01(np.interp(t, np.asarray(motion_ts).ravel(), motion))
    states, thr = cluster_states(sw, thratio, emg_a, motion=motion_a,
                                 swthresh=swthresh, ththresh=ththresh,
                                 emgthresh=emgthresh, sw_factor=sw_factor,
                                 th_factor=th_factor, emg_factor=emg_factor,
                                 drowsy_frac=drowsy_frac)
    dt = np.median(np.diff(t)) if t.size > 1 else DT_S
    if min_secs:
        states = enforce_min_duration(states, min_secs=min_secs, dt=dt)
    return {
        "states": states,
        "timestamps": t,
        "metrics": {"broadbandSlowWave": sw, "thratio": thratio, "emg": emg_a,
                    "motion": motion_a},
        "thresholds": thr,
    }


def save(result, path):
    """Save an auto-scoring result to ``path`` (.npz), readable by the GUI panel."""
    np.savez(path,
             states=result["states"].astype(np.int16),
             timestamps=result["timestamps"].astype(np.float64),
             broadbandSlowWave=result["metrics"]["broadbandSlowWave"],
             thratio=result["metrics"]["thratio"],
             emg=(result["metrics"]["emg"] if result["metrics"]["emg"] is not None
                  else np.array([])),
             swthresh=result["thresholds"]["swthresh"],
             ththresh=result["thresholds"]["ththresh"],
             emgthresh=(result["thresholds"]["emgthresh"]
                        if result["thresholds"]["emgthresh"] is not None else np.nan))
    return path


def load_states(path):
    """Load a saved auto-scoring result: returns (states, timestamps)."""
    d = np.load(path, allow_pickle=False)
    return np.asarray(d["states"]).ravel().astype(int), np.asarray(d["timestamps"]).ravel()


DEFAULT_OUT = "buzsaki_states.npz"


def _load_emg(lfp_dir, fs):
    """Load the EMG-from-LFP signal from an LFP_Output folder, else (None, None)."""
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
    """Load the accelerometer motion magnitude (decimated) + timestamps, else
    (None, None). Used for the wake/sleep split in preference to EMG-from-LFP."""
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


def score_from_lfp_output(lfp_dir, channel=None, ctx_channel=None,
                          sr_channel=None, **kw):
    """Run the pipeline on an LFP_Output folder. Returns (result, channel_used).

    Layer-specific channels (recommended, per-rat): ``ctx_channel`` (cortex) drives
    the slow-wave/NREM metric, ``sr_channel`` (stratum radiatum) drives the
    theta/REM metric. If only ``channel`` (or none) is given, a single channel
    drives both (legacy behaviour). Channel numbers are 1-based tetrode numbers
    (channels_npy) or 1-based columns (lfp_data.npy), per find_lfp_source.
    """
    from processing import (find_lfp_source, load_lfp_channel,
                            detect_sampling_rate, find_output)

    lfp_dir = Path(lfp_dir)
    src = find_lfp_source(str(lfp_dir))
    if src is None:
        raise FileNotFoundError(f"no lfp_data.npy or channels_npy/ in {lfp_dir}")
    fs = detect_sampling_rate(find_output(lfp_dir, "lfp_timestamps.npy")) or 1500.0

    # Auto-load per-rat cortex/sr tetrodes saved by the tracker (SLEEP_CHANNELS_<rat>)
    # unless the caller passed them explicitly.
    if ctx_channel is None and sr_channel is None:
        scf = find_output(lfp_dir, "sleep_channels.npy")
        if scf is not None:
            sc = np.load(scf, allow_pickle=True).item()
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

    emg, emg_ts = _load_emg(lfp_dir, fs)
    motion, motion_ts = _load_motion(lfp_dir, fs)
    src_name = ("motion+EMG" if (motion is not None and emg is not None)
                else "motion" if motion is not None
                else "EMG" if emg is not None
                else "slow-wave only (no EMG/motion)")
    print(f"  wake/sleep split from: {src_name}")
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
    args = ap.parse_args()

    res, ch = score_from_lfp_output(args.lfp_folder, channel=args.channel,
                                    min_secs=args.min_secs)
    if args.out:
        out = args.out
    else:
        from processing import output_prefix
        pfx = output_prefix(args.lfp_folder)          # match the session's naming
        out = str(Path(args.lfp_folder) / f"{pfx}{DEFAULT_OUT}")
    save(res, out)

    st = res["states"]
    total = st.size or 1
    names = {WAKE: "WAKE", LIGHT: "LIGHT", NREM: "NREM", INTER: "INTER", REM: "REM"}
    print(f"Scored channel {ch}: {st.size} bins")
    for code, name in names.items():
        pct = 100.0 * np.count_nonzero(st == code) / total
        print(f"  {name:5}: {pct:5.1f}%")
    thr = res["thresholds"]
    print(f"  thresholds: SW={thr['swthresh']:.3f} theta={thr['ththresh']:.3f} "
          f"EMG={thr['emgthresh']}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
