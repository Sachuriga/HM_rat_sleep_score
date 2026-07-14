"""Buzsáki automatic sleep scoring (WAKE / NREM / REM) from LFP + EMG.

A Python port of the Buzsáki lab ``SleepScoreMaster`` / ``ClusterStates`` pipeline
(Watson et al. 2016), using the power-spectrum-slope (PSS) metrics from
``anjal_sleepscore`` (``compute_delta_/compute_theta_buzsakiMethod.m``) and the
EMG-from-LFP signal produced by the tracker's step 8.

Three metrics, one value per 1 s bin:
  * broadbandSlowWave = -slope of the log-log power spectrum over 4–90 Hz
    (NREM steepens the 1/f slope, so higher = more NREM).
  * thratio           = peak oscillatory residual over 5–10 Hz (theta).
  * EMG               = EMG-from-LFP (muscle tone).

Each is smoothed, 0–1 normalised, and split at its bimodal-histogram dip:
  NREM  = SW > swthresh
  REM   = ~NREM & EMG < emgthresh & theta > ththresh
  WAKE  = everything else

States use the HM codes: 1 = WAKE, 3 = NREM, 5 = REM (0 = unscored).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import spectrogram
from scipy.ndimage import uniform_filter1d

# HM state codes (match TheStateEditor / the Python state editor).
WAKE, NREM, REM = 1, 3, 5

SW_FRANGE = (4.0, 90.0)      # slow-wave PSS fit range (Hz)
TH_FRANGE = (2.0, 20.0)      # theta 1/f fit range (Hz)
TH_BAND = (5.0, 10.0)        # theta band (Hz)
WINDOW_S = 2.0              # spectrogram window (s)
DT_S = 1.0                 # spectrogram step / bin size (s)
SMOOTH_S = 15.0            # metric smoothing window (s)


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


def compute_metrics(lfp, fs, emg=None, emg_ts=None):
    """Compute (times, broadbandSlowWave, thratio, emg_aligned), all 0–1 per 1 s bin.

    ``lfp`` is one good LFP channel (a clear cortical/hippocampal channel) at ``fs``.
    ``emg`` + ``emg_ts`` is the EMG-from-LFP signal and its timestamps (s); if not
    given, the EMG metric is None and REM/WAKE fall back to theta only.
    """
    f, t, Sxx = _spectrogram(lfp, fs)
    logf_all = np.log10(f + 1e-12)
    logP = np.log10(Sxx + 1e-12)
    dt = np.median(np.diff(t)) if t.size > 1 else DT_S
    fs_bins = 1.0 / dt

    # slow wave: -slope over 4–90 Hz
    sw_mask = (f >= SW_FRANGE[0]) & (f <= SW_FRANGE[1])
    slope, _ = _loglog_fit(logf_all[sw_mask], logP[sw_mask, :])
    sw = _norm01(_smooth(-slope, fs_bins))

    # theta: oscillatory residual over 5–10 Hz (fit 1/f over 2–20 Hz)
    th_mask = (f >= TH_FRANGE[0]) & (f <= TH_FRANGE[1])
    _, resid = _loglog_fit(logf_all[th_mask], logP[th_mask, :])
    fth = f[th_mask]
    tband = (fth >= TH_BAND[0]) & (fth <= TH_BAND[1])
    thratio = np.clip(resid[tband, :], 0, None).max(axis=0)
    thratio = _norm01(_smooth(thratio, fs_bins))

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


def cluster_states(sw, thratio, emg, swthresh=None, ththresh=None, emgthresh=None,
                   sticky=True):
    """Classify each 1 s bin into WAKE/NREM/REM from the three 0–1 metrics.

    Returns ``(states, thresholds)`` where ``states`` is an int vector of HM codes
    and ``thresholds`` is a dict of the swthresh/ththresh/emgthresh used.
    """
    n = len(sw)
    swt = bimodal_threshold(sw) if swthresh is None else swthresh
    tht = bimodal_threshold(thratio) if ththresh is None else ththresh

    nrem = sw > swt
    hightheta = thratio > tht
    if emg is not None:
        emgt = bimodal_threshold(emg) if emgthresh is None else emgthresh
        highmotion = emg > emgt
    else:
        emgt = None
        highmotion = np.zeros(n, dtype=bool)

    rem = (~nrem) & (~highmotion) & hightheta
    wake = (~nrem) & (~rem)

    states = np.zeros(n, dtype=int)
    states[nrem] = NREM
    states[rem] = REM
    states[wake] = WAKE
    return states, {"swthresh": swt, "ththresh": tht, "emgthresh": emgt}


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
def score(lfp, fs, emg=None, emg_ts=None, min_secs=6,
          swthresh=None, ththresh=None, emgthresh=None):
    """Full Buzsáki auto-scoring. Returns a dict with states, timestamps, metrics.

    ``states`` is one HM code (1/3/5) per 1 s bin; ``timestamps`` are the bin
    centres (s). ``metrics`` holds the 0–1 sw/theta/emg and the thresholds used.
    """
    t, sw, thratio, emg_a = compute_metrics(lfp, fs, emg=emg, emg_ts=emg_ts)
    states, thr = cluster_states(sw, thratio, emg_a, swthresh=swthresh,
                                 ththresh=ththresh, emgthresh=emgthresh)
    dt = np.median(np.diff(t)) if t.size > 1 else DT_S
    if min_secs:
        states = enforce_min_duration(states, min_secs=min_secs, dt=dt)
    return {
        "states": states,
        "timestamps": t,
        "metrics": {"broadbandSlowWave": sw, "thratio": thratio, "emg": emg_a},
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


def score_from_lfp_output(lfp_dir, channel=None, **kw):
    """Run the pipeline on an LFP_Output folder. Returns (result, channel_used).

    Loads one LFP channel (``--channel`` or the first available), the sampling
    rate from ``lfp_timestamps.npy``, and the EMG-from-LFP if present. Supports
    both the ``lfp_data.npy`` and ``channels_npy/`` layouts (via processing.py).
    """
    from processing import (find_lfp_source, load_lfp_channel,
                            detect_sampling_rate, find_output)

    lfp_dir = Path(lfp_dir)
    src = find_lfp_source(str(lfp_dir))
    if src is None:
        raise FileNotFoundError(f"no lfp_data.npy or channels_npy/ in {lfp_dir}")
    fs = detect_sampling_rate(find_output(lfp_dir, "lfp_timestamps.npy")) or 1500.0
    if channel is None:
        channel = src["channels"][0]
    lfp = load_lfp_channel(src, channel)
    emg, emg_ts = _load_emg(lfp_dir, fs)
    if emg is None:
        print("  (no emg_from_lfp found — REM/WAKE split uses theta only)")
    return score(lfp, fs, emg=emg, emg_ts=emg_ts, **kw), channel


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
    ap.add_argument("--min_secs", type=float, default=6.0,
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
    names = {WAKE: "WAKE", NREM: "NREM", REM: "REM"}
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
