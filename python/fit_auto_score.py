"""Fit the automatic scorer to manually scored sessions.

The bimodal-threshold scorer in :mod:`buzsaki_score` needs no training, but its
cutoffs are only as good as the histogram dips it finds — on our recordings the
slow-wave dip sits low enough to swallow REM into NREM, and it has no notion of
intermediate sleep at all. Given one or more hand-scored sessions this module
instead *fits* the state boundaries: a Gaussian (shared-covariance) emission
model per state plus the state-transition matrix taken from the scorer's own
labels, decoded with Viterbi so the hypnogram follows the transition statistics
the scorer actually produced.

Features are computed from the LFP alone — no accelerometer, no EMG file — so a
model fitted on one session runs on any session, including ones with no motion
recorded. Eight per-second features, each smoothed (15 s), then z-scored within
the session so electrode gain cancels:

    sw          z(log delta 0.5-4) - z(log gamma 40-100)   cortex   NREM
    thdelta     log theta 5-10 - log delta 0.5-4           cortex   REM
    hf          log power 275-500 Hz                       cortex   WAKE / muscle
    ripple      log 100-200 Hz minus log 1-100 Hz          SR       WAKE, arousals
    spindle     log 10-16 Hz minus log 1-100 Hz            SR       intermediate
    spindle_hi  log power 13-18 Hz                         SR       intermediate vs REM
    sr_thdelta  log theta 5-10 - log delta 0.5-4           SR       REM
    amp_cv      CV of the 1-30 Hz envelope within a second cortex   REM is steady

``hf`` and ``ripple`` are muscle tone and arousal read off the LFP's own
high-frequency content (the EMG-from-LFP idea): ``hf`` separates WAKE from sleep
as well as the recorded EMG (AUC 0.96) and better than the accelerometer (0.89),
and needs no extra file. ``ripple`` adds the 100-200 Hz band of the SR channel,
whose wake-to-sleep contrast is the widest of any band we measured (1.3 log
units, against 0.8 on cortex); it is what gets the arousal *after REM* right,
taking that recall from 0.53 to 0.64 and the share of REM epochs ending in WAKE
from 45% to 96% (with :func:`buzsaki_score.post_rem_arousal`).

Instantaneous high-frequency power was tried too — Hilbert envelope of those
bands, then per-second peak or burst fraction. The burst fraction ranks brief
arousals distinctly better on its own (AUC 0.91 against NREM, vs 0.83 for the
spectrogram average), but it does not improve the fitted model: a bounded
fraction has a point mass at zero in every sleep state, which a Gaussian
emission fits badly, and log-transforming it or applying it as a post-decode
arousal override both came out neutral-to-worse. The arousals the model misses
are not high-burst seconds — they are genuinely weak ones.

The two spindle features are what find **intermediate sleep**. On the stratum
radiatum channel, relative to broadband, intermediate sits ~8 dB above both
NREM and REM right across 9-13 Hz (AUC 0.97 against everything else). The
lower slice detects it; the higher 13-18 Hz slice is what keeps it distinct
from REM, which it otherwise runs into (AUC 0.88 vs 0.87). Intermediate is also
decoded with a duration cap (``INTER_MAX_S``) and a small penalty
(``INTER_BIAS``), because it is a brief transition — it never lasted beyond 46 s
in our hand scoring — and without those the decoder stays in it through the REM
epoch that follows.

Usage::

    python fit_auto_score.py --lfp_folder <LFP_Output> --labels <scoring.npz>
    python fit_auto_score.py --lfp_folder A --labels a.npz \
                             --lfp_folder B --labels b.mat --out model.npz

Intermediate is fitted whenever the labels contain enough of it
(``MIN_INTER_BINS``); ``--no_intermediate`` fits WAKE/NREM/REM only. Writes
``sleep_score_model.npz`` next to the first LFP folder (or ``--out``), which
:mod:`buzsaki_score` picks up automatically on its next run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

import buzsaki_score as bz

WAKE, NREM, INTER, REM = 1, 3, 4, 5
STATE_NAMES = {WAKE: "WAKE", NREM: "NREM", INTER: "INTER", REM: "REM"}

FEATURES = ("sw", "thdelta", "hf", "ripple", "spindle", "spindle_hi",
            "sr_thdelta", "amp_cv")
SMOOTH_S = 15.0
SPINDLE_BAND = (10.0, 16.0)      # sleep spindles: the intermediate-state marker
SPINDLE_HI_BAND = (13.0, 18.0)   # upper spindle slice — separates INTER from REM
BROADBAND = (1.0, 100.0)         # reference band for the gain-free spindle ratio
RIPPLE_BAND = (100.0, 200.0)     # high-frequency band that rises on arousal
AMP_BAND = (1.0, 30.0)           # band whose envelope steadiness marks REM
AMP_BLOCK_BINS = 1800            # envelope computed in 30 min blocks (flat memory)
DEFAULT_MODEL = "sleep_score_model.npz"

# Intermediate sleep is a brief NREM->REM transition (10-46 s in our hand
# scoring, against 33-141 s REM epochs). Left to a plain HMM the decoder enters
# it at REM onset and never leaves, eating the REM epoch; capping its duration
# and penalising it slightly costs a little INTER recall and buys back REM.
INTER_MAX_S = 30.0
INTER_BIAS = 2.0
MIN_INTER_BINS = 60              # below this, don't try to fit intermediate


# ---------------------------------------------------------------------------- #
#  Features
# ---------------------------------------------------------------------------- #
def amplitude_cv(lfp, fs, n_bins):
    """Within-second amplitude variability: CV of the ``AMP_BAND`` envelope.

    REM is the steadiest state there is — a regular theta rhythm holds its
    amplitude — while intermediate sleep fluctuates, its spindles arriving in
    bursts (mean CV 0.40 REM vs 0.47 intermediate, 0.52 NREM; AUC 0.88 for
    intermediate against REM). It is what lets the model call REM *REM* rather
    than just "theta and not NREM": adding it lifts REM precision from 0.81
    to 0.90.

    The Hilbert envelope is taken in ``AMP_BLOCK_BINS``-second blocks, each with
    a second of context either side, so memory stays flat on a long recording.
    """
    from scipy.signal import butter, sosfiltfilt, hilbert

    step = int(round(fs))
    nyq = fs / 2
    sos = butter(4, [AMP_BAND[0] / nyq, min(AMP_BAND[1], 0.9 * nyq) / nyq],
                 btype="band", output="sos")
    lfp = np.asarray(lfp)
    usable = min(int(n_bins), lfp.size // step)
    out = np.zeros(int(n_bins), dtype=np.float64)
    for s0 in range(0, usable, AMP_BLOCK_BINS):
        s1 = min(usable, s0 + AMP_BLOCK_BINS)
        pad0 = step if s0 > 0 else 0
        pad1 = step if s1 < usable else 0
        seg = np.asarray(lfp[s0 * step - pad0:s1 * step + pad1], dtype=np.float64)
        env = np.abs(hilbert(sosfiltfilt(sos, seg).astype(np.float32)))
        e = env[pad0:pad0 + (s1 - s0) * step].reshape(s1 - s0, step).astype(np.float64)
        out[s0:s1] = e.std(axis=1) / (e.mean(axis=1) + 1e-12)
    if usable < n_bins:                      # ragged tail: hold the last value
        out[usable:] = out[usable - 1] if usable else 0.0
    return out


def extract_features(lfp, fs, sr_lfp=None):
    """Per-1 s-bin feature matrix. Returns ``(times, X)``, columns = ``FEATURES``.

    ``lfp`` is the cortex channel; ``sr_lfp`` the stratum radiatum channel,
    which supplies the spindle and SR theta features. Without it they come from
    the cortex channel too — workable, but intermediate sleep is much harder to
    see there (spindle AUC 0.78 against 0.97).
    """
    def channel(sig):
        f, t, Sxx = bz._spectrogram(sig, fs)
        logP = np.log10(Sxx + 1e-12)

        def band(lo, hi):
            return logP[(f >= lo) & (f <= hi), :].mean(axis=0)
        return t, band

    t, ctx = channel(lfp)
    hf_hi = min(bz.HF_BAND[1], 0.95 * fs / 2)          # stay under Nyquist
    delta = ctx(*bz.DELTA_BAND)
    raw = {
        "sw": bz._zscore(delta) - bz._zscore(ctx(*bz.GAMMA_BAND)),
        "thdelta": ctx(*bz.TH_BAND) - delta,
        "hf": ctx(min(bz.HF_BAND[0], 0.6 * hf_hi), hf_hi),
    }

    t_sr, sr = (channel(sr_lfp) if sr_lfp is not None else (t, ctx))
    sr_delta = sr(*bz.DELTA_BAND)
    sr_broad = sr(*BROADBAND)
    sr_feats = {
        "ripple": sr(*RIPPLE_BAND) - sr_broad,
        "spindle": sr(*SPINDLE_BAND) - sr_broad,
        "spindle_hi": sr(*SPINDLE_HI_BAND),
        "sr_thdelta": sr(*bz.TH_BAND) - sr_delta,
    }
    for name, values in sr_feats.items():        # align if the two channels differ
        raw[name] = (values if values.size == t.size
                     else np.interp(t, t_sr, values))

    raw["amp_cv"] = amplitude_cv(lfp, fs, t.size)

    fs_bins = 1.0 / (np.median(np.diff(t)) if t.size > 1 else bz.DT_S)
    win = max(1, int(round(SMOOTH_S * fs_bins)))
    X = np.column_stack([bz._zscore(uniform_filter1d(raw[k], win, mode="nearest"))
                         for k in FEATURES])
    return t, X


def features_from_lfp_output(lfp_dir, fs=None, channel=None, sr_channel=None):
    """Feature matrix for an LFP_Output folder. Returns ``(times, X, channels)``.

    Channels default to the cortex and stratum radiatum tetrodes in
    ``sleep_channels.npy``; without that file one channel supplies everything.
    """
    from processing import (find_lfp_source, load_lfp_channel,
                            detect_sampling_rate, find_output, LFP_FS_MAX)

    lfp_dir = Path(lfp_dir)
    src = find_lfp_source(str(lfp_dir))
    if src is None:
        raise FileNotFoundError(f"no lfp_data.npy or channels_npy/ in {lfp_dir}")
    if fs is None:
        fs = detect_sampling_rate(find_output(lfp_dir, "lfp_timestamps.npy"))
        fs = None if (fs and fs > LFP_FS_MAX) else fs
        fs = fs or 1500.0
    if channel is None or sr_channel is None:
        scf = find_output(lfp_dir, "sleep_channels.npy")
        sc = np.load(scf, allow_pickle=True).item() if scf is not None else {}
        channel = channel if channel is not None else sc.get("cortex")
        sr_channel = sr_channel if sr_channel is not None else sc.get("sr")
    if channel is None:
        channel = src["channels"][0]
    sr_lfp = load_lfp_channel(src, sr_channel) if sr_channel is not None else None
    t, X = extract_features(load_lfp_channel(src, channel), fs, sr_lfp=sr_lfp)
    return t, X, (channel, sr_channel)


# ---------------------------------------------------------------------------- #
#  Manual labels
# ---------------------------------------------------------------------------- #
def load_manual(path):
    """Load a hand scoring (.npz from the editor, or MATLAB -states.mat).

    Returns ``(states, timestamps)``; timestamps are bin starts in seconds.
    """
    path = Path(path)
    if path.suffix.lower() == ".mat":
        from scipy.io import loadmat
        m = loadmat(str(path))
        states = np.asarray(m["states"]).ravel().astype(int)
        ts = np.arange(states.size, dtype=float)
    else:
        d = np.load(str(path), allow_pickle=True)
        states = np.asarray(d["states"]).ravel().astype(int)
        ts = (np.asarray(d["timestamps"]).ravel().astype(float)
              if "timestamps" in d.files else np.arange(states.size, dtype=float))
    return states, ts


def align_labels(states, label_ts, times):
    """Sample a manual scoring onto the feature time base (nearest bin)."""
    idx = np.clip(np.searchsorted(np.asarray(label_ts, float), times),
                  0, len(states) - 1)
    return np.asarray(states, int)[idx]


# ---------------------------------------------------------------------------- #
#  Model
# ---------------------------------------------------------------------------- #
def fit(X, y, codes=(WAKE, NREM, REM), shared_cov=True,
        inter_bias=INTER_BIAS, inter_max_s=INTER_MAX_S):
    """Fit Gaussian emissions + transition matrix for the states in ``codes``.

    ``inter_bias`` / ``inter_max_s`` are stored with the model and applied by
    :func:`predict`; they only matter when ``INTER`` is among ``codes``.
    """
    codes = tuple(codes)
    mus, covs, priors = [], [], []
    for c in codes:
        Xi = X[y == c]
        if len(Xi) < X.shape[1] + 2:
            raise ValueError(f"state {STATE_NAMES.get(c, c)} has only {len(Xi)} "
                             f"labelled bins — too few to fit")
        mus.append(Xi.mean(axis=0))
        covs.append(np.cov(Xi.T) + 1e-6 * np.eye(X.shape[1]))
        priors.append(len(Xi) / len(X))
    if shared_cov:
        covs = [sum(p * C for p, C in zip(priors, covs))] * len(codes)

    k = len(codes)
    idx = {c: i for i, c in enumerate(codes)}
    counts = np.full((k, k), 1e-3)
    for a, b in zip(y[:-1], y[1:]):
        if a in idx and b in idx:
            counts[idx[a], idx[b]] += 1
    return {
        "codes": np.array(codes, dtype=int),
        "features": np.array(FEATURES),
        "means": np.array(mus),
        "covs": np.array(covs),
        "priors": np.array(priors),
        "logtrans": np.log(counts / counts.sum(axis=1, keepdims=True)),
        "inter_bias": np.array(float(inter_bias)),
        "inter_max_s": np.array(float(inter_max_s)),
    }


def _loglik(model, X):
    out = np.empty((len(X), len(model["codes"])))
    for i, (mu, C) in enumerate(zip(model["means"], model["covs"])):
        L = np.linalg.cholesky(C)
        s = np.linalg.solve(L, (X - mu).T)
        out[:, i] = -0.5 * (s ** 2).sum(axis=0) - np.log(np.diag(L)).sum()
    return out


def _viterbi(ll, logtrans, logprior):
    n, k = ll.shape
    dp = logprior + ll[0]
    ptr = np.zeros((n, k), dtype=int)
    for i in range(1, n):
        m = dp[:, None] + logtrans
        ptr[i] = np.argmax(m, axis=0)
        dp = m[ptr[i], np.arange(k)] + ll[i]
    path = np.empty(n, dtype=int)
    path[-1] = int(np.argmax(dp))
    for i in range(n - 1, 0, -1):
        path[i - 1] = ptr[i, path[i]]
    return path


def cap_runs(states, state, max_bins):
    """Trim runs of ``state`` longer than ``max_bins``; the excess takes the
    label of whatever follows (by construction REM or WAKE, for intermediate)."""
    states = np.asarray(states, dtype=int).copy()
    if max_bins <= 0 or states.size == 0:
        return states
    edges = np.flatnonzero(np.diff(states)) + 1
    for a, b in zip(np.r_[0, edges], np.r_[edges, states.size]):
        if states[a] == state and (b - a) > max_bins:
            states[a + max_bins:b] = states[b] if b < states.size else states[a]
    return states


def predict(model, X, smooth=True, dt=1.0):
    """States (HM codes) for a feature matrix. ``smooth`` = Viterbi decoding.

    When the model includes intermediate sleep, its stored penalty and duration
    cap are applied as well — see ``INTER_MAX_S`` / ``INTER_BIAS``.
    """
    codes = list(np.asarray(model["codes"], int))
    ll = _loglik(model, X)
    bias = float(np.asarray(model.get("inter_bias", 0.0)))
    if INTER in codes and bias:
        ll[:, codes.index(INTER)] -= bias
    logprior = np.log(np.asarray(model["priors"], float))
    if smooth:
        path = _viterbi(ll, np.asarray(model["logtrans"], float), logprior)
    else:
        path = np.argmax(ll + logprior, axis=1)
    states = np.asarray(codes, int)[path]
    max_s = float(np.asarray(model.get("inter_max_s", 0.0)))
    if INTER in codes and max_s > 0:
        states = cap_runs(states, INTER, int(round(max_s / dt)))
    return states


def save_model(model, path):
    np.savez(str(path), **model)
    return str(path)


def load_model(path):
    d = np.load(str(path), allow_pickle=False)
    return {k: d[k] for k in d.files}


def find_model(*folders):
    """First ``sleep_score_model.npz`` found in ``folders`` (prefixed or not)."""
    from processing import find_output
    for folder in folders:
        if not folder:
            continue
        hit = find_output(folder, DEFAULT_MODEL)
        if hit is not None:
            return hit
    return None


# ---------------------------------------------------------------------------- #
#  Scoring / validation
# ---------------------------------------------------------------------------- #
def confusion(pred, truth, codes):
    """Rows = manual state, columns = predicted, restricted to ``codes``."""
    codes = list(codes)
    M = np.zeros((len(codes), len(codes)), dtype=int)
    for i, a in enumerate(codes):
        for j, b in enumerate(codes):
            M[i, j] = int(np.sum((truth == a) & (pred == b)))
    return M


def agreement(pred, truth, codes=(WAKE, NREM, REM)):
    """Accuracy, balanced accuracy, Cohen's kappa and per-state recall."""
    m = np.isin(truth, list(codes))
    if not m.any():
        return {}
    acc = float(np.mean(pred[m] == truth[m]))
    rec = {c: (float(np.mean(pred[truth == c] == c)) if (truth == c).any() else np.nan)
           for c in codes}
    pe = sum(np.mean(truth[m] == c) * np.mean(pred[m] == c) for c in codes)
    return {"acc": acc, "bacc": float(np.nanmean(list(rec.values()))),
            "kappa": float((acc - pe) / (1 - pe)) if pe < 1 else np.nan,
            "recall": rec, "n": int(m.sum())}


def state_runs(states, code):
    """[(start, end_exclusive)] of every run of ``code``."""
    states = np.asarray(states, int)
    edges = np.flatnonzero(np.diff(states)) + 1
    return [(a, b) for a, b in zip(np.r_[0, edges], np.r_[edges, states.size])
            if states[a] == code]


def rem_exits(states):
    """``(n_ending_in_wake, n_rem_epochs)`` — REM should terminate in an arousal.

    A REM epoch running straight into NREM is not a thing the animal does, so
    this is a quick check that a hypnogram is physiologically shaped, not just
    accurate bin by bin.
    """
    states = np.asarray(states, int)
    runs = state_runs(states, REM)
    to_wake = sum(1 for a, b in runs if b < states.size and states[b] == WAKE)
    return to_wake, len(runs)


def epoch_detection(pred, truth, code=INTER):
    """Epoch-level score for a rare state: ``(found, total, extra, predicted)``.

    Bin-level precision is a poor guide for a state that covers ~1% of the
    recording — one boundary disagreement swamps it. What matters is whether
    each hand-scored epoch was found at all, and how many extra were proposed.
    """
    man, auto = state_runs(truth, code), state_runs(pred, code)
    found = sum(any(x < b and y > a for x, y in auto) for a, b in man)
    extra = sum(not any(x < b and y > a for a, b in man) for x, y in auto)
    return found, len(man), extra, len(auto)


def cross_validate(X, y, codes=(WAKE, NREM, REM), nfold=5, shared_cov=True):
    """Time-blocked CV: fit on all but one contiguous block, test on it.

    Contiguous blocks, not shuffled bins — neighbouring seconds are nearly
    identical, so a shuffled split would score a memorised recording.
    """
    n = len(y)
    folds = np.array_split(np.arange(n), nfold)
    preds = np.zeros(n, dtype=int)
    for f in folds:
        train = np.setdiff1d(np.arange(n), f)
        train = train[np.isin(y[train], list(codes))]
        preds[f] = predict(fit(X[train], y[train], codes, shared_cov), X[f])
    return preds, agreement(preds, y, codes)


def print_report(pred, y, codes, title, min_secs=10.0):
    codes = list(codes)
    if min_secs:
        pred = bz.enforce_min_duration(pred, min_secs=min_secs)
    r = agreement(pred, y, codes)
    M = confusion(pred, y, codes)
    print(f"\n{title}")
    print(f"  accuracy {r['acc']:.3f}   balanced {r['bacc']:.3f}   "
          f"kappa {r['kappa']:.3f}   ({r['n']} bins)")
    head = "  manual\\auto " + "".join(f"{STATE_NAMES[c]:>8s}" for c in codes)
    print(head + "    recall")
    for i, c in enumerate(codes):
        row = "".join(f"{v:8d}" for v in M[i])
        print(f"  {STATE_NAMES[c]:>11s} {row}    {r['recall'][c]:.3f}")
    # predicted-state precision, so over-calling a state is visible too
    prec = ["%s %.3f" % (STATE_NAMES[c], M[i, i] / M[:, i].sum() if M[:, i].sum() else np.nan)
            for i, c in enumerate(codes)]
    print("  precision: " + "  ".join(prec))
    if INTER in codes:
        found, total, extra, n_auto = epoch_detection(pred, y, INTER)
        print(f"  intermediate epochs: found {found} of your {total}, "
              f"{extra} extra ({n_auto} proposed)")
    if REM in codes:
        auto_w, auto_n = rem_exits(pred)
        man_w, man_n = rem_exits(y)
        print(f"  REM epochs ending in WAKE: {auto_w} of {auto_n} "
              f"(yours {man_w} of {man_n})")
    return pred


# ---------------------------------------------------------------------------- #
#  CLI
# ---------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Fit the automatic sleep scorer to hand-scored sessions. "
                    "Pass --lfp_folder/--labels once per session.")
    ap.add_argument("--lfp_folder", action="append", required=True,
                    help="LFP_Output folder of a hand-scored session.")
    ap.add_argument("--labels", action="append", required=True,
                    help="Its manual scoring (.npz from the editor, or -states.mat).")
    ap.add_argument("--channel", type=int, default=None,
                    help="Cortex channel (default: sleep_channels.npy, else the first).")
    ap.add_argument("--sr_channel", type=int, default=None,
                    help="Stratum radiatum channel — supplies the spindle features "
                         "(default: sleep_channels.npy).")
    ap.add_argument("--fs", type=float, default=None, help="LFP sampling rate (Hz).")
    ap.add_argument("--out", default=None,
                    help=f"Model file (default: <first lfp_folder>/<prefix>{DEFAULT_MODEL}).")
    ap.add_argument("--no_intermediate", action="store_true",
                    help="Fit WAKE/NREM/REM only, ignoring intermediate labels.")
    ap.add_argument("--with_intermediate", action="store_true",
                    help="Fit intermediate even with few labelled bins.")
    ap.add_argument("--inter_max_s", type=float, default=INTER_MAX_S,
                    help="Longest intermediate epoch the decoder may produce (s).")
    ap.add_argument("--inter_bias", type=float, default=INTER_BIAS,
                    help="Penalty against calling intermediate (0 = none).")
    ap.add_argument("--nfold", type=int, default=5, help="Time-blocked CV folds.")
    args = ap.parse_args()

    if len(args.lfp_folder) != len(args.labels):
        ap.error("--lfp_folder and --labels must be given the same number of times")

    Xs, ys = [], []
    for folder, labels in zip(args.lfp_folder, args.labels):
        t, X, (ch, sr_ch) = features_from_lfp_output(
            folder, fs=args.fs, channel=args.channel, sr_channel=args.sr_channel)
        states, lts = load_manual(labels)
        y = align_labels(states, lts, t)
        y[y == 2] = WAKE                      # legacy drowsy code folds into WAKE
        counts = {c: int((y == c).sum()) for c in (WAKE, NREM, INTER, REM)}
        print(f"{Path(folder).name}: cortex ch {ch}, SR ch {sr_ch}, {len(t)} bins "
              f"({', '.join(f'{STATE_NAMES[c]} {n}' for c, n in counts.items())})")
        if sr_ch is None:
            print("  note: no SR channel — the spindle features come from the cortex "
                  "channel, where intermediate sleep is much harder to see")
        Xs.append(X)
        ys.append(y)

    X = np.vstack(Xs)
    y = np.concatenate(ys)

    n_inter = int((y == INTER).sum())
    want_inter = (not args.no_intermediate
                  and (args.with_intermediate or n_inter >= MIN_INTER_BINS))
    if not want_inter and n_inter and not args.no_intermediate:
        print(f"  only {n_inter} intermediate bins (< {MIN_INTER_BINS}) — fitting "
              f"3 states; force it with --with_intermediate")
    codes = (WAKE, NREM, INTER, REM) if want_inter else (WAKE, NREM, REM)
    keep = np.isin(y, list(codes))
    X, y = X[keep], y[keep]
    print(f"  fitting {len(codes)} states: "
          f"{', '.join(STATE_NAMES[c] for c in codes)}  ({keep.sum()} bins)")

    fit_kw = dict(inter_bias=args.inter_bias, inter_max_s=args.inter_max_s)
    if args.nfold > 1 and len(Xs) == 1:
        n = len(y)
        preds = np.zeros(n, dtype=int)
        for f in np.array_split(np.arange(n), args.nfold):
            train = np.setdiff1d(np.arange(n), f)
            preds[f] = predict(fit(X[train], y[train], codes, **fit_kw), X[f])
        print_report(preds, y, codes,
                     f"Held-out agreement ({args.nfold} time-blocked folds)")

    model = fit(X, y, codes, **fit_kw)
    print_report(predict(model, X), y, codes, "In-sample agreement (final model)")

    out = args.out
    if out is None:
        from processing import output_prefix
        pfx = output_prefix(args.lfp_folder[0])
        out = str(Path(args.lfp_folder[0]) / f"{pfx}{DEFAULT_MODEL}")
    save_model(model, out)
    print(f"\nSaved {out}\n  buzsaki_score picks this up automatically; "
          f"delete it to fall back to the threshold scorer.")


if __name__ == "__main__":
    main()
