"""Signal processing for the Python sleep-scoring toolkit.

Faithful port of the preprocessing and spectrogram pipeline used by
``TheStateEditor.m``:

    raw LFP  ->  MAD artifact clip (+-5 sigma)
             ->  50 Hz notch (zero-phase)
             ->  AR(1) whitening
             ->  multitaper spectrogram (DPSS, NW=3, 5 tapers, nFFT=3072)

The numbers (nFFT, window length, NW, taper count, frequency range, the
MAD threshold, the notch Q) are taken directly from the MATLAB source so the
Python output matches the original as closely as practical.
"""

from __future__ import annotations

import os
import re

import numpy as np
from scipy.signal import iirnotch, filtfilt, lfilter, firwin
from scipy.signal.windows import dpss


def matlab_round(x: float) -> int:
    """MATLAB ``round`` rounds halves away from zero (numpy rounds to even)."""
    return int(np.floor(np.abs(x) + 0.5) * np.sign(x))


def mad_clip(sig: np.ndarray, n_sigma: float = 5.0) -> np.ndarray:
    """Clip large artifacts at +-``n_sigma`` using a robust MAD estimate.

    Mirrors:  sigma = median(|x - median(x)|) / 0.6745;  x = clip(x, +-5 sigma)
    Note the clip is symmetric about zero, exactly as in the MATLAB code.
    """
    sig = np.asarray(sig, dtype=np.float64)
    mad = np.median(np.abs(sig - np.median(sig)))
    sigma = mad / 0.6745
    thresh = n_sigma * sigma
    if thresh == 0:  # flat signal; nothing to clip
        return sig
    return np.clip(sig, -thresh, thresh)


def notch_filter(sig: np.ndarray, fs: float, f0: float = 50.0, q: float = 35.0) -> np.ndarray:
    """Zero-phase 50 Hz notch.

    MATLAB:  wo = 50/(fs/2); bw = wo/35; [b,a] = iirnotch(wo, bw)
    scipy's ``iirnotch`` takes (w0_normalised, Q) with Q = wo/bw = 35.
    """
    w0 = f0 / (fs / 2.0)
    b, a = iirnotch(w0, q)
    return filtfilt(b, a, sig)


def whiten_ar1(sig: np.ndarray) -> np.ndarray:
    """AR(1) pre-whitening, matching ``WhitenSignalIn`` with ArOrder = 1.

    Fits a first-order autoregressive model and applies the inverse filter
    ``b = [1, -a]`` to flatten the 1/f spectrum, then applies the same single
    sample circular shift ``Filter0In`` uses to keep it ~zero-phase.
    """
    x = np.asarray(sig, dtype=np.float64)
    n = x.size
    if n < 3:
        return x
    # Lag-0 / lag-1 autocorrelation -> AR(1) coefficient (Yule-Walker, order 1).
    r0 = np.dot(x, x) / n
    r1 = np.dot(x[:-1], x[1:]) / n
    a = r1 / r0 if r0 != 0 else 0.0
    b = np.array([1.0, -a])
    y = lfilter(b, [1.0], x)
    # Filter0In shifts by length(b)/2 = 1 sample to undo the filter delay.
    return np.concatenate([y[1:], y[:1]])


def multitaper_spectrogram(
    sig: np.ndarray,
    fs: float,
    nfft: int = 3072,
    win_length: int | None = None,
    nw: float = 3.0,
    n_tapers: int = 5,
    freq_range: tuple[float, float] = (0.0, 200.0),
):
    """Multitaper spectrogram, port of ``mtchglongIn`` for a single channel.

    Returns
    -------
    spec : (n_freq, n_time) float array     auto-spectrum (power)
    fo   : (n_freq,)        frequency axis (Hz)
    to   : (n_time,)        time axis (s, bin centres at 1 s spacing)
    """
    sig = np.asarray(sig, dtype=np.float64)
    if win_length is None:
        win_length = int(round(fs))
    winstep = win_length  # nOverlap = 0
    n_samples = sig.size
    n_chunks = max(1, matlab_round((n_samples - win_length) / winstep))

    tapers = dpss(win_length, nw, n_tapers)        # (n_tapers, win_length)
    freqs = np.arange(nfft // 2 + 1) * fs / nfft   # rfft bins
    sel = (freqs > freq_range[0]) & (freqs < freq_range[1])
    fo = freqs[sel]

    normfac = np.sqrt(2.0 / nfft)
    spec = np.empty((fo.size, n_chunks), dtype=np.float64)
    for j in range(n_chunks):
        seg = sig[j * winstep: j * winstep + win_length]
        if seg.size < win_length:                  # last chunk guard
            seg = np.pad(seg, (0, win_length - seg.size))
        tapered = tapers * seg[None, :]            # (n_tapers, win_length)
        ft = np.fft.rfft(tapered, n=nfft, axis=1) * normfac
        ft = ft[:, sel]
        spec[:, j] = np.mean(np.abs(ft) ** 2, axis=0)  # average over tapers

    to = winstep * np.arange(n_chunks) / fs
    return spec, fo, to


def compute_channel_spectrogram(raw: np.ndarray, fs: float):
    """Full per-channel pipeline: clip -> notch -> whiten -> spectrogram.

    Returns ``(spec, fo, to, cleaned)`` where ``cleaned`` is the clipped +
    notched LFP trace (NOT whitened) used for the time-domain display, matching
    what the MATLAB GUI plots.
    """
    cleaned = notch_filter(mad_clip(raw), fs)
    whitened = whiten_ar1(cleaned)
    spec, fo, to = multitaper_spectrogram(whitened, fs)
    return spec, fo, to, cleaned.astype(np.float32)


def detect_sampling_rate(timestamps_file: str, default: float = 1000.0) -> float | None:
    """Infer the LFP sampling rate from a timestamps ``.npy`` file.

    Returns ``fs = (n - 1) / (t[-1] - t[0])`` rounded to a sensible value, or
    ``None`` if the file is missing/unusable.
    """
    try:
        ts = np.load(timestamps_file, mmap_mode="r")
        ts = np.asarray(ts).ravel()
        if ts.size < 2:
            return None
        span = float(ts[-1]) - float(ts[0])
        if span <= 0:
            return None
        fs = (ts.size - 1) / span
        # snap to the nearest 1 Hz; common rates land exactly
        return float(round(fs))
    except Exception:
        return None


# Match a channel file whether or not it carries a rat_sessiondate_ prefix.
_CHANNEL_FILE_RE = re.compile(r"lfp_nt(\d+)_ch\d+\.npy$", re.IGNORECASE)

# rat token ... Trodes datetime YYYYMMDD_HHMMSS -> "Rat6_20260707_091045_"
_SESSION_RE = re.compile(r"(?P<rat>[A-Za-z]+\d+).*?(?P<dt>\d{8}_\d{6})")

# A full session prefix ("Rat6_20260707_091045_") and nothing more. Used to tell
# an unprefixed / properly-prefixed output file apart from a longer name that
# merely *ends* with the same suffix (e.g. emg_from_lfp_timestamps.npy vs
# lfp_timestamps.npy).
_PREFIX_ONLY_RE = re.compile(r"^[A-Za-z]+\d+_\d{8}_\d{6}_$")


def session_prefix(name) -> str:
    """Return ``'Rat6_20260707_091045_'`` from a recording name, or ``''``."""
    if not name:
        return ""
    m = _SESSION_RE.search(str(name))
    return f"{m.group('rat')}_{m.group('dt')}_" if m else ""


def find_output(folder, suffix):
    """Find ``<folder>/<prefix?><suffix>`` (prefixed or not). Returns a path or None.

    Only accepts the bare ``suffix`` or a full ``Rat<n>_<date>_<time>_`` session
    prefix + suffix. This stops a ``*<suffix>`` glob from matching a longer name
    that merely ends with the same text — e.g. ``lfp_timestamps.npy`` must not
    resolve to ``..._emg_from_lfp_timestamps.npy`` (a 5 Hz EMG timebase), which
    would otherwise mis-detect the LFP sampling rate.
    """
    exact = os.path.join(folder, suffix)
    if os.path.isfile(exact):
        return exact
    import glob
    matches = []
    for p in glob.glob(os.path.join(folder, f"*{suffix}")):
        prefix = os.path.basename(p)[:-len(suffix)]
        if prefix == "" or _PREFIX_ONLY_RE.match(prefix):
            matches.append(p)
    matches.sort(key=os.path.getmtime)
    return matches[-1] if matches else None


def output_prefix(folder):
    """Recover the ``rat_sessiondate_`` prefix already used by files in ``folder``.

    Looks at known step-8 outputs and returns their common prefix (``''`` if the
    files are unprefixed / absent), so new files (e.g. buzsaki_states.npz) can
    match the session's existing naming.
    """
    for suffix in ("lfp_data.npy", "lfp_timestamps.npy", "channel_map.npy"):
        p = find_output(folder, suffix)
        if p is not None:
            base = os.path.basename(p)
            if base.endswith(suffix) and len(base) > len(suffix):
                return base[:-len(suffix)]
    return ""


def find_lfp_source(folder: str):
    """Locate the LFP data in ``folder``, supporting both on-disk layouts.

    Two layouts are recognised, in priority order:

    1. ``lfp_data.npy`` - a single ``[n_samples, n_channels]`` matrix (the
       layout the MATLAB ``Sleep_score_HM_neuron.m`` expects).
    2. ``channels_npy/lfp_ntNN_ch01.npy`` - one file per tetrode/channel
       (the layout the example ``LFP_Output/`` folder actually ships).

    Returns a dict describing the source, or ``None`` if neither is found::

        {'kind': 'matrix',   'path': ...,  'channels': [1..n], 'n_samples': N}
        {'kind': 'channels', 'files': {ch: path}, 'channels': [...], 'n_samples': N}

    ``channels`` is the sorted list of channel numbers available to select.
    """
    mat = find_output(folder, "lfp_data.npy")   # prefixed or not
    if mat is not None:
        arr = np.load(mat, mmap_mode="r")
        n_samples = arr.shape[0]
        n_ch = arr.shape[1] if arr.ndim > 1 else 1
        return {"kind": "matrix", "path": mat,
                "channels": list(range(1, n_ch + 1)), "n_samples": n_samples}

    ch_dir = os.path.join(folder, "channels_npy")
    files: dict[int, str] = {}
    if os.path.isdir(ch_dir):
        for name in os.listdir(ch_dir):
            m = _CHANNEL_FILE_RE.search(name)   # .search tolerates a prefix
            if m:
                files[int(m.group(1))] = os.path.join(ch_dir, name)
    if files:
        first = np.load(next(iter(files.values())), mmap_mode="r")
        return {"kind": "channels", "files": files,
                "channels": sorted(files), "n_samples": first.shape[0]}
    return None


def load_lfp_channel(source: dict, ch: int) -> np.ndarray:
    """Load one channel (1-based) from a source returned by :func:`find_lfp_source`."""
    if source["kind"] == "matrix":
        arr = np.load(source["path"], mmap_mode="r")
        col = arr[:, ch - 1] if arr.ndim > 1 else arr
        return np.asarray(col, dtype=np.float64)
    path = source["files"].get(ch)
    if path is None:
        raise ValueError(f"channel {ch} not found in channels_npy/")
    return np.asarray(np.load(path, mmap_mode="r"), dtype=np.float64).ravel()


def cache_path(out_folder: str, base_name: str) -> str:
    return os.path.join(out_folder, f"{base_name}.eegstates.npz")


def save_cache(path, chs, eeg_fs, specs, fos, to, raw_eeg, motion):
    """Persist computed spectrograms so re-opening a session is instant."""
    np.savez_compressed(
        path,
        chs=np.asarray(chs),
        eeg_fs=np.asarray([eeg_fs]),
        specs=np.stack(specs),                 # (n_ch, n_freq, n_time)
        fo=np.asarray(fos[0]),
        to=np.asarray(to),
        raw_eeg=np.stack([np.asarray(e) for e in raw_eeg]),
        motion=np.asarray(motion),
    )


def load_cache(path, chs, eeg_fs):
    """Return cached data if it matches the requested channels + rate, else None."""
    try:
        d = np.load(path, allow_pickle=False)
    except Exception:
        return None
    if list(d["chs"]) != list(chs) or float(d["eeg_fs"][0]) != float(eeg_fs):
        return None
    specs = [d["specs"][i] for i in range(d["specs"].shape[0])]
    fos = [d["fo"]] * len(specs)
    raw = [d["raw_eeg"][i] for i in range(d["raw_eeg"].shape[0])]
    return specs, fos, d["to"], raw, d["motion"]


def downsample_motion(motion: np.ndarray, n_samples: int, fs: float) -> np.ndarray:
    """Reduce a motion/EMG signal to one value per spectrogram bin.

    Mirrors the logic in ``Sleep_score_HM_neuron.m``: full-rate signals are
    averaged into 1 s bins; already-binned signals are linearly resampled.
    """
    motion = np.asarray(motion, dtype=np.float64).ravel()
    target_len = max(1, matlab_round((n_samples - fs) / fs))
    fs_i = int(round(fs))

    if motion.size > target_len:
        usable = target_len * fs_i
        if motion.size >= usable:
            motion = motion[:usable]
        else:
            motion = np.concatenate([motion, np.zeros(usable - motion.size)])
        return motion.reshape(target_len, fs_i).mean(axis=1)

    orig_x = np.linspace(0.0, 1.0, motion.size)
    target_x = np.linspace(0.0, 1.0, target_len)
    return np.interp(target_x, orig_x, motion)


# --------------------------------------------------------------------------- #
#  Motion processing (ports of TheStateEditor's motion branches)
# --------------------------------------------------------------------------- #
def _to_channels_by_time(arr) -> np.ndarray:
    """Return the signal as ``[n_channels, n_samples]`` (channels = smaller dim)."""
    if arr.ndim == 1:
        return arr[None, :]
    return arr.T if arr.shape[0] > arr.shape[1] else arr


def _zscore_rows(x: np.ndarray) -> np.ndarray:
    """z-score each row (channel) across time, matching ``zscore(x')'``."""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (x - mu) / sd


def _fir_bandpass(sig: np.ndarray, low: float, high: float, fs: float,
                  numtaps: int = 501) -> np.ndarray:
    """Zero-phase FIR band-pass, matching ``filter2(fir1(500,[lo hi]), sig)``.

    ``fir1`` uses a Hamming window (scipy's ``firwin`` default) and produces
    symmetric (linear-phase) taps, so ``filter2`` correlation == convolution.
    """
    nyq = fs / 2.0
    high = min(high, 0.99 * nyq)      # guard: cutoff must stay below Nyquist
    low = max(low, 1e-6)
    taps = firwin(numtaps, [low, high], fs=fs, pass_zero=False)
    return np.convolve(sig, taps, mode="same")


def process_motion(motion_raw, n_samples: int, fs: float,
                   mode: str = "accelerometer") -> np.ndarray:
    """Turn a raw motion signal into one value per spectrogram bin.

    Ports the motion branches of ``TheStateEditor.m``:

    * ``"accelerometer"`` (case 3, the default): ``|z-score|`` each channel,
      sum across channels, 0.1-1 Hz FIR band-pass, average into 1 s bins.
    * ``"meg"`` (case 4): z-score + sum, 100-600 Hz band-pass, square, then
      0.1-1 Hz band-pass, 1 s bins.
    * ``"file"`` (case 5): no processing - just downsample (see
      :func:`downsample_motion`).

    Multi-channel input may be ``[n_channels, n_samples]`` or
    ``[n_samples, n_channels]``; it is aligned to the LFP length first so the
    band-pass runs at the LFP sampling rate, exactly as in the MATLAB tool
    (which expects the motion channels resampled to ``eegFS``).
    """
    mode = (mode or "accelerometer").lower()
    if mode == "file":
        return downsample_motion(motion_raw, n_samples, fs)

    arr = _to_channels_by_time(np.asarray(motion_raw))     # [nCh, N] (maybe mmap)
    n = arr.shape[1]
    if n != n_samples:
        # align to the LFP rate via nearest-sample indexing (memory-light: only
        # reads n_samples points, so huge motion files stay cheap)
        idx = np.linspace(0, n - 1, n_samples).astype(np.int64)
        arr = np.asarray(arr[:, idx], dtype=np.float64)
    else:
        arr = np.asarray(arr, dtype=np.float64)

    if mode == "meg":
        m = _zscore_rows(arr).sum(axis=0)
        m = _fir_bandpass(m, 100.0, 600.0, fs)
        sd = m.std() or 1.0
        m = ((m - m.mean()) / sd) ** 2
        m = _fir_bandpass(m, 0.1, 1.0, fs)
    else:                                                  # accelerometer (case 3)
        m = np.abs(_zscore_rows(arr)).sum(axis=0)
        m = _fir_bandpass(m, 0.1, 1.0, fs)

    return downsample_motion(m, n_samples, fs)             # 1 s bin average
