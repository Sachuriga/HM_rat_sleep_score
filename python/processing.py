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

import numpy as np
from scipy.signal import iirnotch, filtfilt, lfilter
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
