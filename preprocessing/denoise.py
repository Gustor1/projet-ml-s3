"""
preprocessing/denoise.py — Noise Reduction Filters

Implements two classical DSP denoising algorithms:
    1. Wiener Filter (scipy.signal.wiener)
    2. Spectral Subtraction (overlap-add FFT with noise floor estimation)

Both methods were evaluated across 900+ inferences in Experiments 2–5 (Role 4).

Key findings (see docs/final_report_data_engineer.md):
    - Wiener helps ASR only under stationary white noise at 5 dB SNR (ΔWER = −2.75%)
    - Wiener degrades ASR on pink noise (+11.1% WER), urban noise (+9.3%), babble (+8.1%)
    - Spectral subtraction degrades ASR under ALL conditions (+6.8% to +27.0% WER)
    - Both methods destroy prosodic features needed for SER (−21.4% accuracy)

Engineering recommendation: Do NOT apply denoising by default.
    Use parallel routing (see preprocessing/pipeline.py) when denoising is needed.

References:
    [1] Oppenheim & Lim, "The importance of phase in signals," Proc. IEEE, 1981.
    [2] Evans et al., "On the Fundamental Limitations of Spectral Subtraction," EUSIPCO, 2005.
    [3] Gong et al., "Whisper-AT," Interspeech, 2023.
"""

import numpy as np
from scipy.signal import wiener as _scipy_wiener


def wiener_filter(audio: np.ndarray, mysize: int = 3) -> np.ndarray:
    """
    Apply a Wiener filter to a 1-D audio signal.

    The Wiener filter is an adaptive linear estimator that minimizes the
    mean-square error between the estimated and true signal spectra (MMSE).
    It assumes stationary noise with a flat power spectral density (PSD).

    Parameters
    ----------
    audio : np.ndarray
        Input audio signal (1-D, float32, mono, 16 kHz recommended).
    mysize : int, optional
        Smoothing window size. Must be odd (auto-corrected if even).
        Larger values → more aggressive smoothing.
        Default: 3 (minimal smoothing).

    Returns
    -------
    np.ndarray
        Filtered audio signal (float32).

    Notes
    -----
    - Works well on **stationary white Gaussian noise** at severe SNR (≤ 5 dB).
    - Degrades performance on colored noise (pink 1/f) due to spectral tilt
      distortion: over-attenuates F2/F3 formants above 2 kHz.
    - Degrades performance on non-stationary noise (urban, babble) due to
      tracking lag: the noise estimate cannot follow transient acoustic events.
    - Destroys prosodic micro-features (jitter, shimmer, pitch contours)
      needed by downstream SER models — acts as an "emotional eraser."

    Examples
    --------
    >>> import soundfile as sf
    >>> audio, sr = sf.read("noisy.wav", dtype="float32")
    >>> clean = wiener_filter(audio, mysize=5)
    """
    # Scipy wiener requires odd window size
    if mysize % 2 == 0:
        mysize += 1
    return _scipy_wiener(audio, mysize=mysize).astype(np.float32)


def spectral_subtraction(
    audio: np.ndarray,
    sr: int = 16000,
    alpha: float = 2.0,
    beta: float = 0.01,
    nfft: int = 2048,
    noise_duration: float = 0.5,
) -> np.ndarray:
    """
    Apply spectral subtraction denoising via overlap-add FFT.

    Estimates the noise power spectrum from an initial silence segment,
    then subtracts it (scaled by alpha) from each frame's power spectrum.
    A spectral floor (beta) prevents negative power values.

    Parameters
    ----------
    audio : np.ndarray
        Input audio signal (1-D, float32, mono).
    sr : int, optional
        Sample rate in Hz. Default: 16000.
    alpha : float, optional
        Oversubtraction factor. Higher values remove more noise but
        introduce more spectral holes and musical noise artifacts.
        Typical range: 1.0–5.0. Default: 2.0.
    beta : float, optional
        Spectral floor as a fraction of estimated noise power.
        Prevents complete zeroing of frequency bins.
        Typical range: 0.001–0.1. Default: 0.01.
    nfft : int, optional
        FFT window size. Default: 2048 (128 ms at 16 kHz).
    noise_duration : float, optional
        Duration (seconds) of initial audio used for noise estimation.
        Default: 0.5 s.

    Returns
    -------
    np.ndarray
        Denoised audio signal (float32), peak-normalized.

    Notes
    -----
    - This method was initially broken due to an FFT boundary mismatch
      on short final frames (documented in docs/journal/2026-06-13-debug-fft.md).
      The fix uses `chunk_len = min(nfft, n - i)` for safe overlap-add.
    - After the fix, spectral subtraction **consistently degraded** ASR
      performance across ALL noise types (+6.79% to +27.0% WER increase).
    - Degradation is caused by magnitude subtraction errors that create
      spectral holes (zeroed frequency bands) and musical noise (isolated
      spectral peaks). Whisper's self-attention interprets these artifacts
      as phonemic structures, leading to word insertion errors.
    - Kept in the codebase as a documented negative result, per the
      professor's guidance that "negative results are completely acceptable."

    References
    ----------
    Evans et al., "On the Fundamental Limitations of Spectral Subtraction,"
    Proc. EUSIPCO, 2005.
    """
    n = len(audio)
    hop = nfft // 2

    # Estimate noise PSD from initial silence segment
    noise_len = int(min(noise_duration * sr, n))
    noise_frame = audio[:noise_len]
    noise_spec = np.abs(np.fft.rfft(noise_frame, n=nfft))
    noise_pow = np.mean(noise_spec ** 2)

    result = np.zeros(n)
    window = np.hanning(nfft)

    for i in range(0, n, hop):
        frame = audio[i : i + nfft]
        if len(frame) < nfft:
            frame = np.pad(frame, (0, nfft - len(frame)))

        spec = np.fft.rfft(frame * window)
        power = np.abs(spec) ** 2
        clean_pow = np.maximum(power - alpha * noise_pow, beta * noise_pow)
        clean_spec = np.sqrt(clean_pow) * np.exp(1j * np.angle(spec))
        clean_frame = np.fft.irfft(clean_spec) * window

        # Safe boundary check (FFT bug fix — see docs/journal/2026-06-13-debug-fft.md)
        chunk_len = min(nfft, n - i)
        result[i : i + chunk_len] += clean_frame[:chunk_len]

    # Peak normalization
    max_val = np.max(np.abs(result))
    if max_val > 0:
        result = result / max_val
    return result[:n].astype(np.float32)
