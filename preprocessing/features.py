"""
preprocessing/features.py — Acoustic Feature Extraction

Provides utility functions for extracting acoustic features from audio signals:
    1. estimate_snr       — Signal-to-Noise Ratio estimation
    2. estimate_pitch_contour — Fundamental frequency (F0) tracking via YIN

These features are used for:
    - SNR-based adaptive preprocessing decisions (Insight 2: the "Goldilocks Zone")
    - Pitch-based multimodal fusion calibration (Experiment 6, Role 4)
    - Audio quality monitoring in production pipelines

References:
    [1] de Cheveigné & Kawahara, "YIN, a fundamental frequency estimator for
        speech and music," JASA, vol. 111(4), pp. 1917–1930, 2002.
"""

import numpy as np

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False


def estimate_snr(audio: np.ndarray, sr: int = 16000, noise_duration: float = 0.5) -> float:
    """
    Estimate the Signal-to-Noise Ratio (SNR) of an audio signal.

    Uses a simple energy-based method: assumes the first `noise_duration`
    seconds contain primarily noise, and the remainder contains signal + noise.

    Parameters
    ----------
    audio : np.ndarray
        Audio signal (1-D, float32).
    sr : int, optional
        Sample rate in Hz. Default: 16000.
    noise_duration : float, optional
        Duration (seconds) of initial segment assumed to be noise-only.
        Default: 0.5 s.

    Returns
    -------
    float
        Estimated SNR in dB. Returns 0.0 on error.

    Notes
    -----
    - This is a coarse estimate. For accurate SNR measurement, use
      a VAD-segmented approach or reference-based methods (PESQ, STOI).
    - Used in Insight 2 to demonstrate that Wiener filtering only helps
      when SNR < 10 dB (the "Goldilocks Zone").
    - Negative SNR values indicate that the noise segment has higher
      energy than the signal segment, which can happen if the recording
      starts with speech rather than silence.
    """
    try:
        noise_len = int(min(noise_duration * sr, len(audio)))
        if noise_len <= 0:
            return 0.0

        noise_power = np.mean(audio[:noise_len] ** 2)
        signal_region = audio[noise_len:] if len(audio) > noise_len else audio
        signal_power = np.mean(signal_region ** 2)

        # Avoid log(0) with epsilon floor
        if noise_power < 1e-10:
            noise_power = 1e-10
        if signal_power < 1e-10:
            signal_power = 1e-10

        snr = 10 * np.log10(signal_power / noise_power)
        return float(snr)
    except Exception:
        return 0.0


def estimate_pitch_contour(
    audio: np.ndarray,
    sr: int = 16000,
    fmin: float = 75.0,
    fmax: float = 400.0,
    max_duration: float = 10.0,
) -> tuple:
    """
    Extract the fundamental frequency (F0) contour using the YIN algorithm.

    Returns the mean pitch, pitch standard deviation, and the full F0
    contour array. These are used for multimodal fusion calibration
    (see preprocessing/fusion.py).

    Parameters
    ----------
    audio : np.ndarray
        Audio signal (1-D, float32).
    sr : int, optional
        Sample rate in Hz. Default: 16000.
    fmin : float, optional
        Minimum expected F0 in Hz. Default: 75.0 (low male voice).
    fmax : float, optional
        Maximum expected F0 in Hz. Default: 400.0 (high female/child voice).
    max_duration : float, optional
        Maximum audio duration (seconds) to analyze for pitch.
        Caps computation time on long recordings. Default: 10.0 s.

    Returns
    -------
    tuple of (float, float, np.ndarray)
        - mean_f0 : Mean fundamental frequency in Hz (0.0 if no valid pitch found)
        - std_f0  : Standard deviation of F0 in Hz
        - f0      : Full F0 contour array (may contain NaN/out-of-range values)

    Notes
    -----
    - Requires librosa for YIN estimation. Returns (0.0, 0.0, empty array)
      if librosa is not installed.
    - High mean pitch (> 180 Hz) combined with positive text sentiment
      is used to correct false "angry" → "happy" misclassifications in
      the multimodal fusion heuristic (see fusion.py).
    - Low mean pitch (< 130 Hz) boosts "sad" and "neutral" classes.

    References
    ----------
    de Cheveigné & Kawahara, "YIN, a fundamental frequency estimator for
    speech and music," JASA, vol. 111(4), pp. 1917–1930, 2002.
    """
    if not _HAS_LIBROSA:
        return 0.0, 0.0, np.array([])

    try:
        # Cap analysis duration to avoid long processing times
        y_pitch = audio
        max_samples = int(max_duration * sr)
        if len(y_pitch) > max_samples:
            y_pitch = y_pitch[:max_samples]

        # YIN fundamental frequency estimation
        f0 = librosa.yin(y_pitch, fmin=fmin, fmax=fmax, sr=sr)

        # Filter out invalid values
        valid_f0 = f0[
            (f0 >= fmin) & (f0 <= fmax) & (~np.isnan(f0)) & (~np.isinf(f0))
        ]

        if len(valid_f0) > 0:
            return float(np.mean(valid_f0)), float(np.std(valid_f0)), f0
        else:
            return 0.0, 0.0, f0
    except Exception:
        return 0.0, 0.0, np.array([])
