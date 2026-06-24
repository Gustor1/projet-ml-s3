"""
preprocessing/vad.py — Voice Activity Detection & Audio Normalization

Provides three audio conditioning utilities:
    1. trim_silence     — Removes silent margins using energy-based splitting
    2. normalize_volume — Peak normalization to [-1.0, 1.0]
    3. energy_vad       — Simple energy-threshold Voice Activity Detection

These utilities are applied BEFORE feeding audio to SER models (Wav2Vec2),
ensuring that volume variations from microphone distance and recording
conditions do not skew emotion classification.

Design rationale (from Experiment 6, Role 4):
    Proximity clipping and silent padding cause the SER model to misclassify
    emotions. Applying trim_silence + normalize_volume before SER inference
    improved classification accuracy from 35.71% to 42.86% (+20% relative).

References:
    [1] Livingstone & Russo, "The Ryerson Audio-Visual Database of Emotional
        Speech and Song (RAVDESS)," PLoS ONE, 2018.
"""

import numpy as np

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False


def trim_silence(y: np.ndarray, top_db: float = 30.0, sr: int = 16000) -> np.ndarray:
    """
    Remove leading and trailing silence from an audio signal.

    Uses librosa's energy-based interval detection to find contiguous
    speech regions and crops the signal to the first–last active interval.

    Parameters
    ----------
    y : np.ndarray
        Audio signal (1-D, float32).
    top_db : float, optional
        Threshold below peak energy (in dB) to classify as silence.
        Lower values → more aggressive trimming.
        Default: 30.0 dB (standard for clean speech).
    sr : int, optional
        Sample rate (unused but kept for API consistency). Default: 16000.

    Returns
    -------
    np.ndarray
        Trimmed audio signal. Returns the original signal if no
        non-silent intervals are found.

    Notes
    -----
    - Uses `librosa.effects.split()` internally.
    - Falls back to a simple energy-based trimmer if librosa is unavailable.
    - Applied to the RAW audio stream (not the denoised stream) before
      SER inference in the parallel routing architecture.
    """
    if _HAS_LIBROSA:
        try:
            intervals = librosa.effects.split(y, top_db=top_db)
            if len(intervals) > 0:
                start = intervals[0][0]
                end = intervals[-1][1]
                return y[start:end]
        except Exception:
            pass
        return y
    else:
        # Fallback: simple energy-based trimming without librosa
        return _energy_trim(y, top_db=top_db, sr=sr)


def _energy_trim(y: np.ndarray, top_db: float = 30.0, sr: int = 16000,
                 frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """
    Fallback energy-based silence trimming (no librosa dependency).
    """
    threshold_linear = 10 ** (-top_db / 20.0) * np.max(np.abs(y))
    
    # Compute frame-level energy
    n_frames = 1 + (len(y) - frame_length) // hop_length
    if n_frames <= 0:
        return y

    energies = np.array([
        np.sqrt(np.mean(y[i * hop_length : i * hop_length + frame_length] ** 2))
        for i in range(n_frames)
    ])

    active = np.where(energies > threshold_linear)[0]
    if len(active) == 0:
        return y

    start_sample = active[0] * hop_length
    end_sample = min(active[-1] * hop_length + frame_length, len(y))
    return y[start_sample:end_sample]


def normalize_volume(y: np.ndarray) -> np.ndarray:
    """
    Peak-normalize audio amplitude to [-1.0, 1.0].

    Divides the signal by its absolute maximum value to ensure
    consistent input amplitude for downstream neural models,
    regardless of recording gain or microphone distance.

    Parameters
    ----------
    y : np.ndarray
        Audio signal (1-D, float32).

    Returns
    -------
    np.ndarray
        Peak-normalized audio signal (float32).

    Notes
    -----
    - Returns the original signal unchanged if max(|y|) == 0
      (all-zero / digital silence).
    - This is a simple but critical step: Wav2Vec2 SER accuracy
      improves significantly when input amplitude is consistent,
      because the model's learned feature representations are
      calibrated to a specific dynamic range.
    """
    max_val = np.max(np.abs(y))
    if max_val > 0:
        return (y / max_val).astype(np.float32)
    return y.astype(np.float32)


def energy_vad(
    y: np.ndarray,
    sr: int = 16000,
    frame_duration_ms: float = 25.0,
    hop_duration_ms: float = 10.0,
    energy_threshold_db: float = -40.0,
) -> np.ndarray:
    """
    Simple energy-based Voice Activity Detection (VAD).

    Classifies each audio frame as speech or silence based on
    whether its log-energy exceeds a fixed threshold.

    Parameters
    ----------
    y : np.ndarray
        Audio signal (1-D, float32).
    sr : int, optional
        Sample rate in Hz. Default: 16000.
    frame_duration_ms : float, optional
        Frame length in milliseconds. Default: 25.0 ms.
    hop_duration_ms : float, optional
        Hop length in milliseconds. Default: 10.0 ms.
    energy_threshold_db : float, optional
        Minimum log-energy (dB re: 1.0) to classify as speech.
        Default: −40.0 dB.

    Returns
    -------
    np.ndarray
        Boolean array of shape (n_frames,) where True = speech frame.

    Notes
    -----
    - This is a simple baseline VAD. For production use, consider
      WebRTC VAD (py-webrtcvad) or Silero VAD for better accuracy.
    - Frame-level VAD labels can be used to compute speech-to-silence
      ratio, or to mask non-speech regions before SNR estimation.
    """
    frame_length = int(frame_duration_ms * sr / 1000.0)
    hop_length = int(hop_duration_ms * sr / 1000.0)

    n_frames = 1 + (len(y) - frame_length) // hop_length
    if n_frames <= 0:
        return np.array([], dtype=bool)

    # Compute per-frame log energy
    vad_labels = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        frame = y[i * hop_length : i * hop_length + frame_length]
        rms = np.sqrt(np.mean(frame ** 2))
        # Convert to dB (ref = 1.0)
        energy_db = 20 * np.log10(rms + 1e-10)
        vad_labels[i] = energy_db > energy_threshold_db

    return vad_labels
