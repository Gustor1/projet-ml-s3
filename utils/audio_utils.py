#!/usr/bin/env python3
"""
utils/audio_utils.py
Shared audio helper functions used across the pipeline.

Role 1: Pipeline Architect & DevOps

These utilities are intentionally lightweight and model-agnostic.
They complement the DSP-heavy functions in preprocessing/ by providing
common operations needed by multiple pipeline stages.
"""

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def load_audio(audio_path: str, target_sr: int = 16000) -> tuple:
    """
    Load an audio file and convert to mono float32.

    Parameters
    ----------
    audio_path : str
        Path to the audio file (WAV recommended).
    target_sr : int
        Expected sample rate (used for logging only; no resampling).

    Returns
    -------
    tuple of (np.ndarray, int)
        Audio signal as 1D float32 array, and the file's sample rate.

    Raises
    ------
    FileNotFoundError
        If the audio file does not exist.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio, sr = sf.read(str(audio_path), dtype="float32")

    # Convert stereo to mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
        logger.info(f"  Converted stereo to mono")

    duration = len(audio) / sr
    logger.info(f"Loaded audio: {audio_path.name} | {duration:.2f}s | {sr} Hz")

    if sr != target_sr:
        logger.warning(
            f"  Sample rate mismatch: file={sr} Hz, expected={target_sr} Hz. "
            f"Consider resampling for optimal model performance."
        )

    return audio, sr


def normalize_volume(audio: np.ndarray) -> np.ndarray:
    """
    Peak-normalize audio signal to [-1, 1] range.

    Parameters
    ----------
    audio : np.ndarray
        1D floating-point audio signal.

    Returns
    -------
    np.ndarray
        Peak-normalized audio signal.
    """
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio


def trim_silence(audio: np.ndarray, top_db: int = 30) -> np.ndarray:
    """
    Trim leading and trailing silence using energy-based detection.

    Parameters
    ----------
    audio : np.ndarray
        1D floating-point audio signal.
    top_db : int
        Threshold in dB below peak to consider as silence.

    Returns
    -------
    np.ndarray
        Trimmed audio signal.
    """
    import librosa

    try:
        intervals = librosa.effects.split(audio, top_db=top_db)
        if len(intervals) > 0:
            start = intervals[0][0]
            end = intervals[-1][1]
            return audio[start:end]
    except Exception:
        pass
    return audio
