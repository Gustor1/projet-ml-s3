"""
preprocessing/pipeline.py — End-to-End Preprocessing Pipeline with Parallel Routing

Implements the recommended parallel routing architecture for multi-task
speech processing pipelines that require both ASR and SER:

    ┌─────────────┐
    │  Raw Audio   │
    └──────┬───────┘
           │
     ┌─────┴─────┐
     │   Split    │
     └─────┬─────┘
           │
    ┌──────┴──────────────────┐
    │                         │
    ▼                         ▼
  Denoise                Trim + Normalize
  (Wiener /              (silence removal
   Spec. Sub.)            + peak norm)
    │                         │
    ▼                         ▼
  ASR Stream              SER Stream
  (Whisper)              (Wav2Vec2)
    │                         │
    ▼                         ▼
  Transcription           Emotion Preds
    │                         │
    └────────────┬────────────┘
                 │
           Fusion Engine
        (text sentiment +
         pitch → calibration)
                 │
                 ▼
          Final Predictions

Design rationale (from Experiment 6, Role 4):
    - Wiener filtering drops SER accuracy from 45.8% to 24.4% (−21.4%)
      because it smooths prosodic micro-features (jitter, shimmer, pitch)
    - The parallel routing architecture preserves these features for SER
      while still providing denoised audio for ASR when beneficial
    - Fusion calibration adds +20% relative accuracy gain for SER

References:
    [1] docs/final_report_data_engineer.md §5.3 — Parallel Routing Architecture
    [2] docs/experiment-6-emotions.md — SER Degradation Analysis
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from preprocessing.denoise import wiener_filter, spectral_subtraction
from preprocessing.vad import trim_silence, normalize_volume
from preprocessing.features import estimate_snr, estimate_pitch_contour

logger = logging.getLogger(__name__)


class DenoiseMethod(Enum):
    """Available denoising methods."""
    NONE = "none"
    WIENER = "wiener"
    SPECTRAL_SUBTRACTION = "spectral_subtraction"


@dataclass
class PreprocessingConfig:
    """
    Configuration for the preprocessing pipeline.

    Attributes
    ----------
    denoise_method : DenoiseMethod
        Which denoising algorithm to apply to the ASR stream.
        Default: NONE (recommended — see Insights 8–10).
    sample_rate : int
        Target sample rate in Hz. Default: 16000.
    wiener_size : int
        Smoothing window for Wiener filter (odd integer). Default: 3.
    ss_alpha : float
        Oversubtraction factor for spectral subtraction. Default: 2.0.
    ss_beta : float
        Spectral floor for spectral subtraction. Default: 0.01.
    trim_top_db : float
        Silence threshold (dB below peak) for VAD trimming. Default: 30.0.
    snr_adaptive : bool
        If True, only apply denoising when estimated SNR < snr_threshold.
        Default: False.
    snr_threshold : float
        SNR threshold (dB) below which denoising is activated.
        Default: 10.0 (based on Insight 2: Wiener only helps at ≤ 5 dB).
    compute_pitch : bool
        Whether to extract pitch contour for fusion. Default: True.
    pitch_fmin : float
        Minimum F0 for YIN estimator. Default: 75.0.
    pitch_fmax : float
        Maximum F0 for YIN estimator. Default: 400.0.
    """
    denoise_method: DenoiseMethod = DenoiseMethod.NONE
    sample_rate: int = 16000
    wiener_size: int = 3
    ss_alpha: float = 2.0
    ss_beta: float = 0.01
    trim_top_db: float = 30.0
    snr_adaptive: bool = False
    snr_threshold: float = 10.0
    compute_pitch: bool = True
    pitch_fmin: float = 75.0
    pitch_fmax: float = 400.0


@dataclass
class PreprocessingResult:
    """
    Output of the preprocessing pipeline.

    Contains both the ASR-optimized and SER-optimized audio streams,
    plus extracted acoustic features.

    Attributes
    ----------
    asr_audio : np.ndarray
        Audio optimized for ASR (denoised if configured). Feed to Whisper.
    ser_audio : np.ndarray
        Audio optimized for SER (trimmed + normalized, NO denoising).
        Feed to Wav2Vec2.
    raw_audio : np.ndarray
        Original unmodified audio for reference.
    estimated_snr : float
        Estimated SNR in dB.
    mean_pitch : float
        Mean fundamental frequency (F0) in Hz.
    std_pitch : float
        Standard deviation of F0 in Hz.
    pitch_contour : np.ndarray
        Full F0 contour array.
    denoise_applied : bool
        Whether denoising was actually applied (may be skipped by SNR gate).
    denoise_method_used : str
        Name of the denoising method that was applied.
    """
    asr_audio: np.ndarray = field(default_factory=lambda: np.array([]))
    ser_audio: np.ndarray = field(default_factory=lambda: np.array([]))
    raw_audio: np.ndarray = field(default_factory=lambda: np.array([]))
    estimated_snr: float = 0.0
    mean_pitch: float = 0.0
    std_pitch: float = 0.0
    pitch_contour: np.ndarray = field(default_factory=lambda: np.array([]))
    denoise_applied: bool = False
    denoise_method_used: str = "none"


class PreprocessingPipeline:
    """
    End-to-end audio preprocessing pipeline with parallel routing.

    Produces two output streams from a single input:
        1. ASR stream: optionally denoised (for Whisper)
        2. SER stream: trimmed + normalized only (for Wav2Vec2)

    Also extracts acoustic features (SNR, pitch) for fusion calibration.

    Parameters
    ----------
    config : PreprocessingConfig, optional
        Pipeline configuration. Uses defaults if not provided.

    Examples
    --------
    >>> from preprocessing.pipeline import PreprocessingPipeline
    >>> pipe = PreprocessingPipeline()
    >>> result = pipe.process(audio_array, sr=16000)
    >>> asr_input = result.asr_audio      # Feed to Whisper
    >>> ser_input = result.ser_audio       # Feed to Wav2Vec2
    >>> pitch = result.mean_pitch          # For fusion calibration
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()

    def process(self, audio: np.ndarray, sr: int = 16000) -> PreprocessingResult:
        """
        Run the full preprocessing pipeline on an audio signal.

        Parameters
        ----------
        audio : np.ndarray
            Input audio signal (1-D, float32, mono).
        sr : int, optional
            Sample rate in Hz. Default: 16000.

        Returns
        -------
        PreprocessingResult
            Contains both ASR and SER audio streams plus features.
        """
        result = PreprocessingResult()
        result.raw_audio = audio.copy()

        # --- Step 1: Estimate SNR ---
        result.estimated_snr = estimate_snr(audio, sr=sr)
        logger.info(f"Estimated SNR: {result.estimated_snr:.1f} dB")

        # --- Step 2: Extract pitch contour (from raw audio) ---
        if self.config.compute_pitch:
            result.mean_pitch, result.std_pitch, result.pitch_contour = (
                estimate_pitch_contour(
                    audio, sr=sr,
                    fmin=self.config.pitch_fmin,
                    fmax=self.config.pitch_fmax,
                )
            )
            logger.info(
                f"Pitch: mean={result.mean_pitch:.1f} Hz, "
                f"std={result.std_pitch:.1f} Hz"
            )

        # --- Step 3: Prepare SER stream (trim + normalize, NO denoising) ---
        ser_audio = trim_silence(audio, top_db=self.config.trim_top_db, sr=sr)
        ser_audio = normalize_volume(ser_audio)
        result.ser_audio = ser_audio

        # --- Step 4: Prepare ASR stream (optionally denoise) ---
        denoise_method = self.config.denoise_method
        apply_denoise = denoise_method != DenoiseMethod.NONE

        # SNR-adaptive gating: skip denoising if SNR is high enough
        if apply_denoise and self.config.snr_adaptive:
            if result.estimated_snr > self.config.snr_threshold:
                logger.info(
                    f"SNR ({result.estimated_snr:.1f} dB) > threshold "
                    f"({self.config.snr_threshold} dB) — skipping denoising"
                )
                apply_denoise = False

        if apply_denoise:
            if denoise_method == DenoiseMethod.WIENER:
                result.asr_audio = wiener_filter(audio, mysize=self.config.wiener_size)
                result.denoise_method_used = "wiener"
            elif denoise_method == DenoiseMethod.SPECTRAL_SUBTRACTION:
                result.asr_audio = spectral_subtraction(
                    audio, sr=sr,
                    alpha=self.config.ss_alpha,
                    beta=self.config.ss_beta,
                )
                result.denoise_method_used = "spectral_subtraction"
            result.denoise_applied = True
            logger.info(f"Denoising applied: {result.denoise_method_used}")
        else:
            result.asr_audio = audio.copy()
            result.denoise_method_used = "none"
            result.denoise_applied = False
            logger.info("No denoising applied (recommended default)")

        return result


def run_preprocessing(
    audio: np.ndarray,
    sr: int = 16000,
    denoise_method: str = "none",
    **kwargs,
) -> PreprocessingResult:
    """
    Convenience function to run preprocessing with minimal configuration.

    This is the recommended entry point for simple use cases.
    For full control, instantiate PreprocessingPipeline directly.

    Parameters
    ----------
    audio : np.ndarray
        Input audio signal (1-D, float32, mono).
    sr : int, optional
        Sample rate in Hz. Default: 16000.
    denoise_method : str, optional
        Denoising method: "none", "wiener", or "spectral_subtraction".
        Default: "none" (recommended).
    **kwargs
        Additional keyword arguments passed to PreprocessingConfig.

    Returns
    -------
    PreprocessingResult
        Contains both ASR and SER audio streams plus features.

    Examples
    --------
    >>> import soundfile as sf
    >>> from preprocessing import run_preprocessing
    >>> audio, sr = sf.read("recording.wav", dtype="float32")
    >>> result = run_preprocessing(audio, sr=sr, denoise_method="wiener")
    >>> # Feed result.asr_audio to Whisper, result.ser_audio to Wav2Vec2
    """
    method_map = {
        "none": DenoiseMethod.NONE,
        "wiener": DenoiseMethod.WIENER,
        "spectral_subtraction": DenoiseMethod.SPECTRAL_SUBTRACTION,
    }

    config = PreprocessingConfig(
        denoise_method=method_map.get(denoise_method.lower(), DenoiseMethod.NONE),
        sample_rate=sr,
        **kwargs,
    )

    pipeline = PreprocessingPipeline(config)
    return pipeline.process(audio, sr=sr)
