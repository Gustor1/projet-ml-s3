"""
preprocessing — Audio Preprocessing Module for Speech Pipeline

This package provides modular, reusable audio preprocessing components
for a multimodal speech analysis pipeline (ASR + NLP + SER).

Modules:
    denoise     — Wiener filtering and Spectral Subtraction denoising
    vad         — Voice Activity Detection, silence trimming, and volume normalization
    features    — Acoustic feature extraction (SNR estimation, pitch/F0 tracking via YIN)
    fusion      — Multimodal fusion calibration heuristic (text sentiment + pitch → SER correction)
    pipeline    — End-to-end preprocessing pipeline with parallel routing (denoised→ASR, raw→SER)

Design Rationale (from Role 4 experiments):
    Classical DSP filters (Wiener, Spectral Subtraction) help ASR under stationary
    white noise but degrade performance on realistic noise (pink, urban, babble).
    They also destroy prosodic cues needed for Speech Emotion Recognition.
    
    The recommended architecture is **parallel routing**:
        - Denoised audio stream → ASR (Whisper)
        - Raw audio with only silence trimming + peak normalization → SER (Wav2Vec2)
    
    This module implements that architecture.

References:
    [1] Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision," ICML 2022.
    [2] Gong et al., "Whisper-AT: Noise-Robust ASR are also Strong Audio Event Taggers," Interspeech 2023.
    [3] Evans et al., "On the Fundamental Limitations of Spectral Subtraction," EUSIPCO 2005.
    [4] Valin, "A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement," WASPAA 2018.
"""

from preprocessing.denoise import wiener_filter, spectral_subtraction
from preprocessing.vad import trim_silence, normalize_volume, energy_vad
from preprocessing.features import estimate_snr, estimate_pitch_contour
from preprocessing.fusion import fuse_modalities
from preprocessing.pipeline import PreprocessingPipeline, run_preprocessing

__all__ = [
    # Denoising
    "wiener_filter",
    "spectral_subtraction",
    # VAD & normalization
    "trim_silence",
    "normalize_volume",
    "energy_vad",
    # Acoustic features
    "estimate_snr",
    "estimate_pitch_contour",
    # Multimodal fusion
    "fuse_modalities",
    # Pipeline
    "PreprocessingPipeline",
    "run_preprocessing",
]

__version__ = "1.0.0"
