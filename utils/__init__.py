"""
utils/ — Shared Utilities for the Multimodal Pipeline
======================================================
Role 1: Pipeline Architect & DevOps

Provides common helpers used across all pipeline stages:
  - config_loader: YAML configuration management
  - audio_utils: Audio I/O, normalization, silence trimming
"""

from utils.config_loader import load_config
from utils.audio_utils import load_audio, normalize_volume, trim_silence

__all__ = [
    "load_config",
    "load_audio",
    "normalize_volume",
    "trim_silence",
]
