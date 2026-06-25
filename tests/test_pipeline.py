#!/usr/bin/env python3
"""
tests/test_pipeline.py
Unit tests for core pipeline functions (no model loading required).
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_preprocess_none():
    """preprocess_none should return input unchanged."""
    from main import preprocess_none

    audio = np.random.randn(16000).astype(np.float32)
    result = preprocess_none(audio)
    np.testing.assert_array_equal(result, audio)


def test_preprocess_wiener_output_shape():
    """Wiener filter should preserve audio length."""
    from main import preprocess_wiener

    audio = np.random.randn(16000).astype(np.float32)
    result = preprocess_wiener(audio, size=3)
    assert result.shape == audio.shape, "Wiener output shape must match input"


def test_preprocess_wiener_odd_size():
    """Wiener filter should handle even size by adding 1."""
    from main import preprocess_wiener

    audio = np.random.randn(16000).astype(np.float32)
    # Even size should not crash
    result = preprocess_wiener(audio, size=4)
    assert result.shape == audio.shape


def test_preprocess_spectral_subtraction_output_shape():
    """Spectral subtraction should preserve audio length."""
    from main import preprocess_spectral_subtraction

    audio = np.random.randn(16000).astype(np.float32)
    result = preprocess_spectral_subtraction(audio, sr=16000)
    assert len(result) == len(audio), "Output length must match input"


def test_preprocess_spectral_subtraction_bounded():
    """Spectral subtraction output should be bounded after normalization."""
    from main import preprocess_spectral_subtraction

    audio = np.random.randn(32000).astype(np.float32) * 0.5
    result = preprocess_spectral_subtraction(audio, sr=16000)
    assert np.max(np.abs(result)) <= 1.0 + 1e-6, "Output should be peak-normalized to ≤ 1.0"


def test_normalize_volume():
    """normalize_volume should scale audio to [-1, 1]."""
    from main import normalize_volume

    audio = np.array([0.0, 0.5, -0.5, 0.25], dtype=np.float32)
    result = normalize_volume(audio)
    assert np.max(np.abs(result)) <= 1.0 + 1e-6


def test_normalize_volume_silence():
    """normalize_volume on silence should return zeros."""
    from main import normalize_volume

    audio = np.zeros(16000, dtype=np.float32)
    result = normalize_volume(audio)
    np.testing.assert_array_equal(result, audio)


def test_detect_sarcasm_positive_angry():
    """Positive text + angry voice should trigger sarcasm."""
    from main import detect_sarcasm

    is_sarcastic, reason = detect_sarcasm("positive", "angry")
    assert is_sarcastic is True
    assert "negative voice" in reason.lower() or "angry" in reason.lower()


def test_detect_sarcasm_negative_happy():
    """Negative text + happy voice should trigger sarcasm."""
    from main import detect_sarcasm

    is_sarcastic, reason = detect_sarcasm("negative", "happy")
    assert is_sarcastic is True


def test_detect_sarcasm_normal():
    """Positive text + happy voice should not trigger sarcasm."""
    from main import detect_sarcasm

    is_sarcastic, reason = detect_sarcasm("positive", "happy")
    assert is_sarcastic is False


def test_detect_sarcasm_neutral_emotional():
    """Neutral text + emotional voice should trigger sarcasm."""
    from main import detect_sarcasm

    is_sarcastic, reason = detect_sarcasm("neutral", "angry")
    assert is_sarcastic is True


def test_load_config():
    """load_config should parse config.yaml without errors."""
    from main import load_config

    config = load_config("configs/config.yaml")
    assert isinstance(config, dict)
    assert "project" in config
