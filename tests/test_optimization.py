#!/usr/bin/env python3
"""
tests/test_optimization.py
Unit tests for Role 5 optimization modules.
Tests streaming chunker logic and quantization helpers — no model downloads needed.
"""

import sys
import os
import numpy as np
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------
# StreamingAudioLoader tests
# ---------------------------------------------------------------

def test_streaming_loader_basic_chunking():
    """Verify chunks cover the entire audio with correct boundaries."""
    from optimization.streaming_audio import StreamingAudioLoader, AudioChunk
    import soundfile as sf

    # Create a 10s dummy WAV file
    sr = 16000
    audio = np.random.randn(sr * 10).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sr)
        tmp_path = f.name

    try:
        loader = StreamingAudioLoader(chunk_duration=3.0, overlap_duration=1.0, target_sr=sr)
        chunks = list(loader.load_chunks(tmp_path))

        assert len(chunks) >= 4, f"Expected at least 4 chunks for 10s audio with 3s chunks, got {len(chunks)}"
        assert chunks[0].start_time == 0.0, "First chunk must start at 0"
        assert chunks[-1].is_last is True, "Last chunk must have is_last=True"
        assert chunks[0].is_last is False, "First chunk should not be last"

        # Verify monotonic start times
        for i in range(1, len(chunks)):
            assert chunks[i].start_time > chunks[i - 1].start_time, "Chunks must have increasing start times"
    finally:
        os.unlink(tmp_path)


def test_streaming_loader_short_audio():
    """Audio shorter than chunk_duration should produce exactly 1 chunk."""
    from optimization.streaming_audio import StreamingAudioLoader
    import soundfile as sf

    sr = 16000
    audio = np.random.randn(sr * 2).astype(np.float32)  # 2s audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sr)
        tmp_path = f.name

    try:
        loader = StreamingAudioLoader(chunk_duration=30.0, overlap_duration=5.0, target_sr=sr)
        chunks = list(loader.load_chunks(tmp_path))

        assert len(chunks) == 1, f"Expected 1 chunk for short audio, got {len(chunks)}"
        assert chunks[0].is_last is True
        assert chunks[0].chunk_index == 0
    finally:
        os.unlink(tmp_path)


def test_streaming_loader_overlap_validation():
    """Overlap >= chunk_duration should raise ValueError."""
    from optimization.streaming_audio import StreamingAudioLoader
    import pytest

    with pytest.raises(ValueError):
        StreamingAudioLoader(chunk_duration=5.0, overlap_duration=5.0)

    with pytest.raises(ValueError):
        StreamingAudioLoader(chunk_duration=5.0, overlap_duration=10.0)


def test_streaming_loader_chunk_data_shape():
    """Each chunk's data should be 1D and within expected length."""
    from optimization.streaming_audio import StreamingAudioLoader
    import soundfile as sf

    sr = 16000
    audio = np.random.randn(sr * 5).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sr)
        tmp_path = f.name

    try:
        loader = StreamingAudioLoader(chunk_duration=2.0, overlap_duration=0.5, target_sr=sr)
        for chunk in loader.load_chunks(tmp_path):
            assert chunk.data.ndim == 1, "Chunk data must be 1D"
            assert len(chunk.data) <= int(2.0 * sr) + 1, "Chunk must not exceed chunk_duration samples"
            assert chunk.sample_rate == sr
    finally:
        os.unlink(tmp_path)


def test_streaming_loader_file_not_found():
    """Missing file should raise FileNotFoundError."""
    from optimization.streaming_audio import StreamingAudioLoader
    import pytest

    loader = StreamingAudioLoader()
    with pytest.raises(FileNotFoundError):
        list(loader.load_chunks("/nonexistent/path/audio.wav"))


# ---------------------------------------------------------------
# merge_transcriptions tests
# ---------------------------------------------------------------

def test_merge_transcriptions_no_overlap():
    """Non-overlapping texts should be concatenated."""
    from optimization.streaming_audio import merge_transcriptions

    result = merge_transcriptions(["Hello world", "How are you"])
    assert result == "Hello world How are you"


def test_merge_transcriptions_with_overlap():
    """Overlapping words at boundaries should be deduplicated."""
    from optimization.streaming_audio import merge_transcriptions

    result = merge_transcriptions(["The cat sat on", "sat on the mat"], overlap_words=3)
    assert result == "The cat sat on the mat"


def test_merge_transcriptions_single():
    """Single chunk should return as-is."""
    from optimization.streaming_audio import merge_transcriptions

    result = merge_transcriptions(["Hello world"])
    assert result == "Hello world"


def test_merge_transcriptions_empty():
    """Empty list should return empty string."""
    from optimization.streaming_audio import merge_transcriptions

    result = merge_transcriptions([])
    assert result == ""


# ---------------------------------------------------------------
# quantize_model helper tests
# ---------------------------------------------------------------

def test_get_model_size_returns_positive():
    """get_model_size_mb should return a positive value for any model."""
    from optimization.quantize_model import get_model_size_mb
    import torch

    # Create a tiny dummy model
    model = torch.nn.Linear(10, 5)
    size = get_model_size_mb(model)
    assert size > 0, "Model size must be positive"
    assert size < 1.0, "A tiny linear layer should be well under 1MB"
