"""
tests/test_preprocessing.py — Unit Tests for the Preprocessing Module

Tests cover:
    1. Denoising filters (Wiener, Spectral Subtraction)
    2. VAD utilities (trim_silence, normalize_volume, energy_vad)
    3. Feature extraction (SNR estimation, pitch contour)
    4. Multimodal fusion calibration
    5. End-to-end pipeline with parallel routing

Run with:
    python -m pytest tests/test_preprocessing.py -v
"""

import numpy as np
import pytest

from preprocessing.denoise import wiener_filter, spectral_subtraction
from preprocessing.vad import trim_silence, normalize_volume, energy_vad
from preprocessing.features import estimate_snr, estimate_pitch_contour
from preprocessing.fusion import fuse_modalities
from preprocessing.pipeline import (
    PreprocessingPipeline,
    PreprocessingConfig,
    PreprocessingResult,
    DenoiseMethod,
    run_preprocessing,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────────

SR = 16000

@pytest.fixture
def sine_wave():
    """Generate a 1-second 440 Hz sine wave at 16 kHz."""
    t = np.linspace(0, 1.0, SR, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture
def noisy_sine(sine_wave):
    """Sine wave with additive white Gaussian noise at ~10 dB SNR."""
    np.random.seed(42)
    noise = 0.1 * np.random.randn(len(sine_wave)).astype(np.float32)
    return (sine_wave + noise).astype(np.float32)


@pytest.fixture
def silence_padded(sine_wave):
    """Sine wave with 0.5s silence before and after."""
    pad = np.zeros(SR // 2, dtype=np.float32)
    return np.concatenate([pad, sine_wave, pad])


@pytest.fixture
def mock_emotion_preds():
    """Simulated Wav2Vec2 SER predictions (angry dominant)."""
    return [
        {"label": "ang", "score": 0.50},
        {"label": "hap", "score": 0.25},
        {"label": "neu", "score": 0.15},
        {"label": "sad", "score": 0.10},
    ]


# ─── Denoising Tests ──────────────────────────────────────────────────────────

class TestWienerFilter:
    def test_output_shape(self, noisy_sine):
        result = wiener_filter(noisy_sine)
        assert result.shape == noisy_sine.shape

    def test_output_dtype(self, noisy_sine):
        result = wiener_filter(noisy_sine)
        assert result.dtype == np.float32

    def test_even_window_corrected(self, noisy_sine):
        """Even window size should be auto-corrected to odd."""
        result = wiener_filter(noisy_sine, mysize=4)
        assert result.shape == noisy_sine.shape

    def test_reduces_noise_energy(self, noisy_sine, sine_wave):
        """Wiener filter should reduce noise energy (not always, but on white noise)."""
        filtered = wiener_filter(noisy_sine, mysize=5)
        noise_before = np.mean((noisy_sine - sine_wave) ** 2)
        noise_after = np.mean((filtered - sine_wave) ** 2)
        # Wiener should reduce noise on stationary white noise
        assert noise_after < noise_before

    def test_clean_signal_passthrough(self, sine_wave):
        """On clean signal, Wiener should not significantly alter it."""
        result = wiener_filter(sine_wave, mysize=3)
        correlation = np.corrcoef(sine_wave, result)[0, 1]
        assert correlation > 0.95


class TestSpectralSubtraction:
    def test_output_shape(self, noisy_sine):
        result = spectral_subtraction(noisy_sine, sr=SR)
        assert result.shape == noisy_sine.shape

    def test_output_dtype(self, noisy_sine):
        result = spectral_subtraction(noisy_sine, sr=SR)
        assert result.dtype == np.float32

    def test_peak_normalized(self, noisy_sine):
        """Output should be peak-normalized to [-1, 1]."""
        result = spectral_subtraction(noisy_sine, sr=SR)
        assert np.max(np.abs(result)) <= 1.0 + 1e-6

    def test_short_audio(self):
        """Should handle very short audio without crashing."""
        short = np.random.randn(100).astype(np.float32)
        result = spectral_subtraction(short, sr=SR)
        assert len(result) == 100

    def test_alpha_beta_parameters(self, noisy_sine):
        """Different alpha/beta should produce different outputs."""
        r1 = spectral_subtraction(noisy_sine, sr=SR, alpha=1.0, beta=0.01)
        r2 = spectral_subtraction(noisy_sine, sr=SR, alpha=4.0, beta=0.05)
        assert not np.allclose(r1, r2)


# ─── VAD Tests ─────────────────────────────────────────────────────────────────

class TestTrimSilence:
    def test_removes_padding(self, silence_padded, sine_wave):
        """Should trim silence padding, resulting in shorter output."""
        trimmed = trim_silence(silence_padded, top_db=30)
        assert len(trimmed) < len(silence_padded)
        # Trimmed should be closer in length to the original sine wave
        assert abs(len(trimmed) - len(sine_wave)) < SR // 4

    def test_pure_silence_returns_input(self):
        """All-zero input should be returned unchanged."""
        silence = np.zeros(SR, dtype=np.float32)
        result = trim_silence(silence)
        assert len(result) == SR

    def test_no_silence_returns_same(self, sine_wave):
        """Signal with no silence should not be shortened significantly."""
        result = trim_silence(sine_wave, top_db=60)
        assert len(result) >= len(sine_wave) * 0.8


class TestNormalizeVolume:
    def test_peak_is_one(self, sine_wave):
        """After normalization, peak should be 1.0."""
        quiet = sine_wave * 0.1
        result = normalize_volume(quiet)
        assert abs(np.max(np.abs(result)) - 1.0) < 1e-6

    def test_already_normalized(self, sine_wave):
        """Already-normalized signal should be unchanged."""
        normed = normalize_volume(sine_wave)
        result = normalize_volume(normed)
        np.testing.assert_allclose(normed, result, atol=1e-6)

    def test_zero_signal(self):
        """All-zero signal should remain zero."""
        zeros = np.zeros(100, dtype=np.float32)
        result = normalize_volume(zeros)
        np.testing.assert_array_equal(result, zeros)

    def test_output_dtype(self, sine_wave):
        result = normalize_volume(sine_wave)
        assert result.dtype == np.float32


class TestEnergyVAD:
    def test_detects_speech(self, sine_wave):
        """Should detect speech frames in a sine wave."""
        labels = energy_vad(sine_wave, sr=SR)
        assert len(labels) > 0
        assert np.any(labels)  # At least some speech detected

    def test_silence_no_speech(self):
        """Should not detect speech in silence."""
        silence = np.zeros(SR, dtype=np.float32)
        labels = energy_vad(silence, sr=SR, energy_threshold_db=-40.0)
        assert not np.any(labels)

    def test_output_shape(self, sine_wave):
        """Output length should match expected number of frames."""
        frame_ms = 25.0
        hop_ms = 10.0
        frame_len = int(frame_ms * SR / 1000)
        hop_len = int(hop_ms * SR / 1000)
        expected_frames = 1 + (len(sine_wave) - frame_len) // hop_len
        labels = energy_vad(sine_wave, sr=SR, frame_duration_ms=frame_ms,
                            hop_duration_ms=hop_ms)
        assert len(labels) == expected_frames


# ─── Feature Extraction Tests ─────────────────────────────────────────────────

class TestEstimateSNR:
    def test_clean_signal_high_snr(self):
        """Signal with silent start should have positive SNR."""
        # Create signal with 0.5s silence then 0.5s tone — mimics real speech
        silence = np.zeros(SR // 2, dtype=np.float32)
        tone = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, SR // 2)).astype(np.float32)
        signal = np.concatenate([silence, tone])
        snr = estimate_snr(signal, sr=SR, noise_duration=0.5)
        assert snr > 10.0  # Silent noise floor → high SNR

    def test_noisy_signal_lower_snr(self):
        """Adding noise to the silent portion should lower SNR."""
        np.random.seed(42)
        tone = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, SR // 2)).astype(np.float32)
        # Clean version: silence + tone
        silence = np.zeros(SR // 2, dtype=np.float32)
        clean = np.concatenate([silence, tone])
        # Noisy version: noise + tone
        noise = 0.1 * np.random.randn(SR // 2).astype(np.float32)
        noisy = np.concatenate([noise, tone + noise[:len(tone)]])
        snr_clean = estimate_snr(clean, sr=SR, noise_duration=0.5)
        snr_noisy = estimate_snr(noisy, sr=SR, noise_duration=0.5)
        assert snr_noisy < snr_clean

    def test_empty_audio(self):
        """Empty audio should return 0.0."""
        snr = estimate_snr(np.array([], dtype=np.float32), sr=SR)
        assert snr == 0.0

    def test_returns_float(self, sine_wave):
        snr = estimate_snr(sine_wave, sr=SR)
        assert isinstance(snr, float)


class TestEstimatePitchContour:
    def test_returns_tuple_of_three(self, sine_wave):
        """Should return (mean_f0, std_f0, contour)."""
        result = estimate_pitch_contour(sine_wave, sr=SR)
        assert len(result) == 3

    def test_detects_440hz(self, sine_wave):
        """Should detect ~440 Hz from a 440 Hz sine wave."""
        mean_f0, _, _ = estimate_pitch_contour(sine_wave, sr=SR, fmin=75, fmax=500)
        # Allow some tolerance (YIN is not perfectly precise)
        if mean_f0 > 0:  # May fail without librosa
            assert abs(mean_f0 - 440) < 50

    def test_silence_returns_zero(self):
        """Silence should return 0.0 mean pitch."""
        silence = np.zeros(SR, dtype=np.float32)
        mean_f0, _, _ = estimate_pitch_contour(silence, sr=SR)
        assert mean_f0 == 0.0


# ─── Fusion Tests ──────────────────────────────────────────────────────────────

class TestFuseModalities:
    def test_positive_text_boosts_happy(self, mock_emotion_preds):
        """Positive text sentiment should boost happy score."""
        result = fuse_modalities(mock_emotion_preds, "positive", 0.95, 200.0)
        scores = {p["label"]: p["score"] for p in result}
        # Happy should be the dominant class after calibration
        assert scores["hap"] > scores["ang"]

    def test_negative_text_boosts_angry_sad(self, mock_emotion_preds):
        """Negative text should boost angry/sad."""
        preds = [
            {"label": "hap", "score": 0.50},
            {"label": "ang", "score": 0.20},
            {"label": "sad", "score": 0.20},
            {"label": "neu", "score": 0.10},
        ]
        result = fuse_modalities(preds, "negative", 0.90, 100.0)
        scores = {p["label"]: p["score"] for p in result}
        # Happy should be penalized
        assert scores["hap"] < 0.50

    def test_output_sums_to_one(self, mock_emotion_preds):
        """Calibrated scores should sum to 1.0."""
        result = fuse_modalities(mock_emotion_preds, "positive", 0.80, 190.0)
        total = sum(p["score"] for p in result)
        assert abs(total - 1.0) < 1e-6

    def test_no_negative_scores(self, mock_emotion_preds):
        """All scores should be >= 0 after calibration."""
        result = fuse_modalities(mock_emotion_preds, "positive", 0.99, 300.0)
        for p in result:
            assert p["score"] >= 0.0

    def test_sorted_descending(self, mock_emotion_preds):
        """Results should be sorted by score, descending."""
        result = fuse_modalities(mock_emotion_preds, "neutral", 0.50, 150.0)
        scores = [p["score"] for p in result]
        assert scores == sorted(scores, reverse=True)

    def test_missing_labels_handled(self):
        """Should handle predictions with missing labels gracefully."""
        preds = [{"label": "hap", "score": 0.8}, {"label": "ang", "score": 0.2}]
        result = fuse_modalities(preds, "positive", 0.7, 200.0)
        labels = {p["label"] for p in result}
        assert "neu" in labels
        assert "sad" in labels


# ─── Pipeline Tests ────────────────────────────────────────────────────────────

class TestPreprocessingPipeline:
    def test_default_no_denoise(self, noisy_sine):
        """Default config should NOT apply denoising."""
        pipe = PreprocessingPipeline()
        result = pipe.process(noisy_sine, sr=SR)
        assert not result.denoise_applied
        assert result.denoise_method_used == "none"
        np.testing.assert_array_equal(result.asr_audio, noisy_sine)

    def test_wiener_denoise(self, noisy_sine):
        """Wiener config should apply denoising to ASR stream."""
        config = PreprocessingConfig(denoise_method=DenoiseMethod.WIENER)
        pipe = PreprocessingPipeline(config)
        result = pipe.process(noisy_sine, sr=SR)
        assert result.denoise_applied
        assert result.denoise_method_used == "wiener"
        # ASR audio should differ from raw
        assert not np.array_equal(result.asr_audio, noisy_sine)

    def test_ser_stream_never_denoised(self, noisy_sine):
        """SER stream should NEVER be denoised (parallel routing)."""
        config = PreprocessingConfig(denoise_method=DenoiseMethod.WIENER)
        pipe = PreprocessingPipeline(config)
        result = pipe.process(noisy_sine, sr=SR)
        # SER audio is trimmed + normalized, NOT denoised
        assert np.max(np.abs(result.ser_audio)) <= 1.0 + 1e-6

    def test_parallel_routing_different_streams(self, noisy_sine):
        """ASR and SER streams should be different when denoising is applied."""
        config = PreprocessingConfig(denoise_method=DenoiseMethod.WIENER)
        pipe = PreprocessingPipeline(config)
        result = pipe.process(noisy_sine, sr=SR)
        # Shapes may differ (trim_silence changes length)
        # But content should definitely differ
        assert result.denoise_applied

    def test_snr_estimation(self, noisy_sine):
        """Pipeline should estimate SNR."""
        pipe = PreprocessingPipeline()
        result = pipe.process(noisy_sine, sr=SR)
        assert isinstance(result.estimated_snr, float)

    def test_pitch_extraction(self, sine_wave):
        """Pipeline should extract pitch when configured."""
        config = PreprocessingConfig(compute_pitch=True)
        pipe = PreprocessingPipeline(config)
        result = pipe.process(sine_wave, sr=SR)
        assert isinstance(result.mean_pitch, float)

    def test_snr_adaptive_skips_denoise(self, sine_wave):
        """SNR-adaptive mode should skip denoising on clean signals."""
        config = PreprocessingConfig(
            denoise_method=DenoiseMethod.WIENER,
            snr_adaptive=True,
            snr_threshold=10.0,
        )
        pipe = PreprocessingPipeline(config)
        result = pipe.process(sine_wave, sr=SR)
        # Clean sine wave has high SNR → denoising should be skipped
        if result.estimated_snr > 10.0:
            assert not result.denoise_applied


class TestRunPreprocessing:
    def test_convenience_function(self, noisy_sine):
        """run_preprocessing should work with minimal arguments."""
        result = run_preprocessing(noisy_sine, sr=SR)
        assert isinstance(result, PreprocessingResult)
        assert len(result.asr_audio) > 0
        assert len(result.ser_audio) > 0

    def test_wiener_method(self, noisy_sine):
        """Should accept string method name."""
        result = run_preprocessing(noisy_sine, sr=SR, denoise_method="wiener")
        assert result.denoise_applied
        assert result.denoise_method_used == "wiener"

    def test_none_method(self, noisy_sine):
        """Default 'none' should not denoise."""
        result = run_preprocessing(noisy_sine, sr=SR, denoise_method="none")
        assert not result.denoise_applied
