#!/usr/bin/env python3
"""
tests/test_config.py
Unit tests for the pipeline configuration loader.
"""

import yaml
from pathlib import Path


def test_config_file_exists():
    """Verify that the default config file exists."""
    config_path = Path("configs/config.yaml")
    assert config_path.exists(), f"Config file not found: {config_path}"


def test_config_loads_valid_yaml():
    """Verify that config.yaml is valid YAML and contains expected top-level keys."""
    config_path = Path("configs/config.yaml")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert isinstance(config, dict), "Config should be a dictionary"
    assert "project" in config, "Config must have 'project' section"
    assert "asr" in config, "Config must have 'asr' section"
    assert "ser" in config, "Config must have 'ser' section"
    assert "nlp" in config, "Config must have 'nlp' section"
    assert "preprocessing" in config, "Config must have 'preprocessing' section"
    assert "pitch" in config, "Config must have 'pitch' section"
    assert "sarcasm" in config, "Config must have 'sarcasm' section"


def test_config_model_names():
    """Verify model names are specified correctly."""
    config_path = Path("configs/config.yaml")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert config["asr"]["model_name"] == "openai/whisper-tiny"
    assert config["ser"]["model_name"] == "superb/wav2vec2-base-superb-er"
    assert config["nlp"]["model_name"] == "distilbert-base-uncased-finetuned-sst-2-english"


def test_config_preprocessing_defaults():
    """Verify preprocessing defaults are sensible."""
    config_path = Path("configs/config.yaml")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    preprocess = config["preprocessing"]
    assert preprocess["sample_rate"] == 16000
    assert preprocess["method"] in ["none", "wiener", "spectral_subtraction"]
    assert preprocess["snr_threshold_db"] > 0
    assert preprocess["vad"]["sensitivity"] in range(4)  # 0-3


def test_config_pitch_range():
    """Verify pitch estimation frequency bounds are valid."""
    config_path = Path("configs/config.yaml")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pitch = config["pitch"]
    assert pitch["fmin"] > 0, "fmin must be positive"
    assert pitch["fmax"] > pitch["fmin"], "fmax must be greater than fmin"
    assert pitch["fmin"] >= 50, "fmin should be at least 50 Hz for human voice"
    assert pitch["fmax"] <= 600, "fmax should be at most 600 Hz for human voice"
