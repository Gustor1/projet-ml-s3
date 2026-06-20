#!/usr/bin/env python3
"""
scripts/cache_models.py
Pre-downloads all Hugging Face model weights required by the multimodal pipeline.

This script is executed during Docker image build (RUN step in Dockerfile)
to ensure fully offline inference inside the container.

Models cached:
  1. openai/whisper-tiny           — ASR (Speech-to-Text)
  2. superb/wav2vec2-base-superb-er — SER (Speech Emotion Recognition)
  3. distilbert-base-uncased-finetuned-sst-2-english — NLP (Text Sentiment)

Usage:
  python scripts/cache_models.py
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Models to cache — maps a human-readable key to (HF model ID, pipeline task)
MODELS_TO_CACHE = {
    "ASR (Whisper-tiny)": {
        "model_id": "openai/whisper-tiny",
        "task": "automatic-speech-recognition",
    },
    "SER (Wav2Vec2-ER)": {
        "model_id": "superb/wav2vec2-base-superb-er",
        "task": "audio-classification",
    },
    "NLP (DistilBERT Sentiment)": {
        "model_id": "distilbert-base-uncased-finetuned-sst-2-english",
        "task": "sentiment-analysis",
    },
}


def cache_all_models():
    """Download and cache all pipeline models from Hugging Face Hub."""
    from transformers import pipeline as hf_pipeline

    success_count = 0
    total = len(MODELS_TO_CACHE)

    for name, info in MODELS_TO_CACHE.items():
        model_id = info["model_id"]
        task = info["task"]
        logger.info(f"[{success_count + 1}/{total}] Caching {name}: {model_id} ...")
        try:
            # Creating a pipeline forces the download of model weights + tokenizer
            _ = hf_pipeline(task, model=model_id)
            logger.info(f"  ✓ {name} cached successfully.")
            success_count += 1
        except Exception as e:
            logger.error(f"  ✗ Failed to cache {name}: {e}")

    logger.info(f"\nModel caching complete: {success_count}/{total} models cached.")
    if success_count < total:
        logger.warning("Some models failed to cache. The container may not run fully offline.")
        sys.exit(1)


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  Hugging Face Model Caching Script")
    logger.info("  Pre-downloading weights for offline Docker execution")
    logger.info("=" * 60)
    cache_all_models()
