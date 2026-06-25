#!/usr/bin/env python3
"""
main.py — Multimodal Audio Analysis Pipeline Entry Point
=========================================================
Role 1: Pipeline Architect & DevOps

Integrates all team modules into a unified pipeline:
  1. Load raw audio file                  (utils.audio_utils)
  2. Apply configurable preprocessing     (preprocessing.denoise)
  3. Route: Denoised audio → ASR          (transformers pipeline — Whisper)
  4. Route: Normalized audio → SER        (transformers pipeline — Wav2Vec2)
  5. Feed ASR transcription → NLP         (transformers pipeline — DistilBERT)
  6. Execute sarcasm detection heuristic
  7. Output structured results

Usage:
  python main.py --config configs/config.yaml --audio data/emotion_samples/03-01-05-02-01-01-01.wav
  python main.py --audio recording.wav --method spectral_subtraction
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project module imports — integrating team deliverables
# ---------------------------------------------------------------------------
from utils.config_loader import load_config                      # Role 1: utils/
from utils.audio_utils import load_audio, normalize_volume, trim_silence  # Role 1: utils/
from preprocessing.denoise import (                              # Role 2: preprocessing/
    preprocess_none,
    preprocess_wiener,
    preprocess_spectral_subtraction,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# =====================================================================
# Model Loading
# =====================================================================
def load_models(config: dict) -> tuple:
    """Load all three neural models from config."""
    import torch
    from transformers import pipeline as hf_pipeline

    device = 0 if torch.cuda.is_available() else -1
    device_name = "CUDA" if torch.cuda.is_available() else "CPU"
    logger.info(f"Loading models on {device_name}...")

    asr_model = config.get("asr", {}).get("model_name", "openai/whisper-tiny")
    ser_model = config.get("ser", {}).get("model_name", "superb/wav2vec2-base-superb-er")
    nlp_model = config.get("nlp", {}).get(
        "model_name", "distilbert-base-uncased-finetuned-sst-2-english"
    )

    logger.info(f"  ASR model: {asr_model}")
    asr_pipe = hf_pipeline("automatic-speech-recognition", model=asr_model, device=device)

    logger.info(f"  SER model: {ser_model}")
    ser_pipe = hf_pipeline("audio-classification", model=ser_model, device=device)

    logger.info(f"  NLP model: {nlp_model}")
    nlp_pipe = hf_pipeline("sentiment-analysis", model=nlp_model, device=device)

    logger.info("All models loaded successfully.")
    return asr_pipe, ser_pipe, nlp_pipe


# =====================================================================
# Sarcasm Detection Heuristic
# =====================================================================
SER_LABELS = {"neu": "neutral", "hap": "happy", "ang": "angry", "sad": "sad"}


def detect_sarcasm(text_sentiment: str, voice_emotion: str) -> tuple:
    """
    Detect sarcasm by comparing text sentiment vs vocal emotion.

    Returns:
        (is_sarcastic: bool, reason: str)
    """
    if text_sentiment == "positive" and voice_emotion in ["angry", "sad"]:
        return True, f"Positive words spoken with a negative voice ({voice_emotion})."
    elif text_sentiment == "negative" and voice_emotion == "happy":
        return True, "Negative words spoken with a happy/excited voice."
    elif text_sentiment == "neutral" and voice_emotion in ["angry", "happy", "sad"]:
        return True, f"Neutral statement spoken with an emotional voice ({voice_emotion})."
    return False, "Normal vocal alignment."


# =====================================================================
# Main Pipeline
# =====================================================================
def run_pipeline(audio_path: str, config: dict, method_override: str = None) -> dict:
    """
    Execute the full multimodal pipeline on a single audio file.

    Pipeline routes:
      - Denoised Stream → ASR (Whisper) → Sentiment (DistilBERT)
      - Normalized Stream → SER (Wav2Vec2)
      - Cross-modal comparison → Sarcasm Detection
    """
    # --- Load audio (using utils/) ---
    sr = config.get("preprocessing", {}).get("sample_rate", 16000)
    audio, file_sr = load_audio(str(audio_path), target_sr=sr)

    # --- Determine preprocessing method ---
    preprocess_cfg = config.get("preprocessing", {})
    method = method_override or preprocess_cfg.get("method", "none")

    # --- ROUTE 1: Denoised stream for ASR (using preprocessing/) ---
    logger.info(f"[Route 1] Denoising audio with method: {method}")
    if method == "wiener":
        wiener_size = preprocess_cfg.get("wiener", {}).get("size", 3)
        audio_denoised = preprocess_wiener(audio, size=wiener_size)
    elif method == "spectral_subtraction":
        ss_cfg = preprocess_cfg.get("spectral_subtraction", {})
        audio_denoised = preprocess_spectral_subtraction(
            audio, sr=sr, alpha=ss_cfg.get("alpha", 2.0), beta=ss_cfg.get("beta", 0.01)
        )
    else:
        audio_denoised = preprocess_none(audio)

    # --- ROUTE 2: Normalized stream for SER (using utils/) ---
    logger.info("[Route 2] Normalizing audio for SER (trim silence + peak normalize)")
    trim_cfg = preprocess_cfg.get("trim_silence", {})
    audio_normalized = trim_silence(audio, top_db=trim_cfg.get("top_db", 30))
    audio_normalized = normalize_volume(audio_normalized)

    # --- Load models ---
    asr_pipe, ser_pipe, nlp_pipe = load_models(config)

    start_time = time.time()

    # --- Step 1: ASR on denoised stream ---
    logger.info("[Step 1/4] Running ASR (Whisper) on denoised audio...")
    default_lang = config.get("asr", {}).get("default_language", "en")
    asr_result = asr_pipe(audio_denoised, generate_kwargs={"language": default_lang})
    transcription = asr_result["text"].strip()
    logger.info(f'  Transcription: "{transcription}"')

    # --- Step 2: Text sentiment on transcription ---
    logger.info("[Step 2/4] Running text sentiment analysis (DistilBERT)...")
    if transcription:
        sent_result = nlp_pipe(transcription)[0]
        text_sentiment = sent_result["label"].lower()
        sent_confidence = sent_result["score"]
    else:
        text_sentiment = "neutral"
        sent_confidence = 1.0
    logger.info(f"  Sentiment: {text_sentiment.upper()} ({sent_confidence:.1%})")

    # --- Step 3: SER on normalized stream ---
    logger.info("[Step 3/4] Running SER (Wav2Vec2) on normalized audio...")
    ser_result = ser_pipe(audio_normalized)
    raw_emotion = ser_result[0]["label"]
    voice_emotion = SER_LABELS.get(raw_emotion, raw_emotion)
    ser_confidence = ser_result[0]["score"]
    logger.info(f"  Vocal Emotion: {voice_emotion.upper()} ({ser_confidence:.1%})")

    # --- Step 4: Sarcasm detection ---
    sarcasm_enabled = config.get("sarcasm", {}).get("enabled", True)
    if sarcasm_enabled:
        logger.info("[Step 4/4] Running sarcasm detection...")
        is_sarcastic, sarcasm_reason = detect_sarcasm(text_sentiment, voice_emotion)
    else:
        is_sarcastic = False
        sarcasm_reason = "Sarcasm detection disabled."

    latency_ms = (time.time() - start_time) * 1000

    # --- Build results ---
    results = {
        "file": str(audio_path),
        "duration_s": round(len(audio) / file_sr, 2),
        "preprocessing_method": method,
        "transcription": transcription,
        "text_sentiment": text_sentiment,
        "text_sentiment_confidence": round(sent_confidence, 4),
        "vocal_emotion": voice_emotion,
        "vocal_emotion_confidence": round(ser_confidence, 4),
        "vocal_emotion_all": [
            {"label": SER_LABELS.get(p["label"], p["label"]), "score": round(p["score"], 4)}
            for p in ser_result
        ],
        "is_sarcastic": is_sarcastic,
        "sarcasm_reason": sarcasm_reason,
        "pipeline_latency_ms": round(latency_ms, 1),
    }

    # --- Print report ---
    print("\n" + "=" * 64)
    print("  MULTIMODAL PIPELINE RESULTS")
    print("=" * 64)
    print(f"  File             : {Path(audio_path).name}")
    print(f"  Duration         : {results['duration_s']}s")
    print(f"  Preprocessing    : {method}")
    print("-" * 64)
    print(f'  Transcription    : "{transcription}"')
    print(f"  Text Sentiment   : {text_sentiment.upper()} ({sent_confidence:.1%})")
    print(f"  Vocal Emotion    : {voice_emotion.upper()} ({ser_confidence:.1%})")
    print("-" * 64)
    if is_sarcastic:
        print(f"  ⚠️  SARCASM DETECTED!")
        print(f"  Reason           : {sarcasm_reason}")
    else:
        print(f"  ✅ Normal Speech")
        print(f"  Reason           : {sarcasm_reason}")
    print("-" * 64)
    print(f"  Pipeline Latency : {latency_ms:.0f} ms")
    print("=" * 64 + "\n")

    return results


# =====================================================================
# CLI Entry Point
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Audio Analysis Pipeline: ASR + Sentiment + SER + Sarcasm Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --audio data/emotion_samples/03-01-05-02-01-01-01.wav
  python main.py --audio recording.wav --method spectral_subtraction
  python main.py --audio recording.wav --config configs/config.yaml --output results/output.json
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to input WAV audio file (16kHz mono recommended)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["none", "wiener", "spectral_subtraction"],
        default=None,
        help="Override preprocessing method from config",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results as JSON (optional)",
    )
    args = parser.parse_args()

    # Load config (using utils/)
    config = load_config(args.config)
    project_name = config.get("project", {}).get("name", "unknown")
    logger.info(f"Loaded config for project: {project_name}")

    # Check audio file
    if args.audio is None:
        logger.error("No audio file specified. Use --audio <path> to provide an input file.")
        logger.info("Run 'python main.py --help' for usage information.")
        sys.exit(1)

    # Run pipeline
    results = run_pipeline(args.audio, config, method_override=args.method)

    # Optionally save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
