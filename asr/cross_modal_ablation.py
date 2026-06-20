#!/usr/bin/env python3
"""
asr/cross_modal_ablation.py
Cross-Modal Ablation Study — Role 3 Deliverable

Evaluates how ASR transcription errors (from different Whisper model sizes)
cascade into downstream text sentiment predictions and sarcasm detection.

Pipeline per audio file:
    1. Whisper ASR  →  transcription (+ ground truth reference)
    2. DistilBERT   →  text sentiment on ASR output vs. ground truth
    3. Wav2Vec2-SER →  vocal emotion from raw audio
    4. Sarcasm logic →  mismatch detection (voice ≠ text sentiment)

Metrics reported per model size:
    - Mean WER / CER
    - Sentiment Flip Rate  (ASR text vs. reference text give different sentiment)
    - Sarcasm False Positive Rate  (sarcasm detected only due to ASR error)
    - Sarcasm False Negative Rate  (sarcasm missed only due to ASR error)
    - Agreement Rate  (sarcasm verdict identical with ASR vs. ground truth)

Usage:
    python -m asr.cross_modal_ablation --metadata data/emotion_metadata.json
    python -m asr.cross_modal_ablation --metadata data/emotion_metadata.json --models tiny base small
"""

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import pipeline as hf_pipeline

import jiwer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WHISPER_SIZES = ["tiny", "base", "small"]

SER_LABELS = {
    "neu": "neutral",
    "hap": "happy",
    "ang": "angry",
    "sad": "sad",
}

# WER / CER transforms (consistent with asr/evaluator.py)
WER_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])

CER_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfChars(),
])

# ---------------------------------------------------------------------------
# Sarcasm detection logic (mirrors experiments/sarcasm_detector.py)
# ---------------------------------------------------------------------------

def detect_sarcasm(text_sentiment: str, voice_emotion: str) -> dict:
    """
    Detect mismatch between text sentiment and vocal emotion.
    Returns {"is_sarcastic": bool, "reason": str}.
    """
    is_sarcastic = False
    reason = "Normal vocal alignment."

    # Positive words + negative voice
    if text_sentiment == "positive" and voice_emotion in ("angry", "sad"):
        is_sarcastic = True
        reason = f"Positive words spoken with a negative voice ({voice_emotion})."
    # Negative words + happy voice
    elif text_sentiment == "negative" and voice_emotion == "happy":
        is_sarcastic = True
        reason = "Negative words spoken with a happy/excited voice."
    # Neutral words + strong emotional voice
    elif text_sentiment == "neutral" and voice_emotion in ("angry", "happy", "sad"):
        is_sarcastic = True
        reason = f"Neutral statement spoken with an emotional voice ({voice_emotion})."

    return {"is_sarcastic": is_sarcastic, "reason": reason}


def get_text_sentiment(sentiment_pipe, text: str) -> tuple:
    """
    Returns (label, confidence) where label ∈ {"positive", "negative", "neutral"}.
    Empty/very short texts are classified as "neutral".
    """
    if not text or len(text.strip()) < 3:
        return "neutral", 1.0
    result = sentiment_pipe(text[:512])[0]  # DistilBERT max 512 tokens
    label = result["label"].lower()  # 'POSITIVE' / 'NEGATIVE'
    return label, round(result["score"], 4)


# ---------------------------------------------------------------------------
# Core ablation runner
# ---------------------------------------------------------------------------

def run_cross_modal_ablation(
    metadata_path: str,
    models: list = None,
    output_dir: str = "results",
    reference_field: str = "transcription",
) -> dict:
    """
    Run the full cross-modal ablation study.

    Parameters
    ----------
    metadata_path : str
        Path to a JSON file with entries like:
        [{"file_path": "...", "transcription": "...", "emotion": "happy"}, ...]
        If 'transcription' is missing, the study uses only SER and skips WER.
    models : list
        Whisper model sizes to benchmark. Default: ["tiny", "base", "small"]
    output_dir : str
        Directory for CSV output.
    reference_field : str
        JSON key for the ground-truth transcription.

    Returns
    -------
    dict with summary statistics per model size.
    """
    if models is None:
        models = WHISPER_SIZES

    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Filter to files that exist on disk
    valid_entries = [e for e in metadata if Path(e["file_path"]).exists()]
    skipped = len(metadata) - len(valid_entries)
    if skipped:
        logger.warning(f"Skipped {skipped} entries (files not found on disk).")
    if not valid_entries:
        raise RuntimeError("No valid audio files found in metadata.")

    logger.info(f"Cross-modal ablation: {len(valid_entries)} audio files × {len(models)} Whisper sizes")

    # ------------------------------------------------------------------
    # Load shared models (SER + Sentiment — loaded once, shared across)
    # ------------------------------------------------------------------
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device_str}")

    logger.info("Loading SER model (superb/wav2vec2-base-superb-er)...")
    ser_pipe = hf_pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", device=device_str)

    logger.info("Loading sentiment model (distilbert-base-uncased-finetuned-sst-2-english)...")
    sentiment_pipe = hf_pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device_str)

    # ------------------------------------------------------------------
    # Pre-compute SER + ground-truth sentiment (model-independent)
    # ------------------------------------------------------------------
    logger.info("Pre-computing vocal emotions and ground-truth sentiments...")
    file_context = {}  # keyed by file_path

    for entry in valid_entries:
        fpath = entry["file_path"]
        audio, sr = sf.read(fpath, dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # SER
        ser_result = ser_pipe(audio)
        raw_label = ser_result[0]["label"]
        voice_emotion = SER_LABELS.get(raw_label, raw_label)
        ser_conf = ser_result[0]["score"]

        # Ground-truth text sentiment (if reference text exists)
        ref_text = entry.get(reference_field, "")
        gt_sentiment, gt_conf = get_text_sentiment(sentiment_pipe, ref_text)
        gt_sarcasm = detect_sarcasm(gt_sentiment, voice_emotion)

        file_context[fpath] = {
            "audio": audio,
            "voice_emotion": voice_emotion,
            "ser_confidence": round(ser_conf, 4),
            "ref_text": ref_text,
            "gt_sentiment": gt_sentiment,
            "gt_sentiment_conf": gt_conf,
            "gt_is_sarcastic": gt_sarcasm["is_sarcastic"],
            "gt_sarcasm_reason": gt_sarcasm["reason"],
            "true_emotion": entry.get("emotion", "unknown"),
        }

    # ------------------------------------------------------------------
    # Per-model evaluation
    # ------------------------------------------------------------------
    all_rows = []
    summary_per_model = {}

    for model_size in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"[Ablation] Whisper-{model_size}")
        logger.info(f"{'='*60}")

        asr_pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=f"openai/whisper-{model_size}",
            device=device_str,
        )

        model_rows = []
        t_start = time.time()

        for i, entry in enumerate(valid_entries):
            fpath = entry["file_path"]
            ctx = file_context[fpath]

            # ASR transcription
            asr_result = asr_pipe(ctx["audio"], generate_kwargs={"language": "en"})
            asr_text = asr_result["text"].strip()

            # WER / CER (only if ground-truth reference exists)
            ref = ctx["ref_text"]
            if ref:
                wer = round(jiwer.wer(ref, asr_text,
                                       reference_transform=WER_TRANSFORM,
                                       hypothesis_transform=WER_TRANSFORM), 4)
                cer = round(jiwer.cer(ref, asr_text,
                                       reference_transform=CER_TRANSFORM,
                                       hypothesis_transform=CER_TRANSFORM), 4)
            else:
                wer, cer = -1.0, -1.0

            # Sentiment on ASR text
            asr_sentiment, asr_sent_conf = get_text_sentiment(sentiment_pipe, asr_text)

            # Sarcasm on ASR text
            asr_sarcasm = detect_sarcasm(asr_sentiment, ctx["voice_emotion"])

            # Comparison flags
            sentiment_flipped = (asr_sentiment != ctx["gt_sentiment"])
            sarcasm_false_pos = (asr_sarcasm["is_sarcastic"] and not ctx["gt_is_sarcastic"])
            sarcasm_false_neg = (not asr_sarcasm["is_sarcastic"] and ctx["gt_is_sarcastic"])
            sarcasm_agree = (asr_sarcasm["is_sarcastic"] == ctx["gt_is_sarcastic"])

            row = {
                "model": f"whisper-{model_size}",
                "file": os.path.basename(fpath),
                "true_emotion": ctx["true_emotion"],
                "voice_emotion": ctx["voice_emotion"],
                "ser_confidence": ctx["ser_confidence"],
                "ref_text": ref,
                "asr_text": asr_text,
                "wer": wer,
                "cer": cer,
                "gt_sentiment": ctx["gt_sentiment"],
                "asr_sentiment": asr_sentiment,
                "sentiment_flipped": int(sentiment_flipped),
                "gt_sarcastic": int(ctx["gt_is_sarcastic"]),
                "asr_sarcastic": int(asr_sarcasm["is_sarcastic"]),
                "sarcasm_false_positive": int(sarcasm_false_pos),
                "sarcasm_false_negative": int(sarcasm_false_neg),
                "sarcasm_agree": int(sarcasm_agree),
                "asr_sarcasm_reason": asr_sarcasm["reason"],
            }
            model_rows.append(row)

            if (i + 1) % 10 == 0 or (i + 1) == len(valid_entries):
                logger.info(f"  [{i+1}/{len(valid_entries)}] processed")

        elapsed = round(time.time() - t_start, 2)
        all_rows.extend(model_rows)

        # Compute summary
        n = len(model_rows)
        valid_wers = [r["wer"] for r in model_rows if r["wer"] >= 0]
        valid_cers = [r["cer"] for r in model_rows if r["cer"] >= 0]
        n_flips = sum(r["sentiment_flipped"] for r in model_rows)
        n_fp = sum(r["sarcasm_false_positive"] for r in model_rows)
        n_fn = sum(r["sarcasm_false_negative"] for r in model_rows)
        n_agree = sum(r["sarcasm_agree"] for r in model_rows)

        summary = {
            "model": f"whisper-{model_size}",
            "num_samples": n,
            "mean_wer": round(np.mean(valid_wers), 4) if valid_wers else -1,
            "mean_cer": round(np.mean(valid_cers), 4) if valid_cers else -1,
            "sentiment_flip_rate": round(n_flips / n, 4) if n else 0,
            "sarcasm_false_positive_rate": round(n_fp / n, 4) if n else 0,
            "sarcasm_false_negative_rate": round(n_fn / n, 4) if n else 0,
            "sarcasm_agreement_rate": round(n_agree / n, 4) if n else 0,
            "inference_time_s": elapsed,
        }
        summary_per_model[model_size] = summary
        logger.info(f"  → WER={summary['mean_wer']} | Flip={summary['sentiment_flip_rate']} | "
                     f"FP={summary['sarcasm_false_positive_rate']} | FN={summary['sarcasm_false_negative_rate']} | "
                     f"Agree={summary['sarcasm_agreement_rate']} | Time={elapsed}s")

    # ------------------------------------------------------------------
    # Export CSV
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    detail_csv = os.path.join(output_dir, "cross_modal_ablation.csv")
    summary_csv = os.path.join(output_dir, "cross_modal_ablation_summary.csv")

    # Detail CSV (per file per model)
    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    logger.info(f"Detail results saved: {detail_csv}")

    # Summary CSV (per model)
    summary_rows = list(summary_per_model.values())
    if summary_rows:
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info(f"Summary results saved: {summary_csv}")

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CROSS-MODAL ABLATION STUDY - ASR ERRORS -> SARCASM DETECTION")
    print("=" * 70)
    print(f"  Audio files tested: {len(valid_entries)}")
    print(f"  Whisper models:     {', '.join(models)}")
    print("-" * 70)
    print(f"  {'Model':<16} {'WER':>8} {'CER':>8} {'Flip%':>8} {'FP%':>8} {'FN%':>8} {'Agree%':>8}")
    print("-" * 70)
    for s in summary_rows:
        wer_str = f"{s['mean_wer']:.2%}" if s["mean_wer"] >= 0 else "N/A"
        cer_str = f"{s['mean_cer']:.2%}" if s["mean_cer"] >= 0 else "N/A"
        print(f"  {s['model']:<16} {wer_str:>8} {cer_str:>8} "
              f"{s['sentiment_flip_rate']:.2%}  {s['sarcasm_false_positive_rate']:.2%}  "
              f"{s['sarcasm_false_negative_rate']:.2%}  {s['sarcasm_agreement_rate']:.2%}")
    print("=" * 70)
    print("  Flip%  = Sentiment label changed due to ASR errors")
    print("  FP%    = Sarcasm falsely detected (ASR typo triggered it)")
    print("  FN%    = Sarcasm missed (ASR typo masked it)")
    print("  Agree% = Sarcasm verdict matches ground-truth pipeline")
    print("=" * 70 + "\n")

    return summary_per_model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cross-Modal Ablation: ASR errors → Sentiment → Sarcasm cascading impact"
    )
    parser.add_argument(
        "--metadata", type=str, default="data/emotion_metadata.json",
        help="Path to JSON metadata with file_path, transcription, emotion fields"
    )
    parser.add_argument(
        "--models", nargs="+", default=["tiny", "base", "small"],
        choices=["tiny", "base", "small", "medium"],
        help="Whisper model sizes to benchmark"
    )
    parser.add_argument(
        "--output", type=str, default="results",
        help="Output directory for CSV results"
    )
    args = parser.parse_args()
    run_cross_modal_ablation(args.metadata, args.models, args.output)


if __name__ == "__main__":
    main()
