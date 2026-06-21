#!/usr/bin/env python3
"""
optimization/profiler.py
Pipeline Profiling — Role 5 Deliverable (Bilel)

Profiles the joint multimodal inference pipeline:
    1. Whisper ASR  (automatic-speech-recognition)
    2. DistilBERT   (text sentiment analysis)
    3. Wav2Vec2-SER (speech emotion recognition)

Measures per-model:
    - Inference latency (seconds)
    - Peak RAM delta (MB)
    - Throughput (samples/sec)

Usage:
    python optimization/profiler.py --metadata data/emotion_metadata.json
    python optimization/profiler.py --metadata data/emotion_metadata.json --num-samples 5 --whisper-size tiny
"""

import argparse
import csv
import json
import logging
import os
import time
import tracemalloc
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from transformers import pipeline as hf_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Profiling helpers
# -------------------------------------------------------------------------

def profile_block(func, *args, **kwargs):
    """
    Run func(*args, **kwargs) while tracking wall-clock time and peak RAM.
    Returns (result, elapsed_sec, peak_ram_mb).
    """
    tracemalloc.start()
    baseline_mem = tracemalloc.get_traced_memory()[0]

    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - t0

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_delta_mb = round((peak - baseline_mem) / (1024 * 1024), 2)
    return result, round(elapsed, 4), max(peak_delta_mb, 0.0)


# -------------------------------------------------------------------------
# Individual model runners
# -------------------------------------------------------------------------

def run_asr(pipe, audio):
    """Run Whisper ASR and return transcription text."""
    result = pipe(audio, generate_kwargs={"language": "en"})
    return result["text"].strip()


def run_sentiment(pipe, text):
    """Run DistilBERT sentiment and return label."""
    if not text or len(text.strip()) < 3:
        return "neutral"
    result = pipe(text[:512])[0]
    return result["label"].lower()


def run_ser(pipe, audio):
    """Run Wav2Vec2-SER and return emotion label."""
    result = pipe(audio)
    return result[0]["label"]


# -------------------------------------------------------------------------
# Main profiling loop
# -------------------------------------------------------------------------

def profile_pipeline(
    metadata_path: str,
    whisper_size: str = "tiny",
    num_samples: int = 0,
    output_dir: str = "results",
    visuals_dir: str = "visuals",
):
    """
    Profile the 3-model multimodal pipeline on audio samples.

    Parameters
    ----------
    metadata_path : str
        JSON file with entries: {file_path, transcription, emotion}.
    whisper_size : str
        Whisper model size (tiny, base, small).
    num_samples : int
        Number of samples to profile. 0 = all.
    output_dir : str
        CSV output directory.
    visuals_dir : str
        Chart output directory.
    """
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Filter to existing files
    entries = [e for e in metadata if Path(e["file_path"]).exists()]
    if num_samples > 0:
        entries = entries[:num_samples]

    if not entries:
        raise RuntimeError("No valid audio files found.")

    logger.info(f"Profiling {len(entries)} samples with whisper-{whisper_size}")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device_str}")

    # ------------------------------------------------------------------
    # Load models (profiled separately)
    # ------------------------------------------------------------------
    logger.info("Loading Whisper ASR model...")
    asr_pipe, asr_load_time, asr_load_ram = profile_block(
        hf_pipeline,
        "automatic-speech-recognition",
        model=f"openai/whisper-{whisper_size}",
        device=device_str,
    )

    logger.info("Loading DistilBERT sentiment model...")
    sent_pipe, sent_load_time, sent_load_ram = profile_block(
        hf_pipeline,
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device_str,
    )

    logger.info("Loading Wav2Vec2-SER model...")
    ser_pipe, ser_load_time, ser_load_ram = profile_block(
        hf_pipeline,
        "audio-classification",
        model="superb/wav2vec2-base-superb-er",
        device=device_str,
    )

    logger.info(f"Model load times -> ASR: {asr_load_time}s | Sentiment: {sent_load_time}s | SER: {ser_load_time}s")

    # ------------------------------------------------------------------
    # Per-sample profiling
    # ------------------------------------------------------------------
    rows = []

    for i, entry in enumerate(entries):
        fpath = entry["file_path"]
        audio, sr = sf.read(fpath, dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        duration = round(len(audio) / sr, 2)

        # ASR
        asr_text, asr_time, asr_ram = profile_block(run_asr, asr_pipe, audio)

        # Sentiment (on ASR output)
        sent_label, sent_time, sent_ram = profile_block(run_sentiment, sent_pipe, asr_text)

        # SER
        ser_label, ser_time, ser_ram = profile_block(run_ser, ser_pipe, audio)

        total_time = round(asr_time + sent_time + ser_time, 4)

        row = {
            "file": os.path.basename(fpath),
            "duration_s": duration,
            "asr_latency_s": asr_time,
            "asr_peak_ram_mb": asr_ram,
            "sentiment_latency_s": sent_time,
            "sentiment_peak_ram_mb": sent_ram,
            "ser_latency_s": ser_time,
            "ser_peak_ram_mb": ser_ram,
            "total_latency_s": total_time,
            "real_time_factor": round(total_time / duration, 2) if duration > 0 else -1,
        }
        rows.append(row)

        if (i + 1) % 5 == 0 or (i + 1) == len(entries):
            logger.info(f"  [{i+1}/{len(entries)}] {row['file']} -> total {total_time}s (RTF={row['real_time_factor']}x)")

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------
    n = len(rows)
    agg = {
        "model": f"whisper-{whisper_size}",
        "num_samples": n,
        "device": device_str,
        "asr_load_time_s": asr_load_time,
        "asr_load_ram_mb": asr_load_ram,
        "sentiment_load_time_s": sent_load_time,
        "sentiment_load_ram_mb": sent_load_ram,
        "ser_load_time_s": ser_load_time,
        "ser_load_ram_mb": ser_load_ram,
        "mean_asr_latency_s": round(np.mean([r["asr_latency_s"] for r in rows]), 4),
        "mean_sentiment_latency_s": round(np.mean([r["sentiment_latency_s"] for r in rows]), 4),
        "mean_ser_latency_s": round(np.mean([r["ser_latency_s"] for r in rows]), 4),
        "mean_total_latency_s": round(np.mean([r["total_latency_s"] for r in rows]), 4),
        "mean_rtf": round(np.mean([r["real_time_factor"] for r in rows if r["real_time_factor"] > 0]), 2),
        "max_asr_peak_ram_mb": round(max(r["asr_peak_ram_mb"] for r in rows), 2),
        "max_ser_peak_ram_mb": round(max(r["ser_peak_ram_mb"] for r in rows), 2),
    }

    # ------------------------------------------------------------------
    # Export CSV
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(visuals_dir, exist_ok=True)

    detail_csv = os.path.join(output_dir, "profiling_report.csv")
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Detail profiling saved: {detail_csv}")

    summary_csv = os.path.join(output_dir, "profiling_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg.keys()))
        writer.writeheader()
        writer.writerow(agg)
    logger.info(f"Summary profiling saved: {summary_csv}")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    models = ["Whisper ASR", "DistilBERT\nSentiment", "Wav2Vec2\nSER"]
    means = [agg["mean_asr_latency_s"], agg["mean_sentiment_latency_s"], agg["mean_ser_latency_s"]]
    colors = ["#6366f1", "#f59e0b", "#10b981"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Latency bar chart
    ax1 = axes[0]
    bars = ax1.bar(models, means, color=colors, edgecolor="white", linewidth=1.2)
    ax1.set_ylabel("Mean Latency (s)", fontsize=11)
    ax1.set_title(f"Per-Model Inference Latency (whisper-{whisper_size}, {device_str})", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, max(means) * 1.3)
    ax1.grid(axis="y", alpha=0.3)

    # Load time + RAM chart
    ax2 = axes[1]
    load_times = [agg["asr_load_time_s"], agg["sentiment_load_time_s"], agg["ser_load_time_s"]]
    bars2 = ax2.bar(models, load_times, color=colors, edgecolor="white", linewidth=1.2, alpha=0.7)
    ax2.set_ylabel("Load Time (s)", fontsize=11)
    ax2.set_title("Model Loading Time", fontsize=12, fontweight="bold")
    for bar, val in zip(bars2, load_times):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{val:.2f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, max(load_times) * 1.3)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(visuals_dir, "profiling_latency.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Latency chart saved: {chart_path}")

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  MULTIMODAL PIPELINE PROFILING REPORT")
    print("=" * 65)
    print(f"  Whisper model:  {whisper_size}")
    print(f"  Device:         {device_str}")
    print(f"  Samples:        {n}")
    print("-" * 65)
    print(f"  {'Stage':<22} {'Load(s)':<10} {'Mean Inf(s)':<14} {'Peak RAM(MB)':<14}")
    print("-" * 65)
    print(f"  {'Whisper ASR':<22} {agg['asr_load_time_s']:<10} {agg['mean_asr_latency_s']:<14} {agg['max_asr_peak_ram_mb']:<14}")
    print(f"  {'DistilBERT Sent.':<22} {agg['sentiment_load_time_s']:<10} {agg['mean_sentiment_latency_s']:<14} {'<1':<14}")
    print(f"  {'Wav2Vec2 SER':<22} {agg['ser_load_time_s']:<10} {agg['mean_ser_latency_s']:<14} {agg['max_ser_peak_ram_mb']:<14}")
    print("-" * 65)
    print(f"  Total pipeline:    {agg['mean_total_latency_s']}s/sample")
    print(f"  Real-Time Factor:  {agg['mean_rtf']}x")
    print("=" * 65 + "\n")

    return agg


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Profile the multimodal ASR+Sentiment+SER pipeline"
    )
    parser.add_argument(
        "--metadata", type=str, default="data/emotion_metadata.json",
        help="Path to JSON metadata"
    )
    parser.add_argument(
        "--whisper-size", type=str, default="tiny",
        choices=["tiny", "base", "small"],
        help="Whisper model size to profile"
    )
    parser.add_argument(
        "--num-samples", type=int, default=0,
        help="Number of samples to profile (0 = all)"
    )
    parser.add_argument(
        "--output", type=str, default="results",
        help="Output directory for CSV"
    )
    args = parser.parse_args()
    profile_pipeline(args.metadata, args.whisper_size, args.num_samples, args.output)


if __name__ == "__main__":
    main()
