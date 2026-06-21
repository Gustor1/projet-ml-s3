#!/usr/bin/env python3
"""
optimization/quantize_model.py
INT8 Dynamic Quantization — Role 5 Deliverable (Elio)

Applies PyTorch Dynamic Quantization (INT8) to the pipeline models
to reduce memory footprint and improve CPU inference speed.

Targets:
    - Whisper-tiny (encoder)
    - DistilBERT (sentiment analysis)

Wav2Vec2-SER is excluded: its convolutional feature extractor does not
benefit from dynamic linear quantization, and HuggingFace audio pipelines
wrap the model in a way that makes direct quantization impractical.

Metrics reported:
    - Model size before/after (MB)
    - Inference latency before/after (seconds)
    - Speedup ratio

Usage:
    python optimization/quantize_model.py
    python optimization/quantize_model.py --whisper-size base --num-samples 5
"""

import argparse
import csv
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_model_size_mb(model) -> float:
    """Estimate model size in MB by saving to a temp file."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f.name)
        size = os.path.getsize(f.name) / (1024 * 1024)
        os.unlink(f.name)
    return round(size, 2)


def benchmark_whisper(model, processor, num_runs=5):
    """Benchmark Whisper inference on dummy audio."""
    dummy_audio = np.random.randn(16000 * 3).astype(np.float32)  # 3s audio
    input_features = processor(
        dummy_audio, sampling_rate=16000, return_tensors="pt"
    ).input_features

    # Warmup
    with torch.no_grad():
        model.generate(input_features, max_new_tokens=20)

    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(input_features, max_new_tokens=20)
        times.append(time.perf_counter() - t0)

    return round(np.mean(times), 4)


def benchmark_distilbert(model, tokenizer, num_runs=10):
    """Benchmark DistilBERT inference on sample text."""
    sample = "Kids are talking by the door."
    inputs = tokenizer(sample, return_tensors="pt", truncation=True, max_length=128)

    # Warmup
    with torch.no_grad():
        model(**inputs)

    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(**inputs)
        times.append(time.perf_counter() - t0)

    return round(np.mean(times), 4)


def quantize_and_compare(whisper_size="tiny", output_dir="results", visuals_dir="visuals"):
    """
    Quantize Whisper and DistilBERT, compare before/after metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(visuals_dir, exist_ok=True)

    results = []

    # -----------------------------------------------------------------
    # Whisper quantization
    # -----------------------------------------------------------------
    whisper_model_name = f"openai/whisper-{whisper_size}"
    logger.info(f"Loading {whisper_model_name}...")
    processor = WhisperProcessor.from_pretrained(whisper_model_name)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)
    whisper_model.eval()

    size_before = get_model_size_mb(whisper_model)
    latency_before = benchmark_whisper(whisper_model, processor)
    logger.info(f"  Whisper BEFORE: {size_before} MB, {latency_before}s/inference")

    # Dynamic quantization on linear layers
    whisper_quantized = torch.quantization.quantize_dynamic(
        whisper_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    whisper_quantized.eval()

    size_after = get_model_size_mb(whisper_quantized)
    latency_after = benchmark_whisper(whisper_quantized, processor)
    speedup = round(latency_before / latency_after, 2) if latency_after > 0 else 0

    logger.info(f"  Whisper AFTER:  {size_after} MB, {latency_after}s/inference (speedup: {speedup}x)")

    results.append({
        "model": f"whisper-{whisper_size}",
        "size_before_mb": size_before,
        "size_after_mb": size_after,
        "size_reduction_pct": round((1 - size_after / size_before) * 100, 1) if size_before > 0 else 0,
        "latency_before_s": latency_before,
        "latency_after_s": latency_after,
        "speedup": speedup,
    })

    # Free memory
    del whisper_model, whisper_quantized
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # -----------------------------------------------------------------
    # DistilBERT quantization
    # -----------------------------------------------------------------
    distilbert_name = "distilbert-base-uncased-finetuned-sst-2-english"
    logger.info(f"Loading {distilbert_name}...")
    tokenizer = AutoTokenizer.from_pretrained(distilbert_name)
    bert_model = AutoModelForSequenceClassification.from_pretrained(distilbert_name)
    bert_model.eval()

    size_before = get_model_size_mb(bert_model)
    latency_before = benchmark_distilbert(bert_model, tokenizer)
    logger.info(f"  DistilBERT BEFORE: {size_before} MB, {latency_before}s/inference")

    bert_quantized = torch.quantization.quantize_dynamic(
        bert_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    bert_quantized.eval()

    size_after = get_model_size_mb(bert_quantized)
    latency_after = benchmark_distilbert(bert_quantized, tokenizer)
    speedup = round(latency_before / latency_after, 2) if latency_after > 0 else 0

    logger.info(f"  DistilBERT AFTER:  {size_after} MB, {latency_after}s/inference (speedup: {speedup}x)")

    results.append({
        "model": "distilbert-sst2",
        "size_before_mb": size_before,
        "size_after_mb": size_after,
        "size_reduction_pct": round((1 - size_after / size_before) * 100, 1) if size_before > 0 else 0,
        "latency_before_s": latency_before,
        "latency_after_s": latency_after,
        "speedup": speedup,
    })

    # -----------------------------------------------------------------
    # Export CSV
    # -----------------------------------------------------------------
    csv_path = os.path.join(output_dir, "quantization_report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Quantization report saved: {csv_path}")

    # -----------------------------------------------------------------
    # Visualization
    # -----------------------------------------------------------------
    model_names = [r["model"] for r in results]
    x = np.arange(len(model_names))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Size comparison
    before_sizes = [r["size_before_mb"] for r in results]
    after_sizes = [r["size_after_mb"] for r in results]
    bars1 = ax1.bar(x - width/2, before_sizes, width, label="FP32 (original)", color="#6366f1", edgecolor="white")
    bars2 = ax1.bar(x + width/2, after_sizes, width, label="INT8 (quantized)", color="#10b981", edgecolor="white")
    ax1.set_ylabel("Model Size (MB)", fontsize=11)
    ax1.set_title("Model Size: FP32 vs INT8", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, before_sizes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.0f}", ha="center", fontsize=9)
    for bar, val in zip(bars2, after_sizes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.0f}", ha="center", fontsize=9)

    # Latency comparison
    before_lat = [r["latency_before_s"] for r in results]
    after_lat = [r["latency_after_s"] for r in results]
    bars3 = ax2.bar(x - width/2, before_lat, width, label="FP32 (original)", color="#f59e0b", edgecolor="white")
    bars4 = ax2.bar(x + width/2, after_lat, width, label="INT8 (quantized)", color="#10b981", edgecolor="white")
    ax2.set_ylabel("Inference Latency (s)", fontsize=11)
    ax2.set_title("Inference Speed: FP32 vs INT8", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars3, before_lat):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{val:.3f}s", ha="center", fontsize=9)
    for bar, val in zip(bars4, after_lat):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{val:.3f}s", ha="center", fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(visuals_dir, "quantization_speedup.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Chart saved: {chart_path}")

    # -----------------------------------------------------------------
    # Console report
    # -----------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  INT8 DYNAMIC QUANTIZATION REPORT")
    print("=" * 65)
    for r in results:
        print(f"  {r['model']}")
        print(f"    Size:    {r['size_before_mb']} MB -> {r['size_after_mb']} MB ({r['size_reduction_pct']}% reduction)")
        print(f"    Latency: {r['latency_before_s']}s -> {r['latency_after_s']}s ({r['speedup']}x speedup)")
    print("=" * 65 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="INT8 Dynamic Quantization for pipeline models")
    parser.add_argument("--whisper-size", type=str, default="tiny", choices=["tiny", "base", "small"])
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()
    quantize_and_compare(args.whisper_size, args.output)


if __name__ == "__main__":
    main()
