#!/usr/bin/env python3
"""
optimization/profiler.py
Joint Pipeline Profiler — Role 5 Deliverable (Elio & Bilel)

Profiles the full multimodal inference pipeline:
    1. Whisper ASR  (automatic-speech-recognition)
    2. DistilBERT   (text sentiment analysis)
    3. Wav2Vec2-SER (speech emotion recognition)

Measures:
    - Per-stage latency (model loading + inference)
    - Peak RAM usage (via tracemalloc)
    - Model sizes on disk (MB)
    - Total end-to-end latency & Real-Time Factor (RTF)

Modes:
    1. Dataset Profiling: Iterates over data/emotion_metadata.json (Bilel's approach)
    2. Single-File/Dummy Profiling: Run on a single audio file or dummy audio with multiple iterations (Elio's approach)
"""

import argparse
import csv
import json
import logging
import os
import sys
import tempfile
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

def get_model_size_mb(model) -> float:
    """Estimate model size in MB by saving state_dict to a temp file."""
    if hasattr(model, "model"):
        model = model.model
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model.state_dict(), f.name)
            size = os.path.getsize(f.name) / (1024 * 1024)
            os.unlink(f.name)
        return round(size, 2)
    except Exception as e:
        logger.warning(f"Could not estimate model size: {e}")
        return 0.0


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
# Dataset Mode (Bilel's workflow)
# -------------------------------------------------------------------------

def profile_dataset(
    metadata_path: str,
    whisper_size: str = "tiny",
    num_samples: int = 0,
    output_dir: str = "results",
    visuals_dir: str = "visuals",
):
    """
    Profile the 3-model multimodal pipeline on multiple audio samples from metadata.
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
        raise RuntimeError("No valid audio files found in metadata.")

    logger.info(f"Dataset Mode: Profiling {len(entries)} samples with whisper-{whisper_size}")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device_str}")

    # Load models (profiled separately)
    logger.info("Loading Whisper ASR model...")
    asr_pipe, asr_load_time, asr_load_ram = profile_block(
        hf_pipeline,
        "automatic-speech-recognition",
        model=f"openai/whisper-{whisper_size}",
        device=device_str,
    )
    size_asr = get_model_size_mb(asr_pipe)

    logger.info("Loading DistilBERT sentiment model...")
    sent_pipe, sent_load_time, sent_load_ram = profile_block(
        hf_pipeline,
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device_str,
    )
    size_sent = get_model_size_mb(sent_pipe)

    logger.info("Loading Wav2Vec2-SER model...")
    ser_pipe, ser_load_time, ser_load_ram = profile_block(
        hf_pipeline,
        "audio-classification",
        model="superb/wav2vec2-base-superb-er",
        device=device_str,
    )
    size_ser = get_model_size_mb(ser_pipe)

    logger.info(f"Model sizes -> ASR: {size_asr}MB | Sentiment: {size_sent}MB | SER: {size_ser}MB")
    logger.info(f"Model load times -> ASR: {asr_load_time}s | Sentiment: {sent_load_time}s | SER: {ser_load_time}s")

    # Per-sample profiling
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

    # Aggregate statistics
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

    # Export CSVs
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

    # Visualization: Mean Latency Bar Chart & Model Loading Time
    models_labels = ["Whisper ASR", "DistilBERT\nSentiment", "Wav2Vec2\nSER"]
    means = [agg["mean_asr_latency_s"], agg["mean_sentiment_latency_s"], agg["mean_ser_latency_s"]]
    colors = ["#6366f1", "#f59e0b", "#10b981"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Latency bar chart
    ax1 = axes[0]
    bars = ax1.bar(models_labels, means, color=colors, edgecolor="white", linewidth=1.2)
    ax1.set_ylabel("Mean Latency (s)", fontsize=11)
    ax1.set_title(f"Per-Model Inference Latency (whisper-{whisper_size}, {device_str})", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, max(means) * 1.3)
    ax1.grid(axis="y", alpha=0.3)

    # Load time bar chart
    ax2 = axes[1]
    load_times = [agg["asr_load_time_s"], agg["sentiment_load_time_s"], agg["ser_load_time_s"]]
    bars2 = ax2.bar(models_labels, load_times, color=colors, edgecolor="white", linewidth=1.2, alpha=0.7)
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

    # Console report
    print("\n" + "=" * 65)
    print("  MULTIMODAL PIPELINE PROFILING REPORT")
    print("=" * 65)
    print(f"  Whisper model:  {whisper_size}")
    print(f"  Device:         {device_str}")
    print(f"  Samples:        {n}")
    print("-" * 65)
    print(f"  {'Stage':<22} {'Size(MB)':<10} {'Load(s)':<10} {'Mean Inf(s)':<14} {'Peak RAM(MB)':<14}")
    print("-" * 65)
    print(f"  {'Whisper ASR':<22} {size_asr:<10} {agg['asr_load_time_s']:<10} {agg['mean_asr_latency_s']:<14} {agg['max_asr_peak_ram_mb']:<14}")
    print(f"  {'DistilBERT Sent.':<22} {size_sent:<10} {agg['sentiment_load_time_s']:<10} {agg['mean_sentiment_latency_s']:<14} {'<1':<14}")
    print(f"  {'Wav2Vec2 SER':<22} {size_ser:<10} {agg['ser_load_time_s']:<10} {agg['mean_ser_latency_s']:<14} {agg['max_ser_peak_ram_mb']:<14}")
    print("-" * 65)
    total_sizes = size_asr + size_sent + size_ser
    total_load = agg['asr_load_time_s'] + agg['sentiment_load_time_s'] + agg['ser_load_time_s']
    max_peak_ram = max(agg['max_asr_peak_ram_mb'], agg['max_ser_peak_ram_mb'])
    print(f"  {'TOTAL':<22} {total_sizes:<10.2f} {total_load:<10.2f} {agg['mean_total_latency_s']:<14} {max_peak_ram:<14}")
    print("-" * 65)
    print(f"  Real-Time Factor:  {agg['mean_rtf']}x")
    print("=" * 65 + "\n")

    return agg


# -------------------------------------------------------------------------
# Single-File / Dummy Mode (Elio's detailed workflow)
# -------------------------------------------------------------------------

def profile_single(
    audio_path: str = None,
    whisper_size: str = "tiny",
    num_runs: int = 3,
    output_dir: str = "results",
    visuals_dir: str = "visuals",
):
    """
    Profile each stage of the multimodal pipeline on a single audio file or dummy audio.
    Sequential loads/unloads to isolate model size and peak RAM usage accurately.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(visuals_dir, exist_ok=True)

    # Load audio
    if audio_path and Path(audio_path).exists():
        audio_data, sr = sf.read(audio_path, dtype="float32")
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        logger.info(f"Single Mode: Loaded audio {audio_path} ({len(audio_data)/sr:.2f}s @ {sr}Hz)")
    else:
        sr = 16000
        audio_data = np.random.randn(sr * 3).astype(np.float32)  # 3s dummy
        logger.info("Single Mode: Using 3s dummy audio (no valid file provided)")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device_str}")

    stages = {}

    # =================================================================
    # Stage 1: ASR — Whisper
    # =================================================================
    logger.info("=" * 50)
    logger.info(f"Stage 1: ASR (Whisper-{whisper_size})")
    logger.info("=" * 50)

    tracemalloc.start()
    t_load_start = time.perf_counter()

    asr_pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=f"openai/whisper-{whisper_size}",
        device=device_str,
    )

    t_load_end = time.perf_counter()
    load_time_asr = round(t_load_end - t_load_start, 3)
    size_asr = get_model_size_mb(asr_pipe)

    # Warmup
    run_asr(asr_pipe, audio_data)

    # Benchmark
    asr_times = []
    transcription = ""
    for _ in range(num_runs):
        t0 = time.perf_counter()
        transcription = run_asr(asr_pipe, audio_data)
        asr_times.append(time.perf_counter() - t0)

    _, peak_asr = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    avg_asr = round(np.mean(asr_times), 4)
    std_asr = round(np.std(asr_times), 4)

    stages[f"ASR (Whisper-{whisper_size})"] = {
        "load_time_s": load_time_asr,
        "model_size_mb": size_asr,
        "avg_inference_s": avg_asr,
        "std_inference_s": std_asr,
        "peak_ram_mb": round(peak_asr / (1024 * 1024), 2),
    }
    logger.info(f"  Load: {load_time_asr}s | Size: {size_asr}MB | "
                f"Inference: {avg_asr}±{std_asr}s | Peak RAM: {stages[f'ASR (Whisper-{whisper_size})']['peak_ram_mb']}MB")
    logger.info(f"  Transcription: \"{transcription}\"")

    del asr_pipe
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # =================================================================
    # Stage 2: NLP — DistilBERT Sentiment
    # =================================================================
    logger.info("=" * 50)
    logger.info("Stage 2: NLP (DistilBERT Sentiment)")
    logger.info("=" * 50)

    tracemalloc.start()
    t_load_start = time.perf_counter()

    sent_pipe = hf_pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device_str,
    )

    t_load_end = time.perf_counter()
    load_time_nlp = round(t_load_end - t_load_start, 3)
    size_nlp = get_model_size_mb(sent_pipe)

    test_text = transcription if transcription.strip() else "Kids are talking by the door."

    # Warmup
    run_sentiment(sent_pipe, test_text)

    # Benchmark
    nlp_times = []
    sentiment_label = ""
    for _ in range(num_runs):
        t0 = time.perf_counter()
        sentiment_label = run_sentiment(sent_pipe, test_text)
        nlp_times.append(time.perf_counter() - t0)

    _, peak_nlp = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    avg_nlp = round(np.mean(nlp_times), 4)
    std_nlp = round(np.std(nlp_times), 4)

    stages["NLP (DistilBERT)"] = {
        "load_time_s": load_time_nlp,
        "model_size_mb": size_nlp,
        "avg_inference_s": avg_nlp,
        "std_inference_s": std_nlp,
        "peak_ram_mb": round(peak_nlp / (1024 * 1024), 2),
    }
    logger.info(f"  Load: {load_time_nlp}s | Size: {size_nlp}MB | "
                f"Inference: {avg_nlp}±{std_nlp}s | Peak RAM: {stages['NLP (DistilBERT)']['peak_ram_mb']}MB")
    logger.info(f"  Sentiment: {sentiment_label}")

    del sent_pipe
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # =================================================================
    # Stage 3: SER — Wav2Vec2
    # =================================================================
    logger.info("=" * 50)
    logger.info("Stage 3: SER (Wav2Vec2-superb-er)")
    logger.info("=" * 50)

    tracemalloc.start()
    t_load_start = time.perf_counter()

    ser_pipe = hf_pipeline(
        "audio-classification",
        model="superb/wav2vec2-base-superb-er",
        device=device_str,
    )

    t_load_end = time.perf_counter()
    load_time_ser = round(t_load_end - t_load_start, 3)
    size_ser = get_model_size_mb(ser_pipe)

    # Warmup
    run_ser(ser_pipe, audio_data)

    # Benchmark
    ser_times = []
    top_emotion = ""
    for _ in range(num_runs):
        t0 = time.perf_counter()
        top_emotion = run_ser(ser_pipe, audio_data)
        ser_times.append(time.perf_counter() - t0)

    _, peak_ser = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    avg_ser = round(np.mean(ser_times), 4)
    std_ser = round(np.std(ser_times), 4)

    stages["SER (Wav2Vec2)"] = {
        "load_time_s": load_time_ser,
        "model_size_mb": size_ser,
        "avg_inference_s": avg_ser,
        "std_inference_s": std_ser,
        "peak_ram_mb": round(peak_ser / (1024 * 1024), 2),
    }
    logger.info(f"  Load: {load_time_ser}s | Size: {size_ser}MB | "
                f"Inference: {avg_ser}±{std_ser}s | Peak RAM: {stages['SER (Wav2Vec2)']['peak_ram_mb']}MB")
    logger.info(f"  Emotion: {top_emotion}")

    del ser_pipe
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Summary calculations
    total_load = sum(s["load_time_s"] for s in stages.values())
    total_inference = sum(s["avg_inference_s"] for s in stages.values())
    total_size = sum(s["model_size_mb"] for s in stages.values())
    max_ram = max(s["peak_ram_mb"] for s in stages.values())

    print("\n" + "=" * 70)
    print("  JOINT PIPELINE PROFILING REPORT (SINGLE FILE)")
    print("=" * 70)
    print(f"  {'Stage':<25} {'Load(s)':<10} {'Size(MB)':<10} {'Infer(s)':<12} {'RAM(MB)':<10}")
    print(f"  {'-'*25} {'-'*9} {'-'*9} {'-'*11} {'-'*9}")
    for name, m in stages.items():
        print(f"  {name:<25} {m['load_time_s']:<10} {m['model_size_mb']:<10} "
              f"{m['avg_inference_s']:<12} {m['peak_ram_mb']:<10}")
    print(f"  {'-'*25} {'-'*9} {'-'*9} {'-'*11} {'-'*9}")
    print(f"  {'TOTAL':<25} {total_load:<10.3f} {total_size:<10.1f} {total_inference:<12.4f} {max_ram:<10.1f}")
    print(f"\n  End-to-end (load+infer): {total_load + total_inference:.3f}s")
    print("=" * 70 + "\n")

    # Export CSV
    csv_path = os.path.join(output_dir, "profiling_report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["stage", "load_time_s", "model_size_mb", "avg_inference_s", "std_inference_s", "peak_ram_mb"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, m in stages.items():
            writer.writerow({"stage": name, **m})
        writer.writerow({
            "stage": "TOTAL",
            "load_time_s": round(total_load, 3),
            "model_size_mb": round(total_size, 2),
            "avg_inference_s": round(total_inference, 4),
            "std_inference_s": "",
            "peak_ram_mb": round(max_ram, 2),
        })
    logger.info(f"Profiling report saved: {csv_path}")

    # Visualization: stacked latency bar + model size pie
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Stacked bar: load vs inference per stage
    stage_names = list(stages.keys())
    load_times = [stages[s]["load_time_s"] for s in stage_names]
    infer_times = [stages[s]["avg_inference_s"] for s in stage_names]

    x = np.arange(len(stage_names))
    ax1.bar(x, load_times, 0.5, label="Model Loading", color="#6366f1", edgecolor="white")
    ax1.bar(x, infer_times, 0.5, bottom=load_times, label="Inference", color="#f59e0b", edgecolor="white")
    ax1.set_ylabel("Time (s)", fontsize=11)
    ax1.set_title("Latency Breakdown by Stage", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.split(" (")[0] for s in stage_names], fontsize=9)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Pie chart: model sizes
    sizes = [stages[s]["model_size_mb"] for s in stage_names]
    colors = ["#6366f1", "#10b981", "#f59e0b"]
    short_names = [s.split(" (")[1].rstrip(")") if "(" in s else s for s in stage_names]
    ax2.pie(sizes, labels=short_names, autopct="%1.1f%%", colors=colors,
            textprops={"fontsize": 10}, startangle=90)
    ax2.set_title(f"Model Size Distribution ({total_size:.0f} MB total)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    chart_path = os.path.join(visuals_dir, "profiling_breakdown.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Chart saved: {chart_path}")

    return stages


# -------------------------------------------------------------------------
# CLI Entry Point
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Profile the joint multimodal ASR+Sentiment+SER pipeline")
    parser.add_argument("--metadata", type=str, default="data/emotion_metadata.json",
                        help="Path to JSON metadata (dataset mode)")
    parser.add_argument("--audio", type=str, default=None,
                        help="Path to a single audio file for detailed single-file profiling (uses dummy if omitted and metadata not found)")
    parser.add_argument("--whisper-size", type=str, default="tiny", choices=["tiny", "base", "small"],
                        help="Whisper model size to profile")
    parser.add_argument("--num-samples", type=int, default=0,
                        help="Number of samples to profile in dataset mode (0 = all)")
    parser.add_argument("--num-runs", type=int, default=3,
                        help="Number of inference runs for averaging in single-file mode")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for CSV files")
    parser.add_argument("--visuals", type=str, default="visuals",
                        help="Output directory for chart files")
    args = parser.parse_args()

    # Determine mode: if metadata exists and --audio is not specified, run dataset profiling.
    # Otherwise, if --audio is specified or metadata doesn't exist, run single-file profiling.
    metadata_exists = Path(args.metadata).exists()
    
    if metadata_exists and args.audio is None:
        logger.info("Found metadata file. Running in Dataset Profiling Mode...")
        try:
            profile_dataset(
                metadata_path=args.metadata,
                whisper_size=args.whisper_size,
                num_samples=args.num_samples,
                output_dir=args.output,
                visuals_dir=args.visuals
            )
        except Exception as e:
            logger.error(f"Error during dataset profiling: {e}. Falling back to Single/Dummy Mode.")
            profile_single(
                audio_path=args.audio,
                whisper_size=args.whisper_size,
                num_runs=args.num_runs,
                output_dir=args.output,
                visuals_dir=args.visuals
            )
    else:
        if args.audio is not None:
            logger.info(f"Running in Single-File Profiling Mode on: {args.audio}")
        else:
            logger.info(f"Metadata file '{args.metadata}' not found and no --audio file provided. Running in Dummy Audio Profiling Mode...")
        profile_single(
            audio_path=args.audio,
            whisper_size=args.whisper_size,
            num_runs=args.num_runs,
            output_dir=args.output,
            visuals_dir=args.visuals
        )


if __name__ == "__main__":
    main()
