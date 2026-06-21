#!/usr/bin/env python3
"""
optimization/profiler.py
Joint Pipeline Profiler — Role 5 Deliverable (Elio)

Profiles the full multimodal inference pipeline (ASR + NLP + SER) measuring:
    - Per-stage latency (model loading + inference)
    - Peak RAM usage (via tracemalloc)
    - Model sizes on disk (MB)
    - Total end-to-end latency

Generates a CSV report and a stacked bar chart showing where time is spent.

Usage:
    python optimization/profiler.py
    python optimization/profiler.py --audio data/emotion_samples/03-01-05-02-01-01-01.wav
    python optimization/profiler.py --num-runs 5
"""

import argparse
import csv
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
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_model_size_mb(model) -> float:
    """Estimate model size in MB by saving state_dict to a temp file."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f.name)
        size = os.path.getsize(f.name) / (1024 * 1024)
        os.unlink(f.name)
    return round(size, 2)


def profile_pipeline(audio_path=None, num_runs=3, output_dir="results", visuals_dir="visuals"):
    """
    Profile each stage of the multimodal pipeline.

    Stages:
        1. ASR (Whisper-tiny): audio -> text
        2. NLP (DistilBERT): text -> sentiment
        3. SER (Wav2Vec2): audio -> emotion
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(visuals_dir, exist_ok=True)

    # Generate dummy audio if no file provided
    if audio_path and Path(audio_path).exists():
        import soundfile as sf
        audio_data, sr = sf.read(audio_path, dtype="float32")
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        logger.info(f"Loaded audio: {audio_path} ({len(audio_data)/sr:.2f}s @ {sr}Hz)")
    else:
        sr = 16000
        audio_data = np.random.randn(sr * 3).astype(np.float32)  # 3s dummy
        logger.info("Using 3s dummy audio (no file provided)")

    stages = {}

    # =================================================================
    # Stage 1: ASR — Whisper-tiny
    # =================================================================
    logger.info("=" * 50)
    logger.info("Stage 1: ASR (Whisper-tiny)")
    logger.info("=" * 50)

    tracemalloc.start()
    t_load_start = time.perf_counter()

    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    whisper_model.eval()

    t_load_end = time.perf_counter()
    load_time_asr = round(t_load_end - t_load_start, 3)
    size_asr = get_model_size_mb(whisper_model)

    # Benchmark inference
    input_features = whisper_processor(
        audio_data, sampling_rate=16000, return_tensors="pt"
    ).input_features

    # Warmup
    with torch.no_grad():
        whisper_model.generate(input_features, max_new_tokens=50)

    asr_times = []
    transcription = ""
    for _ in range(num_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            ids = whisper_model.generate(input_features, max_new_tokens=50)
        asr_times.append(time.perf_counter() - t0)
        transcription = whisper_processor.batch_decode(ids, skip_special_tokens=True)[0]

    _, peak_asr = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    avg_asr = round(np.mean(asr_times), 4)
    std_asr = round(np.std(asr_times), 4)

    stages["ASR (Whisper-tiny)"] = {
        "load_time_s": load_time_asr,
        "model_size_mb": size_asr,
        "avg_inference_s": avg_asr,
        "std_inference_s": std_asr,
        "peak_ram_mb": round(peak_asr / (1024 * 1024), 2),
    }
    logger.info(f"  Load: {load_time_asr}s | Size: {size_asr}MB | "
                f"Inference: {avg_asr}±{std_asr}s | Peak RAM: {stages['ASR (Whisper-tiny)']['peak_ram_mb']}MB")
    logger.info(f"  Transcription: \"{transcription}\"")

    del whisper_model, whisper_processor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # =================================================================
    # Stage 2: NLP — DistilBERT Sentiment
    # =================================================================
    logger.info("=" * 50)
    logger.info("Stage 2: NLP (DistilBERT Sentiment)")
    logger.info("=" * 50)

    tracemalloc.start()
    t_load_start = time.perf_counter()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    bert_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    bert_model.eval()

    t_load_end = time.perf_counter()
    load_time_nlp = round(t_load_end - t_load_start, 3)
    size_nlp = get_model_size_mb(bert_model)

    # Use actual transcription or fallback
    test_text = transcription if transcription.strip() else "Kids are talking by the door."
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=128)

    # Warmup
    with torch.no_grad():
        bert_model(**inputs)

    nlp_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = bert_model(**inputs)
        nlp_times.append(time.perf_counter() - t0)

    _, peak_nlp = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_label = "POSITIVE" if probs[0][1] > probs[0][0] else "NEGATIVE"

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
    logger.info(f"  Sentiment: {sentiment_label} ({probs[0].max():.2%})")

    del bert_model, tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # =================================================================
    # Stage 3: SER — Wav2Vec2
    # =================================================================
    logger.info("=" * 50)
    logger.info("Stage 3: SER (Wav2Vec2-superb-er)")
    logger.info("=" * 50)

    tracemalloc.start()
    t_load_start = time.perf_counter()

    from transformers import pipeline as hf_pipeline
    ser_pipe = hf_pipeline(
        "audio-classification",
        model="superb/wav2vec2-base-superb-er",
        device=-1,  # CPU
    )

    t_load_end = time.perf_counter()
    load_time_ser = round(t_load_end - t_load_start, 3)
    size_ser = get_model_size_mb(ser_pipe.model)

    # Warmup
    ser_pipe(audio_data)

    ser_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        ser_result = ser_pipe(audio_data)
        ser_times.append(time.perf_counter() - t0)

    _, peak_ser = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    avg_ser = round(np.mean(ser_times), 4)
    std_ser = round(np.std(ser_times), 4)
    top_emotion = ser_result[0]["label"] if ser_result else "unknown"

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

    # =================================================================
    # Summary
    # =================================================================
    total_load = sum(s["load_time_s"] for s in stages.values())
    total_inference = sum(s["avg_inference_s"] for s in stages.values())
    total_size = sum(s["model_size_mb"] for s in stages.values())
    max_ram = max(s["peak_ram_mb"] for s in stages.values())

    print("\n" + "=" * 70)
    print("  JOINT PIPELINE PROFILING REPORT")
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

    # -----------------------------------------------------------------
    # Export CSV
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # Visualization: stacked latency bar + model size pie
    # -----------------------------------------------------------------
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
    short_names = [s.split(" (")[1].rstrip(")") for s in stage_names]
    ax2.pie(sizes, labels=short_names, autopct="%1.1f%%", colors=colors,
            textprops={"fontsize": 10}, startangle=90)
    ax2.set_title(f"Model Size Distribution ({total_size:.0f} MB total)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    chart_path = os.path.join(visuals_dir, "profiling_breakdown.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Chart saved: {chart_path}")

    return stages


def main():
    parser = argparse.ArgumentParser(description="Profile the multimodal ASR+NLP+SER pipeline")
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file (uses dummy if omitted)")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of inference runs for averaging")
    parser.add_argument("--output", type=str, default="results", help="Output directory for CSV")
    args = parser.parse_args()
    profile_pipeline(args.audio, args.num_runs, args.output)


if __name__ == "__main__":
    main()
