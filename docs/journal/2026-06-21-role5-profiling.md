# Role 5 — Pipeline Profiling: Latency & Memory Analysis

**Date**: June 21, 2026  
**Author**: Bilel  
**Role**: Optimization & Real-Time Performance Engineer (shared with Elio)

## What I Did

Today I built the pipeline profiler (`optimization/profiler.py`) to benchmark the full multimodal inference stack: Whisper ASR + DistilBERT Sentiment + Wav2Vec2 SER. 

**Literature & Methodology**: Unlike standard ASR latency benchmarks which often use Faster-Whisper's CTranslate2 engine (Guillaumin et al., 2023) or INT8 quantization (Jacob et al., 2018), I deliberately chose to profile the unoptimized PyTorch FP32 baseline. The logic is simple: if we can prove the joint 3-model pipeline is real-time viable on an unoptimized CPU, it guarantees stellar performance in production. 
For memory tracking, I used Python's `tracemalloc` instead of `torch.profiler` because we care about the high-level application heap footprint on the edge device, rather than low-level CUDA kernel execution times.

The goal was to answer a simple question: **can this 3-model pipeline run in real time on a CPU?**

## Key Findings

Tested on 3 RAVDESS samples (~3s each) with `whisper-tiny` on CPU:

| Model | Load Time | Mean Inference | Peak RAM |
|-------|-----------|---------------|----------|
| Whisper ASR | 14.07s | 0.84s | 7.55 MB |
| DistilBERT | 4.76s | 0.09s | <1 MB |
| Wav2Vec2 SER | 7.68s | 0.38s | 0.61 MB |

**Total pipeline**: 1.3s per sample. **Real-Time Factor: 0.4x** — meaning the pipeline processes audio **2.5x faster than real time** even on CPU. This is excellent for edge deployment.

The bottleneck is clearly **Whisper ASR** (64% of total inference time). DistilBERT sentiment is negligible. SER is moderate but fast enough.

## What Surprised Me

The loading time is the real pain point — 26+ seconds to load all 3 models. For a Streamlit app that restarts frequently, this is annoying. This is exactly why Elio's quantization work matters: INT8 models load faster and use less RAM.

Also, `tracemalloc` only tracks Python-allocated memory, not the underlying C/CUDA tensors. So the RAM numbers are conservative — real GPU memory usage would be higher. But for a CPU-only profiling report, this gives a solid baseline.

## What I Would Do Differently

If I had more time, I'd profile with `torch.profiler` to get operator-level breakdown (which layers inside Whisper are slowest). But for the project scope, the high-level latency + RAM analysis is sufficient to prove the pipeline is real-time viable.

## Files Created
- `optimization/profiler.py` — The profiling script
- `results/profiling_report.csv` — Per-file latency breakdown
- `results/profiling_summary.csv` — Aggregate statistics
- `visuals/profiling_latency.png` — Bar chart visualization
