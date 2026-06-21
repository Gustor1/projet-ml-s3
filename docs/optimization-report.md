# ⚡ Optimization & Real-Time Performance — Technical Report

## Overview

This document covers the engineering work for **Role 5: Optimization & Real-Time Performance**. The goal was to measure where time and memory are spent in the 3-model multimodal pipeline, reduce the footprint via INT8 quantization, and add support for processing long audio files via streaming chunks.

---

## 1. Pipeline Profiling: Where Does Time Go?

Before optimizing anything, we profiled the full pipeline to understand the bottleneck. The profiler (`optimization/profiler.py`) measures each stage independently:

### Methodology
- Load each model separately, measuring load time and peak RAM via `tracemalloc`
- Run inference N times (default: 3) on the same audio, report mean ± std
- Save per-model state_dict to temp file for disk size measurement

### Expected Latency Breakdown (CPU)

| Stage | Model | Approx. Size | Approx. Load | Approx. Inference |
|-------|-------|-------------|-------------|-------------------|
| ASR | Whisper-tiny | ~150 MB | ~3-5s | ~1.5-3s (3s audio) |
| NLP | DistilBERT-SST2 | ~255 MB | ~1-2s | ~0.01-0.03s |
| SER | Wav2Vec2-superb-er | ~360 MB | ~2-4s | ~0.5-1.5s |
| **Total** | | **~765 MB** | **~6-11s** | **~2-4.5s** |

**Observations**:
1. **Whisper dominates inference time** (~60-70% of total) due to autoregressive decoding. Each token requires a full forward pass through the decoder.
2. **DistilBERT is negligible** (<1% of inference time). Text sentiment analysis on a single sentence is essentially instant.
3. **Model loading is a one-time cost** but is substantial (~6-11s total). In production, models should be loaded once and kept in memory.
4. **Total footprint is ~765 MB** for all three models. This fits comfortably on most edge devices (Raspberry Pi 4 has 4GB RAM) but would struggle on microcontrollers.

### Caveat: tracemalloc limitations
`tracemalloc` only tracks Python-level memory allocations. PyTorch tensors allocated via the C++ backend (libtorch) are largely invisible. The reported "Peak RAM" values are *lower bounds*. For accurate GPU profiling, use `torch.cuda.max_memory_allocated()`.

---

## 2. INT8 Dynamic Quantization

### What We Quantized (and What We Didn't)

| Model | Quantized? | Reason |
|-------|-----------|--------|
| Whisper-tiny | ✅ Yes | Encoder has dense linear layers |
| DistilBERT-SST2 | ✅ Yes | Pure transformer encoder, ideal target |
| Wav2Vec2-SER | ❌ No | Conv-heavy feature extractor, <5% gains |

### Why Not Wav2Vec2?
Wav2Vec2's architecture starts with a **1D convolutional feature extractor** (7 layers of `Conv1d`) that processes raw audio waveforms. `torch.quantization.quantize_dynamic` only affects `nn.Linear` and `nn.LSTM` layers — the conv layers that do the heavy computation are untouched. In our testing, quantizing Wav2Vec2 gave <5% size reduction and actually *increased* latency slightly due to quantized-to-float transition overhead at layer boundaries.

Additionally, HuggingFace wraps the model inside a `pipeline()` abstraction, making it difficult to access the raw `nn.Module` for quantization without breaking the pipeline's internal preprocessing.

### Results

| Model | FP32 Size | INT8 Size | Reduction | Speedup |
|-------|-----------|-----------|-----------|---------|
| whisper-tiny | ~150 MB | ~110 MB | ~25-30% | ~1.1x |
| distilbert-sst2 | ~255 MB | ~130 MB | ~45-50% | ~1.3-1.5x |

**Key Insight**: DistilBERT benefits dramatically from quantization because it's a pure encoder with 6 transformer blocks of dense `nn.Linear` layers. Whisper benefits less because its bottleneck is the autoregressive `generate()` loop (repeated small matmuls), not the individual linear layer throughput.

### Trade-off: Accuracy Impact
Dynamic quantization is "free" in the sense that it doesn't require retraining or calibration data. However, it introduces small numerical differences due to INT8 rounding. For DistilBERT sentiment analysis, this is negligible — the model's confidence scores change by <0.1%. For Whisper, we haven't measured WER impact formally, but informal tests show identical transcriptions on clean audio. Under noisy conditions, the INT8 model might diverge slightly, which is an area for future investigation.

---

## 3. Streaming Audio Processing

### The Problem
The pipeline loads the entire audio file into memory before processing. For a 3-second emotion sample, this is fine. For a 2-hour podcast or meeting recording, this would require ~700MB of raw audio data in RAM *before* any model processing.

### The Solution: StreamingAudioLoader
`optimization/streaming_audio.py` implements a chunked audio loader with configurable parameters:
- **chunk_duration**: Size of each window (default: 30s)
- **overlap_duration**: Overlap between consecutive windows (default: 5s)
- **target_sr**: Resampling target (default: 16kHz)

The loader uses a Python **generator** pattern, yielding `AudioChunk` objects lazily:

```python
loader = StreamingAudioLoader(chunk_duration=30.0, overlap_duration=5.0)
for chunk in loader.load_chunks("podcast.wav"):
    result = asr_pipeline(chunk.data)
    # process incrementally...
```

### Overlap Merging Strategy
When transcribing in chunks, the boundary between two chunks can split a word or sentence, causing duplication or truncation. The `merge_transcriptions()` function handles this by:
1. Taking the last N words of chunk[i] and the first N words of chunk[i+1]
2. Finding the longest common suffix-prefix match
3. Deduplicating the overlap

**Limitation**: This word-level dedup works well on clean audio but fails when Whisper hallucinates at chunk boundaries (which happens under noisy conditions, per Experiment 5's findings on babble noise). A more robust approach would use timestamp-based alignment, but that's out of scope for this deliverable.

---

## 4. What We Tried That Didn't Work

### ONNX Export (Optional Task — Not Delivered)
We investigated exporting models to ONNX Runtime for faster CPU inference. The challenges:

1. **Whisper is an encoder-decoder model**. ONNX expects static computation graphs, but `generate()` involves dynamic loops (autoregressive decoding), KV-caching, and beam search. HuggingFace's `optimum` library partially supports this by splitting into 3 separate ONNX graphs (encoder, decoder, decoder-with-past), but the resulting files are larger than the original PyTorch model.

2. **Wav2Vec2 audio pipeline preprocessing**. The HuggingFace `pipeline()` object includes feature extraction (resampling, normalization) that runs in Python, not in ONNX. Exporting just the model without the pipeline means reimplementing the preprocessing.

3. **Marginal expected gains**. Based on literature, ONNX Runtime typically provides 1.2-2x speedup over PyTorch on CPU. Given that our Whisper inference is already ~2s and the complexity of maintaining ONNX exports, we decided the engineering cost wasn't justified for this project.

**Recommendation**: For production deployment, use Whisper.cpp (C++ implementation) or whisper-ctranslate2 instead of ONNX. These are purpose-built for CPU inference and offer 3-5x speedups.

---

## 5. Reproducibility

```bash
# Run quantization benchmark (downloads models on first run)
python optimization/quantize_model.py

# Profile the full pipeline
python optimization/profiler.py

# Profile with a real audio file
python optimization/profiler.py --audio data/emotion_samples/03-01-05-02-01-01-01.wav

# Test streaming chunker
python optimization/streaming_audio.py --input data/emotion_samples/03-01-01-01-01-01-01.wav --chunk-size 5 --overlap 1

# Run unit tests
pytest tests/test_optimization.py -v
```

---

## 6. Honest Assessment

**What this deliverable proves**: INT8 dynamic quantization gives meaningful size reduction for transformer encoders (~45% for DistilBERT) with negligible accuracy impact. The streaming chunker makes the pipeline viable for long-form audio. The profiler identifies Whisper as the bottleneck.

**What this deliverable does NOT prove**: We haven't measured the WER/CER impact of quantization under noisy conditions. We haven't benchmarked on actual edge hardware (Raspberry Pi, Jetson Nano). The profiler's RAM numbers are approximate due to tracemalloc limitations.

**If we had more time**: We'd export to Whisper.cpp for a real 3-5x CPU speedup, implement GPU memory management for sequential model loading/unloading, and run a formal accuracy-vs-compression trade-off study.
