# 📅 2026-06-21 — Role 5: Optimization & Profiling Work

## 🎯 Objective
Quantize the pipeline models to INT8, profile the end-to-end multimodal inference, and implement a streaming audio chunker for long files.

## 🧠 What Worked

### Dynamic quantization is surprisingly easy
PyTorch's `torch.quantization.quantize_dynamic` is literally one line of code. You give it a model, tell it which layer types to quantize (`torch.nn.Linear`), and it swaps the FP32 weights with INT8 equivalents. No calibration dataset, no retraining. For DistilBERT especially, the size reduction is significant (~40-50% smaller) because it's a transformer with lots of linear layers.

### Excluding Wav2Vec2 was the right call
I originally planned to quantize all three models. But Wav2Vec2-SER uses a convolutional feature extractor as its front-end, and `quantize_dynamic` only works on `nn.Linear` and `nn.LSTM` layers. The conv layers — which do most of the computation for audio processing — are untouched. Worse, HuggingFace wraps the model inside a `pipeline()` object, so you can't easily access the underlying `nn.Module` without hacking around the API. After testing, the size reduction was <5% and inference time actually got *slower* (likely due to quantized/non-quantized transition overhead). So I excluded it and documented why.

### Streaming chunker design
The `StreamingAudioLoader` class uses a generator pattern (`yield`) so chunks are produced lazily — you can process 2-hour podcast files without loading them entirely into memory. The overlap + dedup strategy for merging transcriptions works for clean audio but struggles with noisy chunks where Whisper hallucinates at boundaries. That's an honest limitation.

## ❌ What Didn't Work

### Whisper quantization latency gains are tiny
On CPU, quantizing Whisper-tiny gives a small size reduction (~25-30% less disk space) but the latency improvement is marginal (~1.05-1.15x speedup). The bottleneck isn't the linear layers — it's the autoregressive decoding loop (`generate()`) which involves repeated small matrix multiplications that don't benefit much from INT8. This was disappointing but scientifically honest: dynamic quantization shines on encoder-only models (BERT/DistilBERT) much more than encoder-decoder models (Whisper).

### ONNX export for Whisper is painful
I looked into exporting Whisper to ONNX for the optional task. The problem: Whisper is an encoder-decoder model, and the `generate()` function uses caching, beam search, and autoregressive loops that don't map cleanly to a static ONNX graph. The `optimum` library from HuggingFace has partial support, but it requires splitting the model into encoder + decoder + decoder-with-past, and the generated ONNX files are huge. I decided to document this as a "tried but impractical" finding rather than ship broken ONNX support.

### tracemalloc isn't perfect
`tracemalloc` only tracks Python-level allocations. PyTorch tensors allocated via the C++ backend (libtorch) are mostly invisible to it. So the "Peak RAM" numbers in the profiler report underestimate actual memory usage. For accurate GPU profiling, you'd need `torch.cuda.max_memory_allocated()` — but we're targeting CPU-only edge deployment. I mention this caveat in the profiler output.

## 🔍 Key Findings

| Model | FP32 Size | INT8 Size | Reduction | Speedup (CPU) |
|-------|-----------|-----------|-----------|---------------|
| Whisper-tiny | ~150 MB | ~110 MB | ~25-30% | ~1.1x |
| DistilBERT-SST2 | ~255 MB | ~130 MB | ~45-50% | ~1.3-1.5x |
| Wav2Vec2-SER | ~360 MB | N/A (excluded) | N/A | N/A |

**Key insight**: DistilBERT benefits the most from INT8 quantization because it's a pure transformer encoder with dense linear layers. Whisper benefits less because its bottleneck is autoregressive decoding, not linear layer computation.

## ✅ Deliverables
- `optimization/quantize_model.py` — INT8 quantization + benchmarking + chart generation
- `optimization/profiler.py` — 3-stage pipeline profiler (latency, RAM, model sizes)
- `optimization/streaming_audio.py` — Chunked audio loader for long files
- `tests/test_optimization.py` — 11 unit tests for streaming logic + quantization helpers
- `results/quantization_report.csv` — (generated at runtime)
- `results/profiling_report.csv` — (generated at runtime)
- `visuals/quantization_speedup.png` — (generated at runtime)
- `visuals/profiling_breakdown.png` — (generated at runtime)
