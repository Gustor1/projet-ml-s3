# 🎬 Video Script — Role 5: Performance Profiling & Optimization

> **Target duration:** ~2 min 15 s (of a total ≥10 min group video)  
> **Language:** English  
> **Tone:** Engineering-focused, practical and data-driven  
> **Format conventions:**  
> - `[VISUAL: ...]` → screen / slide to show at that moment  
> - `[PAUSE]` → beat/pause for emphasis  
> - `⏱ ~X s` → estimated speaking time for the segment  

---

## 🎬 SEGMENT 1 — Introduction & Baseline Latency
> ⏱ ~30 s

---

`[VISUAL: visuals/slide_r5_title.png — Role 5 title slide + CPU vs GPU icon]`

---

> This section covers **Performance Profiling and Optimization**.
>
> When designing a multimodal pipeline with three sequential neural networks—Whisper, DistilBERT, and Wav2Vec2—the most pressing engineering question is: **can this run in real-time on edge devices without GPU acceleration?**
>
> [PAUSE]
>
> To answer this, we first benchmarked the unoptimized PyTorch baseline on a standard CPU. The results were highly encouraging. For a standard 3-second audio sample, the total inference time across all three models was **1.3 seconds**. 
>
> This yields a **Real-Time Factor of 0.4x**, meaning our pipeline processes speech 2.5 times faster than real-time purely on CPU power.

---

## 🎬 SEGMENT 2 — The Bottleneck & INT8 Quantization
> ⏱ ~35 s

---

`[VISUAL: visuals/slide_r5_quantization.png — Quantization Before/After chart showing MB size reduction and speedup]`

---

> However, the baseline profiling revealed a critical pain point: **Model Loading Time and Memory**. Initializing three full-precision FP32 models takes over 26 seconds and consumes significant memory bandwidth.
>
> To solve this, we implemented **INT8 Dynamic Quantization** using PyTorch. By compressing the linear layers of Whisper and DistilBERT from 32-bit floats down to 8-bit integers, we achieved substantial gains.
>
> [PAUSE]
>
> Model sizes on disk dropped significantly. More importantly, this compression directly translates to much faster loading times and even lower inference latency, making the models significantly lighter on edge CPU caches.

---

## 🎬 SEGMENT 3 — Streaming Audio for Infinite Context
> ⏱ ~30 s

---

`[VISUAL: visuals/slide_r5_streaming.png — Diagram of chunked audio processing with overlap deduplication]`

---

> Beyond static files, edge applications like live podcasts or meetings require continuous processing. 
>
> Loading a 30-minute audio file directly into memory would instantly crash an edge device. Therefore, we engineered a **chunked audio processing engine**. 
>
> The pipeline dynamically slices incoming audio into fixed windows—for example, 5 seconds—with a 1-second overlap region to preserve context across word boundaries. These chunks are processed sequentially, and the results are merged using a deduplication strategy. This ensures the memory footprint remains absolutely flat, regardless of whether the audio is 3 seconds or 3 hours long.

---

## 🎬 SEGMENT 4 — Conclusion
> ⏱ ~15 s

---

`[VISUAL: visuals/slide_r5_conclusion.png — 3 key takeaways: Fast Inference, INT8 Compression, Infinite Streaming]`

---

> Ultimately, through rigorous profiling, INT8 quantization, and streaming memory management, we proved that this complex multimodal AI pipeline is not just a theoretical construct—it is fully viable for real-world, real-time edge deployment.

---

# 📸 Visual Checklist — Role 5

| # | Segment | File | Status | Description |
|---|---|---|:---:|---|
| 1 | Seg 1 — title | `visuals/slide_r5_title.png` | ⚠️ **To generate** | Role 5 title + profiling overview |
| 2 | Seg 2 — quantization | `visuals/slide_r5_quantization.png` | ⚠️ **To generate** | Chart: FP32 vs INT8 size and latency |
| 3 | Seg 3 — streaming | `visuals/slide_r5_streaming.png` | ⚠️ **To generate** | Chunking diagram with overlapping windows |
| 4 | Seg 4 — conclusion | `visuals/slide_r5_conclusion.png` | ⚠️ **To generate** | Summary of optimization techniques |

---

## 🎥 Optional Screen Recording

To make the video dynamic, you can run and record:
1. `python optimization/profiler.py` to show the 1.3s fast inference in action.
2. `python optimization/quantize_model.py` to show the console output confirming INT8 size reductions.
3. `python optimization/streaming_audio.py` processing a long audio file in chunks without memory spikes.

## 📦 Delivery Checklist
- [ ] Record narration in English following the script above.
- [ ] Add the slide visuals to the video editor.
- [ ] Optional: overlay screen recordings of the terminal logs.
