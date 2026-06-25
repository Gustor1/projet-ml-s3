# 🎬 Video Script — Role 5 (Part 1): Performance Profiling & Optimization

> **Target duration:** ~1 min 30 s (of a total ≥10 min group video)  
> **Language:** English  
> **Tone:** Engineering-focused, practical and data-driven  
> **Format conventions:**  
> - `[VISUAL: ...]` → screen / slide to show at that moment  
> - `[PAUSE]` → beat/pause for emphasis  
> - `⏱ ~X s` → estimated speaking time for the segment  

---

## 🎬 SEGMENT 1 — Introduction & The Real-Time Challenge
> ⏱ ~20 s

---

`[VISUAL: visuals/slide_r5_title.png — Role 5 title slide + CPU vs GPU icon]`

---

> This section covers **Performance Profiling and Optimization**.
>
> When designing a multimodal pipeline with three sequential neural networks—Whisper, DistilBERT, and Wav2Vec2—the most pressing engineering question is: **can this run in real-time on edge devices without GPU acceleration?**
>
> [PAUSE]
>
> To answer this, we explicitly chose to benchmark the unoptimized PyTorch FP32 baseline on a standard CPU. If we can prove viability here, production deployment becomes trivial.

---

## 🎬 SEGMENT 2 — Latency & Memory Results
> ⏱ ~30 s

---

`[VISUAL: visuals/slide_r5_latency_chart.png — Bar chart showing 1.3s total inference vs 3s audio length]`

---

> The profiling results were highly encouraging. 
>
> For a standard 3-second audio sample, the total inference time across all three models was **1.3 seconds**. This gives us a **Real-Time Factor of 0.4x**. In other words, our pipeline processes speech 2.5 times faster than real-time, purely on a CPU.
>
> Unsurprisingly, the bottleneck is the **Whisper ASR module**, which consumes 64% of the total inference time. The DistilBERT text sentiment analysis is extremely fast, taking under 100 milliseconds, and the Wav2Vec2 emotion recognition takes around 380 milliseconds.
>
> From a memory standpoint, the Python-allocated heap footprint during inference remained very lean, peaking at under 10 Megabytes for tensor operations.

---

## 🎬 SEGMENT 3 — The Loading Bottleneck & Transition to Quantization
> ⏱ ~25 s

---

`[VISUAL: visuals/slide_r5_loading_issue.png — 26-second loading time warning + INT8 Quantization preview]`

---

> However, the baseline profiling revealed a critical pain point: **Model Loading Time**.
>
> Initializing the three full-precision models into memory takes over 26 seconds. For interactive applications like our Streamlit dashboard, this cold-start delay is unacceptable.
>
> This bottleneck clearly demonstrates the need for weight compression. As Elio will explain next, applying INT8 Dynamic Quantization is the logical next step to solve this loading overhead while further reducing the runtime memory footprint.

---

# 📸 Visual Checklist — Role 5 (Bilel's Part)

| # | Segment | File | Status | Description |
|---|---|---|:---:|---|
| 1 | Seg 1 — title | `visuals/slide_r5_title.png` | ⚠️ **To generate** | Role 5 title + profiling overview |
| 2 | Seg 2 — latency | `visuals/slide_r5_latency_chart.png` | ⚠️ **To generate** | Inference times: Whisper 0.84s / SER 0.38s / NLP 0.09s |
| 3 | Seg 3 — loading | `visuals/slide_r5_loading_issue.png` | ⚠️ **To generate** | 26s load time vs INT8 solution |

---

## 🎥 Optional Screen Recording

Run `python optimization/profiler.py` in your terminal and record the screen to show:
1. The 26+ seconds loading time (you can speed this up in editing).
2. The rapid 1.3s inference logs appearing in real-time.
3. The final `.csv` report being generated successfully.

## 📦 Delivery Checklist
- [ ] Record narration in English following the script above.
- [ ] Record the terminal running the profiler.
- [ ] Mention the smooth transition to Elio's quantization work at the end.
