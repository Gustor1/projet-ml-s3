# 🎬 Video Script — Role 3: ASR Integration & Cross-Modal Evaluation
### Topic 3: Local Audio Preprocessing for Better ASR Performance

> **Target duration:** ~2 min 00 s (of a total ≥10 min group video)  
> **Language:** English  
> **Tone:** Research-oriented, scientific but accessible  
> **Format conventions:**  
> - `[VISUAL: ...]` → screen / slide to show at that moment  
> - `[PAUSE]` → beat/pause for emphasis  
> - `⏱ ~X s` → estimated speaking time for the segment  

---

## 🎬 SEGMENT 1 — Introduction & The Core Problem
> ⏱ ~25 s

---

`[VISUAL: visuals/slide_r3_title.png — Role 3 title slide + pipeline diagram]`

---

> This section covers **ASR Integration and Cross-Modal Evaluation**.
>
> In multimodal voice applications, Automatic Speech Recognition is the initial step, feeding directly into downstream sentiment and sarcasm classifiers. 
>
> This creates a critical vulnerability: the **error cascade effect**.
>
> [PAUSE]
>
> A single phonetic substitution in the ASR layer can flip text sentiment predictions and trigger false sarcasm alerts downstream. The key research question is: **how much does ASR quality affect multimodal pipeline reliability, and is Word Error Rate a sufficient metric to measure this impact?**

---

## 🎬 SEGMENT 2 — Model Selection & Architecture Design
> ⏱ ~25 s

---

`[VISUAL: visuals/slide_r3_model_selection.png — Whisper vs Faster-Whisper vs WhisperX / DistilBERT vs RoBERTa vs VADER / Wav2Vec2 vs WavLM vs HuBERT]`

---

> To investigate this cascade, three models were integrated into a unified Python architecture.
>
> For ASR, **Whisper** by Radford et al. (2023) was chosen for its zero-shot generalization. Compiled engines like Faster-Whisper were ruled out because they restrict access to the internal token-level log-probabilities required for perplexity analysis. 
>
> For text sentiment, **DistilBERT** (Sanh et al., 2019) was selected, offering 97% of BERT's performance while being 40% smaller and 60% faster. 
>
> For Speech Emotion Recognition, **Wav2Vec2-base-superb-er** was used, which remains the standard benchmark for acoustic emotion extraction.

---

## 🎬 SEGMENT 3 — Architecture & Sarcasm Logic
> ⏱ ~20 s

---

`[VISUAL: visuals/slide_r3_pipeline_arch.png — BaseASR API, WhisperWrapper, Evaluator, Sarcasm Logic]`

---

> The ASR subsystem uses a unified `BaseASR` abstract interface, allowing modular model swapping. Standardized evaluation is handled by an `Evaluator` using `jiwer` with consistent text normalization.
>
> The downstream sarcasm engine operates on a **multimodal incongruity principle**, detecting sarcasm when emotional voice cues contradict verbal sentiment. For instance, a positive text sentiment paired with a sad or angry voice triggers a sarcasm flag.

---

## 🎬 SEGMENT 4 — The Error Cascade: A Concrete Example
> ⏱ ~25 s

---

`[VISUAL: visuals/slide_r3_error_cascade.png — 3-step cascade diagram: ASR error → sentiment flip → sarcasm false positive]`

---

> Consider a concrete error cascade from the RAVDESS experiments. 
>
> The clean audio input is *"Kids are talking by the door"*, spoken in a happy tone. 
>
> Whisper-tiny transcribes this as *"Kids are talking by the **dollar**"*, introducing a 16.7% Word Error Rate.
>
> [PAUSE]
>
> DistilBERT receives *"by the dollar"* and misclassifies the text sentiment as **negative**. The sarcasm engine then pairs this negative sentiment with the happy voice, triggering a **false positive sarcasm alert**. 
>
> This proves that standard WER is an insufficient metric; the semantic weight and location of errors are critical.

---

## 🎬 SEGMENT 5 — Ablation Results: The Numbers
> ⏱ ~25 s

---

`[VISUAL: visuals/slide_r3_ablation_results.png — Results table: Whisper tiny vs base vs small, WER / Flip Rate / FP Rate / Agreement]`

---

> To measure this cascade systematically, a cross-modal ablation study was run comparing Whisper tiny, base, and small.
>
> **Whisper-tiny** yielded an 8.33% WER, causing a **10.71% Sentiment Flip Rate** and a **7.14% Sarcasm False Positive Rate**.
>
> **Whisper-base** dropped the WER to 2.38%, completely eliminating downstream cascade errors with a **0.00% Flip Rate** and **0.00% False Positive Rate**.
>
> **Whisper-small** achieved a 0.00% WER, yielding identical downstream results to base.
>
> Upgrading to Whisper-base represents a crucial engineering trade-off: it doubles inference latency but achieves perfect downstream stability.

---

## 🎬 SEGMENT 6 — Conclusion & Future Directions
> ⏱ ~20 s

---

`[VISUAL: visuals/slide_r3_conclusion.png — 4-card conclusion summary]`

---

> Key limitations of this study include the simple vocabulary of the RAVDESS dataset. Future research should evaluate richer corpora.
>
> Architecturally, pipelines should move from sequential processing to **Modality-Gated Fusion**, where downstream models dynamically weight ASR outputs based on decoder confidence scores, and ultimately to end-to-end audio language models that bypass discrete text representation entirely.

---

# 📸 Visual Checklist — Role 3

| # | Segment | File | Status | Description |
|---|---|---|:---:|---|
| 1 | Seg 1 — title | `visuals/slide_r3_title.png` | ✅ **Ready** | Role 3 title + pipeline overview diagram |
| 2 | Seg 2 — model selection | `visuals/slide_r3_model_selection.png` | ✅ **Ready** | Whisper / DistilBERT / Wav2Vec2 choice rationale |
| 3 | Seg 3 — architecture | `visuals/slide_r3_pipeline_arch.png` | ✅ **Ready** | BaseASR API, Evaluator, Sarcasm Logic |
| 4 | Seg 4 — error cascade | `visuals/slide_r3_error_cascade.png` | ✅ **Ready** | **KEY VISUAL** — 3-step cascade: door→dollar→sarcasm |
| 5 | Seg 5 — ablation results | `visuals/slide_r3_ablation_results.png` | ✅ **Ready** | **KEY VISUAL** — tiny vs base vs small results table |
| 6 | Seg 6 — conclusion | `visuals/slide_r3_conclusion.png` | ✅ **Ready** | 4 key contributions summary |

**Total: 6 slides, all generated and ready — zero manual work needed.**

---

## 🎥 Optional Screen Recording

Run `streamlit run demo/app.py` and demonstrate:
1. Select a RAVDESS sample with **happy** emotion + **negative** text (e.g., "Dogs are sitting")
2. Show sarcasm alert fires correctly (ground truth: sarcastic)
3. Switch to Wiener filter → show how preprocessing affects the SER vocal emotion output → sarcasm verdict may change

---

## 🎬 Recommended Video Editing Flow
1. Segment 1: `slide_r3_title.png` (25 s)
2. Segment 2: `slide_r3_model_selection.png` (25 s)
3. Segment 3: `slide_r3_pipeline_arch.png` (20 s)
4. Segment 4: `slide_r3_error_cascade.png` — **hold 5 s extra on the cascade diagram after narration** (30 s)
5. Segment 5: `slide_r3_ablation_results.png` — **KEY SLIDE, hold 5 s on the results table after narration, zoom on the 0.00% Flip Rate row** (30 s)
6. Segment 6: `slide_r3_conclusion.png` (20 s closing)

## 📦 Delivery Checklist
- [ ] Record narration in English following the script above
- [ ] 1080p minimum resolution
- [ ] Overlay key numbers as text captions: `10.71% → 0.00% Flip Rate`, `"door" → "dollar"`, `100% sarcasm agreement`
- [ ] Ambient lo-fi music at −20 dB under voice (optional)
