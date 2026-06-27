# 🎬 Video Script — Role 4: Data & Experimentation Engineer
### Topic 3: Local Audio Preprocessing for Better ASR Performance

> **Target duration:** ~2 min 15 s (of a total ≥10 min group video)  
> **Language:** English  
> **Tone:** Research-oriented, scientific but accessible  
> **Format conventions:**  
> - `[VISUAL: ...]` → screen / slide to show at that moment  
> - `[PAUSE]` → beat/pause for emphasis  
> - `⏱ ~X s` → estimated speaking time for the segment  

---

## 🎬 SEGMENT 1 — Introduction & Research Hypotheses
> ⏱ ~25 s

---

`[VISUAL: slide_role4_title.png — Title slide "Role 4: Data & Experimentation Engineer"]`

---

> This section covers **Role 4: Data & Experimentation**. 
>
> The core research question is: **does applying classical DSP noise reduction before feeding audio to a modern AI speech recognizer actually help?**
>
> To answer this, four falsifiable hypotheses were tested.
>
> [PAUSE]
>
> - **H1**: Classical filters degrade ASR on real-world non-stationary noise.
> - **H2**: Denoising destroys Speech Emotion Recognition accuracy by smoothing prosody.
> - **H3**: Decoder perplexity predicts ASR hallucinations in low-quality conditions.
> - **H4**: Parallel routing and multimodal calibration recover emotion recognition accuracy.

---

## 🎬 SEGMENT 2 — Model Choice & Experimental Design
> ⏱ ~25 s

---

`[VISUAL: slide_model_comparison.png — Whisper-tiny vs Faster-Whisper vs WhisperX / DistilBERT / Wav2Vec2]`

---

> For ASR, **Whisper-tiny** was selected. Unlike Faster-Whisper, the standard PyTorch implementation provides full access to token-level log-probabilities, which is required to calculate decoder perplexity.
>
> The experimental pipeline evaluates **6 noise conditions** — clean baseline, white noise, pink noise, real urban noise from the DEMAND database, babble noise, and speech emotion recognition. 
>
> Each was evaluated across three preprocessing configurations and three SNR levels, totaling over **900 model inferences**.

---

## 🎬 SEGMENT 3 — The Lab-to-Real-World Gap (ASR Results)
> ⏱ ~35 s

---

`[VISUAL: visuals/all_noise_types_comparison.png — 4 noise types side by side]`

---

> The results reveal a significant lab-to-real-world gap.
>
> Under stationary **white noise** at 5 dB SNR, Wiener filtering is beneficial, reducing Word Error Rate from 27.5% to 24.7%.
>
> However, on non-stationary noise, the Wiener filter severely degrades performance.
>
> On **pink noise**, WER jumps from 22% raw to **33%** after filtering — a 11-point degradation. This is caused by spectral tilt, where the filter over-attenuates high frequencies.
>
> On **urban noise**, WER increases from 26% to **35.5%** because the filter's noise estimate lags behind transient acoustic events.
>
> By distorting the noise context, classical DSP deprives Whisper of acoustic patterns it expects to decode. This confirms hypothesis **H1**.

---

## 🎬 SEGMENT 4 — Babble Noise & Hallucination Detection
> ⏱ ~25 s

---

`[VISUAL: visuals/perplexity_vs_wer_scatter.png — scatter plot PPL vs WER showing hallucination cluster]`

---

> Under severe **babble noise**, ASR models frequently suffer from autoregressive hallucinations, generating entirely fabricated sentences and exceeding a 100% Word Error Rate.
>
> To detect this, the ASR **decoder perplexity** was extracted. 
>
> While non-hallucinated runs averaged a perplexity of **898**, hallucinated runs averaged **34,881**. 
>
> Setting a detection threshold at **10,000** identifies hallucinations with **100% recall**, providing a robust production monitoring signal. This confirms hypothesis **H3**.

---

## 🎬 SEGMENT 5 — The SER Conflict & Parallel Routing
> ⏱ ~25 s

---

`[VISUAL: visuals/emotion_accuracy.png — SER accuracy comparison and Parallel routing diagram]`

---

> Evaluating Speech Emotion Recognition reveals a fundamental conflict: **what helps speech transcription is destructive to emotion recognition**.
>
> At 5 dB SNR, Wiener filtering drops SER accuracy from 45.8% to **24.4%**. The filter acts as an "emotional eraser," smoothing out the pitch micro-variations and prosody that define emotional speech.
>
> To resolve this, a **parallel routing architecture** was implemented, splitting the raw audio: the denoised stream goes to ASR, while the clean, original audio goes to SER. 
>
> A multimodal fusion engine then refines emotional predictions, achieving a **+20% relative accuracy gain**. This confirms hypotheses **H2** and **H4**.

---

## 🎬 SEGMENT 6 — SOTA & Conclusion
> ⏱ ~15 s

---

`[VISUAL: visuals/slide_hypothesis_validation.png — All 4 hypotheses confirmed summary]`

---

> Comparing this to the state-of-the-art, neural enhancement like RNNoise improves ASR but still degrades emotion recognition prosody, making parallel routing essential.
>
> The key conclusion is that **classical preprocessing should not be applied by default**. Modern neural speech recognition is robust, and multi-task voice pipelines must use parallel routing and perplexity monitoring to remain reliable in production.

---

# 📸 Visual Checklist — Role 4

All visual assets are available in the `visuals/` directory. No manual screenshots needed.

## ✅ Complete Visual Checklist

| # | Segment | File | Status | Description |
|---|---|---|:---:|---|
| 1 | Seg 1 — title | `visuals/slide_role4_title.png` | ✅ **Ready** | Title slide + pipeline overview |
| 2 | Seg 1 — hypotheses | `visuals/slide_hypotheses.png` | ✅ **Ready** | H1–H4 research hypotheses table |
| 3 | Seg 2 — model choice | `visuals/slide_model_comparison.png` | ✅ **Ready** | Whisper-tiny vs Faster-Whisper vs WhisperX |
| 4 | Seg 2 — design | `visuals/slide_experiment_design.png` | ✅ **Ready** | 6 experiments, 900+ inferences overview |
| 5 | Seg 3 — results | `visuals/all_noise_types_comparison.png` | ✅ **Ready** | **KEY VISUAL** — 4 noise types side by side |
| 6 | Seg 4 — perplexity | `visuals/perplexity_vs_wer_scatter.png` | ✅ **Ready** | **KEY VISUAL** — PPL vs WER hallucination cluster |
| 7 | Seg 5 — SER | `visuals/emotion_accuracy.png` | ✅ **Ready** | SER accuracy drop + parallel routing diagram |
| 8 | Seg 6 — conclusion | `visuals/slide_hypothesis_validation.png` | ✅ **Ready** | All 4 hypotheses confirmed summary |

**Total: 8 visuals, all generated and ready — zero manual work needed.**

---

## 🎥 Optional Screen Recording

Run: `streamlit run demo/app.py` and record:
1. Select a RAVDESS sample (e.g. angry tone) → show SER prediction = angry
2. Apply Wiener filter → show SER prediction changes (accuracy drops)
3. Switch to live recording → say a happy phrase with sarcastic tone → show sarcasm alert fires

---

## 🎬 Recommended Video Editing Flow
1. Segment 1: `slide_role4_title.png` → `slide_hypotheses.png` (25 s)
2. Segment 2: `slide_model_comparison.png` → `slide_experiment_design.png` (25 s)
3. Segment 3: `all_noise_types_comparison.png` (35 s)
4. Segment 4: `perplexity_vs_wer_scatter.png` (25 s)
5. Segment 5: `emotion_accuracy.png` (25 s)
6. Segment 6: `slide_hypothesis_validation.png` (15 s)

## 📦 Delivery Checklist
- [ ] Record narration in English following the script above
- [ ] 1080p minimum resolution
- [ ] Overlay key numbers as text captions: `−21.4% SER`, `PPL > 10,000`, `+20% relative gain`
- [ ] Ambient lo-fi music at −20 dB under voice (optional)
- [ ] Upload to shared drive and link in `submission.txt`
