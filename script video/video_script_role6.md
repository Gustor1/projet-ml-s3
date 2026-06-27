# 🎬 Video Script — Role 6: Demo, Visualization & Video Production
### Topic 3: Local Audio Preprocessing for Better ASR Performance

> **Target duration:** ~2 min 00 s (of a total ≥10 min group video)  
> **Language:** English  
> **Tone:** Professional, engaging, and highly visual  
> **Format conventions:**  
> - `[VISUAL: ...]` → screen / slide to show at that moment  
> - `[PAUSE]` → beat/pause for emphasis  
> - `⏱ ~X s` → estimated speaking time for the segment  

---

## 🎬 SEGMENT 1 — Introduction & Dashboard Mission
> ⏱ ~25 s

---

`[VISUAL: visuals/slide_r6_title.png — Title slide "Role 6: Demo, Visualization & Video Production" with pipeline overview]`

---

> This section covers **Interactive Demonstration, Visualizations, and Video Production**.
>
> In machine learning research, presenting raw numbers and CSV files is never enough. To truly understand how local audio preprocessing influences speech models, we built a premium, real-time playground.
>
> The mission of this dashboard was twofold: first, to provide a tangible interface for testing our findings live, and second, to visualize the subtle signal degradations that lead to downstream model errors.

---

## 🎬 SEGMENT 2 — UI Design & Glassmorphism Aesthetics
> ⏱ ~20 s

---

`[VISUAL: visuals/slide_r6_dashboard.png — Title "Interactive Visuals & Multimodal Integration" showing CSS specifications]`

---

> Built entirely in Python using **Streamlit**, the application implements a custom-engineered **Glassmorphism Design System** via CSS injection.
>
> It features a deep dark indigo gradient background, modern typography from Google Fonts using the Outfit family, and semi-transparent cards with backdrop-blur filters.
>
> In addition, the interface includes micro-animations, such as a pulsing red drop-shadow alert banner to draw immediate attention when sarcasm is detected.

---

## 🎬 SEGMENT 3 — Real-Time Signal visualizations (Live Demo)
> ⏱ ~30 s

---

`[VISUAL: Screen recording of the Streamlit app showing side-by-side raw vs preprocessed waveforms, and log-frequency spectrograms with the hot-pink pitch tracking contour]`

---

> Let's look at the real-time signal analysis section.
>
> The dashboard displays the raw and preprocessed waveforms side-by-side, but the real power lies in the log-frequency spectrograms.
>
> Here, the system extracts the fundamental frequency contour, or **F0**, using the **YIN algorithm** bounded between 75 and 400 Hertz, overlaying it as a hot-pink line directly on the spectrogram.
>
> This lets the user see exactly how classical DSP filters like Wiener smoothing destroy the micro-fluctuations in pitch that Wav2Vec2 relies on for emotion detection.

---

## 🎬 SEGMENT 4 — Multimodal Pipeline & Sarcasm Alerts (Live Demo)
> ⏱ ~25 s

---

`[VISUAL: Screen recording showing a live microphone recording or RAVDESS sample selection, clicking "Run Full Pipeline", and the Sarcasm Alert banner pulsing red with details]`

---

> The dashboard is fully functional, supporting three input modes: RAVDESS samples, custom WAV uploads, or live microphone recording directly in the browser.
>
> When the user clicks **Run Full Pipeline**, a synchronized multi-model chain executes locally. 
>
> Whisper Tiny transcribes the audio, DistilBERT classifies the text sentiment, and Wav2Vec2 extracts the vocal emotion.
>
> If literal text sentiment and acoustic vocal emotion are incongruent, the custom sarcasm engine flashes a real-time warning, highlighting the exact reasoning behind the classification.

---

## 🎬 SEGMENT 5 — Calibration, Performance Metrics & Conclusion
> ⏱ ~20 s

---

`[VISUAL: Screen recording of sidebar sliders being dragged (e.g. Wiener size, Spectral Subtraction alpha) and the SNR delta / latency cards updating instantly]`

---

> Finally, users can interactively tune hyperparameters in the sidebar, modifying the Wiener window size or Spectral Subtraction oversubtraction factors, seeing the immediate visual impact on the spectrogram and the numerical change in estimated Signal-to-Noise ratio.
>
> The metrics card displays the active pipeline latency, averaging around 700 milliseconds, and highlights the relative accuracy gains of our peak calibration model.
>
> Through this interactive demo, we bridge the gap between abstract DSP equations and real-world human perception, proving the necessity of parallel audio routing.
