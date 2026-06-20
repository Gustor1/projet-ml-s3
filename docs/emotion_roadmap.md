# 🗺️ Action Plan: Emotion & Sarcasm Integration (Extension "Fun")

This roadmap documents the transition of our project towards integrating **Speech Emotion Recognition (SER)** and **Sarcasm Detection** to satisfy the "fun & fancy" project requirements, tracking what has been completed and what is left to do.

---

## 📌 Context & Motivation
To stand out and make the project engaging for the final presentation, we extended the baseline audio preprocessing and ASR pipeline (Topic 3) with a **vocal emotion recognition** capability. By combining this with **text sentiment analysis**, we created a multimodal pipeline that can detect **sarcasm and passive-aggressive behavior** (e.g., positive statements spoken in an angry voice).

---

## 🏆 Part 1: What Has Already Been Done

### 1. Dataset & Noise Augmentation Pipeline
- **Wrote** `scripts/download_emotion_samples.py`: Downloader and prep script for RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) Actor 01 emotional voice files.
- **Wrote** `scripts/augment_emotion_noise.py`: Injects white Gaussian noise and real urban noise (traffic, street cafe) at different SNR levels (20dB and 5dB) to test the robustness of emotion models.

### 2. Scientific Evaluation & Benchmarking
- **Wrote** `experiments/evaluate_emotion_robustness.py`: Compares Speech Emotion Recognition (SER) classification accuracy using `superb/wav2vec2-base-superb-er` across clean, noisy, and preprocessed audio.
- **Wrote** `scripts/generate_emotion_visuals.py`: Generates the final comparison charts showing accuracy trends (`visuals/emotion_accuracy.png`).
- **Extracted key findings**:
  - Found that the Wiener filter heavily degrades emotion classification (oversmoothes pitch/prosody).
  - Identified that spectral subtraction creates artifact-driven "false accuracy" under white noise.

### 3. Multimodal Sarcasm Detection Logic
- **Wrote** `experiments/sarcasm_detector.py`: Combines:
  1. **ASR (Whisper)** to transcribe speech to text.
  2. **NLP Sentiment (DistilBERT)** to classify the sentiment of the transcribed words (positive/negative).
  3. **SER (Wav2Vec2)** to classify the vocal emotion (neutral, happy, sad, angry).
  4. **Mismatch Checker** to trigger an alert if a positive sentence is spoken in an angry/sad voice, or a negative sentence in a happy voice.

### 4. Technical Documentation & Translations
- **Wrote** `docs/experiment-6-emotions.md`: Full scientific report documenting setup, results, and root-cause analysis.
- **Wrote** `docs/emotion_integration.md`: A step-by-step developer integration guide.
- **Cleaned & translated** all documentation (`submission.txt`, `role.md`, `insights.md`, `tradeoffs.md`) from French to English.

---

## 📋 Part 2: What Still Needs To Be Done

### 1. Streamlit Interactive Web Application (`demo/app.py`)
To make the project interactive and visually impressive for the final grading/video presentation:
- [ ] Add `streamlit` to `requirements.txt`.
- [ ] Develop the user interface in `demo/app.py`:
  - **Audio Recording/Upload Component**: Allow recording directly from the browser microphone or uploading WAV files.
  - **DSP Preprocessing Live Switch**: Toggle between "None", "Wiener Filter", and "Spectral Subtraction" and display updated waveforms and spectrograms.
  - **ASR Transcription Display**: Show transcribed text in real-time.
  - **Visual Emotion Indicators**: Display the dominant emotion with giant animated emojis (😡, 😄, 😢, 😐) and confidence progress bars.
  - **Sarcasm Alert Flash Banner**: Add a shiny banner showing an warning when sarcasm is detected.

### 2. Multi-Model Profiling & Latency Optimization
- [ ] Run profiling on the multi-model pipeline (ASR + NLP + SER) running concurrently.
- [ ] Optimize memory footprint (e.g., share GPU memory or fall back to CPU if memory is limited).

### 3. Video Presentation Recording
- [ ] Record the final 10-minute presentation in English, showcasing the interactive Streamlit dashboard.
