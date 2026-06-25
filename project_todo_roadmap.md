# 📋 Project Roadmap: Remaining Tasks & Deliverables by Role

This document details the checklist of tasks remaining to finalize the Audio Preprocessing, ASR & Vocal Emotion Integration Project (Topic 3), structured by group roles. All roles have been aligned to support the joint multimodal pipeline (ASR transcription + Text Sentiment + Speech Emotion Recognition + Sarcasm Detection).

---

## 👥 Remaining Tasks by Role

### 1️⃣ Pipeline Architect & DevOps
* **Objective**: Package the 3-model multimodal stack, manage global configs, and integrate the final pipeline entry point.
* **Tasks**:
  - [x] **Docker Model Caching**: Write a caching script to pre-download Hugging Face weights (`openai/whisper-tiny`, `superb/wav2vec2-base-superb-er`, and `distilbert-base-uncased-finetuned-sst-2-english`) during Docker image build. This guarantees the joint ASR+NLP+SER container executes fully offline.
  - [x] **Pipeline Entry Point (`main.py`)**: Integrate the modules inside [main.py](file:///c:/Users/eliot/projet-ml-s3/main.py) to run the full sequence: load raw audio, apply VAD, trigger parallel routes (denoised audio to ASR; normalized audio to SER), feed ASR transcription to DistilBERT, and execute sarcasm checks.
  - [x] **Config Specification (`configs/config.yaml`)**: Complete the YAML configuration to set default SNR thresholds, model paths, YIN pitch min/max frequencies, and VAD sensitivity.
  - [x] **CI Actions**: Set up basic GitHub Actions workflows for format checkers (linting) and unit testing.

### 2️⃣ Audio Preprocessing Engineer (Split: Bilel & Elio)
* **Objective**: Export DSP filters, implement ASR vs. SER routing, and code VAD helpers.
* **Tasks**:
  - [x] **Extract Denoising APIs (`preprocessing/denoise.py`)**: Move the Wiener filter and Spectral Subtraction implementations out of the demo code into importable modular functions.
  - [x] **Parallel Stream Routing**: Adapt the pipeline to support parallel routes: a **Denoised Stream** (optimized for ASR transcription, as classical filters clean static noise) and a **Normalized Stream** (optimized for SER, as classical filters destroy pitch prosody but peak scaling + silent margin trimming preserves emotion features).
  - [x] **VAD Implementation (`preprocessing/vad.py`)**: Integrate a Voice Activity Detection utility (e.g., using `webrtcvad` or energy levels) to automatically isolate voiced segments.
  - [x] **Unit Tests**: Write testing scripts to check signal properties (sample rate matches 16kHz, output amplitude bounds, etc.).

### 3️⃣ ASR Integration & Evaluation Engineer
* **Objective**: Maintain wrapper features and analyze how ASR transcription quality impacts downstream NLP sentiment.
* **Tasks**:
  - [x] **ASR Wrapper Support (`asr/whisper_wrapper.py`)**: Maintain wrapper inference capabilities with EN/FR/ZH support.
  - [x] **Cross-Modal Ablation Study**: Evaluate how ASR transcription errors cascade into text sentiment predictions. Benchmark how Whisper model size (tiny vs. base vs. small) affects sarcasm detection reliability (e.g., analyzing if ASR typos trigger false positive sarcasm alerts).


### 4️⃣ Experimentation & Data Engineer (Eliott)
* **Objective**: Complete scientific report documentation and explore dataset expansion.
* **Tasks**:
  - [x] **Download & Augment Data**: zenodo RAVDESS Actor 01 downloads and noise injection scripts.
  - [x] **SER & Sarcasm Heuristic**: Evaluation script, sarcasm detector logic, and YIN pitch extraction.
  - [x] **Multimodal Fusion Calibration**: Implementation and verification showing +20% relative SER accuracy improvement on RAVDESS (from 35.7% to 42.8%).
  - [x] **Scientific Expansion**: Expand evaluation runs from Actor 01 (28 files) to a larger subset (e.g., all 24 actors) to increase statistical significance.
  - [x] **Document Calibration Gains**: Add a short section in `docs/experiment-6-emotions.md` explaining how peak normalization and multimodal fusion solved live close-mic errors.
  - [x] **Explicit Research Hypotheses**: H1–H4 formulated and added to `docs/final_report_data_engineer.md` section 1.1, with validation summary table in section 5.4.
  - [x] **SOTA Neural Comparison**: DeepFilterNet, RNNoise, Conv-TasNet, Demucs comparison table added to `docs/final_report_data_engineer.md` §6.1, `docs/insights.md` Insight 13, and `docs/tradeoffs.md` Trade-off 6.
  - [x] **README.md enriched**: Full project README with key findings, architecture diagram, setup instructions, and experiment table.
  - [x] **GitHub URL**: Updated in `role.md` (was placeholder, now `https://github.com/Gustor1/projet-ml-s3`).

### 5️⃣ Optimization & Real-Time Performance Engineer (Split: Bilel & Elio)
* **Objective**: Quantize the 3-model pipeline, profile GPU/CPU resource allocation, and analyze execution latency.
* **Tasks**:
  - [x] **Model Quantization (`optimization/quantize_model.py`)**: [Elio] Quantize the models (Whisper-tiny, DistilBERT) to INT8 using PyTorch Dynamic Quantization to reduce memory footprints on edge CPUs. Wav2Vec2 excluded (conv-heavy architecture, <5% gains).
  - [x] **Joint Pipeline Profiling (`optimization/profiler.py`)**: [Bilel & Elio] Profile execution latency, peak RAM usage, and model sizes during joint ASR + SER + NLP multimodal inference runs.
  - [x] **Streaming Audio (`optimization/streaming_audio.py`)**: [Elio] Chunked audio loader for processing long files with overlap-aware transcription merging for real-time edge constraints.
  - [x] **ONNX Runtime (Investigated)**: [Elio] Documented as impractical for Whisper encoder-decoder architecture — recommending Whisper.cpp for production.

### 6️⃣ Demo, Visualization & Video Production Engineer
* **Objective**: Maintain the Web dashboard responsive and produce the final presentation video.
* **Tasks**:
  - [x] **Streamlit App Interface**: Finished building [app.py](file:///c:/Users/eliot/projet-ml-s3/demo/app.py) with dual columns, hyperparameter sliders, and pitch-over-spectrogram plots.
  - [ ] **Video Production (MANDATORY)**: Record, edit, and export a $\ge 10$-minute final presentation video in English demonstrating the project goals, explaining the science of the ASR-vs-SER trade-offs, and showcasing the Streamlit dashboard in action.

---

## 📦 General Submission Checklist

- [ ] **GitHub Balanced History**: Ensure all 6 members have committed code regularly on their dedicated branches (`feature/role-name`).
- [ ] **Submission File**: Verify that [submission.txt](file:///c:/Users/eliot/projet-ml-s3/submission.txt) is fully complete (Eliott and Boatel are listed; other members must fill in their deliverables).
- [ ] **Presentation Video**: File exported and uploaded (minimum 10 minutes, English).
- [ ] **Final Zip Bundle**: Create zipped project, upload to Drive, and send the link privately as instructed.
