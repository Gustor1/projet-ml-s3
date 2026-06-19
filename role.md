# 🎙️ Role Distribution — ASR & Audio Preprocessing Project (Topic 3)
> **Selected Topic:** `Local Audio Preprocessing for Better ASR Performance`  
> **Group:** 6 students | **Final Video:** ≥ 10 min, in English, fun & fancy  
> **Professor's Golden Rule:** GitHub history will be checked. Every member must have regular and visible technical commits.

---

## 📌 Context & Project Goal
Build a complete pipeline that **preprocesses audio locally** (noise reduction, VAD, enhancement, echo cancellation, etc.) to **significantly improve the performance of an ASR model** (Whisper, Wav2Vec, etc.).  
To make the project **fun and scientifically premium**, we have extended it with a **Speech Emotion Recognition (SER) and Sarcasm Detection pipeline**, comparing how classical DSP filters affect non-verbal vocal cues versus verbal transcription.

---

## 👥 Adapted Rôles & Technical Missions

### 1️⃣ Pipeline Architect & DevOps
- 🎯 **Mission:** Structure the project, manage environmental configuration, and ensure containerization (Docker) and dependency management for both standard DSP and heavy Deep Learning models.
- 💻 **Expected Commits:** `main.py`, `config.yaml`, `utils/`, `requirements.txt`, `Dockerfile`, model caching mechanisms for deployment, GitHub Actions.
- 📊 **Contribution:** ~17%
- 🛠️ **Stack:** Python, YAML, Docker, Hugging Face Hub (Model caching).

### 2️⃣ Audio Preprocessing Engineer
- 🎯 **Mission:** Develop audio preprocessing algorithms (denoising, VAD, spectral subtraction) and export them as clean, importable APIs to be run in real-time by the demo dashboard.
- 💻 **Expected Commits:** `preprocessing/denoise.py`, `preprocessing/vad.py`, `preprocessing/spectral_subtraction.py`, `preprocessing/pipeline.py`, unit tests.
- 📊 **Contribution:** ~17%
- 🛠️ **Stack:** `librosa`, `noisereduce`, `webrtcvad`, `scipy.signal`.

### 3️⃣ ASR Integration & Evaluation Engineer
- 🎯 **Mission:** Integrate ASR models (Whisper, Wav2Vec2), compute WER/CER benchmarks, and write helper functions to run Whisper inference on the fly for the web dashboard.
- 💻 **Expected Commits:** `asr/whisper_wrapper.py`, `asr/wav2vec_wrapper.py`, `asr/evaluator.py`, benchmark scripts, multilingual support (EN/FR/ZH).
- 📊 **Contribution:** ~17%
- 🛠️ **Stack:** `transformers`, `jiwer`, `openai-whisper`.

### 4️⃣ Experimentation & Data Engineer (Eliott)
- 🎯 **Mission:** Set up dataset pipelines, design ASR & SER experiments, benchmark Wav2Vec2 SER robustness under noise, and implement the multi-modal sarcasm detection algorithm.
- 💻 **Expected Commits:** `scripts/download_emotion_samples.py`, `scripts/augment_emotion_noise.py`, `experiments/evaluate_emotion_robustness.py`, `experiments/sarcasm_detector.py`, analysis notebooks, `results/` CSV generation.
- 📊 **Contribution:** ~17%
- 🛠️ **Stack:** PyTorch, `transformers` (Wav2Vec2, DistilBERT), `pandas`, `matplotlib`, `seaborn`.

### 5️⃣ Optimization & Real-Time Performance Engineer
- 🎯 **Mission:** Profile the multi-model execution (ASR Whisper + SER Wav2Vec2 + NLP DistilBERT), optimize memory footprint (GPU memory allocation), and analyze overall latency.
- 💻 **Expected Commits:** `optimization/profiler.py`, `optimization/streaming_audio.py`, model quantization scripts (`optimization/quantize_model.py`), CPU-only profiling reports.
- 📊 **Contribution:** ~16%
- 🛠️ **Stack:** `torch.profiler`, `onnxruntime`, `pydub`.

### 6️⃣ Demo, Visualization & Video Production Engineer
- 🎯 **Mission:** Build the final interactive Streamlit web dashboard, develop real-time spectrogram/waveform visualizations, and produce the final presentation video (≥10 min).
- 💻 **Expected Commits:** `demo/app.py` (Streamlit interface), `demo/export_demo.py`, `demo/spectrogram_viz.py`, video recording scripts and visual assets.
- 📊 **Contribution:** ~16%
- 🛠️ **Stack:** `streamlit`, `ffmpeg`, `moviepy`, `plotly`/`matplotlib` (real-time plots).

---

## 🛠️ GitHub Rules & Workflow (MANDATORY)
The professor will check the commit history. To ensure everything is validated:

| Rule | Detail |
|-------|--------|
| 🌿 Branches | Each member works on their own branch: `feature/role-name` |
| 📝 Commits | Minimum **2 to 3 commits/week/person** (even small ones: `fix:`, `docs:`, `test:`) |
| 🔀 Merge | All integrations go through **Pull Requests** with reviews |
| 🤖 AI | Use of Copilot/Cursor/Qwen is highly encouraged, but **insights & trade-offs must be documented** |
| 📁 Structure | Respect the defined folder structure. No files outside of assigned folders |
| 📊 Traceability | Each GitHub issue = one task assigned to a specific person |

---

## 📦 Final Submission Checklist
- [ ] Final video `≥ 10 min`, in **English**, fun & fancy (demonstrating the Streamlit app)
- [ ] Clean, structured source code, hosted on GitHub (URL provided)
- [ ] `submission.txt` file indicating roles + % contribution per member (fully in English)
- [ ] Balanced GitHub history (visible commits for all 6)
- [ ] Zipped folder → Upload to Google Drive → Link sent **in private** (WeChat/Email)

---

## 🚀 Next Steps
1. ✅ Each member **validates their role** in this file
2. 🍴 Fork the base repository → Create dedicated branch
3. 💻 Environment setup (WSL2/Docker/AutoDL) → **First commit within 48h**
4. 🗓️ 30-minute weekly sync (progress + blockers)
5. 🎬 Final video editing 2 weeks before the deadline

---
🔗 *GitHub Repo:* [To be completed]  
📅 *Deadline:* First Friday of SHU exam weeks, 23:59  
👥 *Group:* 6 students | *Topic:* 3 — Local Audio Preprocessing for Better ASR