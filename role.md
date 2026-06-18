# 🎙️ Role Distribution — ASR & Audio Preprocessing Project (Topic 3)
> **Selected Topic:** `Local Audio Preprocessing for Better ASR Performance`  
> **Group:** 6 students | **Final Video:** ≥ 10 min, in English, fun & fancy  
> **Professor's Golden Rule:** GitHub history will be checked. Every member must have regular and visible technical commits.

---

## 📌 Context & Project Goal
Build a complete pipeline that **preprocesses audio locally** (noise reduction, VAD, enhancement, echo cancellation, etc.) to **significantly improve the performance of an ASR model** (Whisper, Wav2Vec, etc.).  
The project must go beyond the basic use of APIs: it must show **technical insights**, **ablation studies**, **engineering trade-offs**, and a **functional demo**.

---

## 👥 The 6 Technical Roles & Missions

### 1️⃣ Pipeline Architect & DevOps
- 🎯 **Mission:** Structure the project, ensure module integration, manage the environment and CI/CD.
- 💻 **Expected Commits:** `main.py`, `config.yaml`, `utils/`, `requirements.txt`, `Dockerfile`, installation scripts, GitHub Actions.
- 📊 **Contribution:** ~17%
- 🛠️ **Stack:** Python, YAML, Docker/WSL, GitHub, CLI tools.

### 2️⃣ Audio Preprocessing Engineer
- 🎯 **Mission:** Develop audio preprocessing algorithms (denoising, VAD, echo cancellation, beamforming, enhancement).
- 💻 **Expected Commits:** `preprocessing/denoise.py`, `preprocessing/vad.py`, `preprocessing/beamform.py`, `preprocessing/pipeline.py`, unit tests.
- 📊 **Contribution:** ~17%
- 🛠️ **Stack:** `librosa`, `noisereduce`, `webrtcvad`, `demucs`, `scipy.signal`.

### 3️⃣ ASR Integration & Evaluation Engineer
- 🎯 **Mission:** Integrate ASR models, calculate WER/CER, compare performances with and without preprocessing.
- 💻 **Expected Commits:** `asr/whisper_wrapper.py`, `asr/wav2vec_wrapper.py`, `asr/evaluator.py`, benchmark scripts, multilingual support (EN/FR/ZH).
- 📊 **Contribution:** ~17%
- 🛠️ **Stack:** `transformers`, `jiwer`, `openai-whisper`, `pyannote` (if needed).

### 4️⃣ Experimentation & Data Engineer
- 🎯 **Mission:** Prepare datasets, design experiments, run ablation studies, generate metrics & plots.
- 💻 **Expected Commits:** `experiments/ablation_study.py`, `experiments/plot_results.py`, data loading scripts, CSV/log management, analysis notebooks.
- 📊 **Contribution:** ~17%
- 🛠️ **Stack:** `matplotlib`, `seaborn`, `pandas`, AutoDL/GPU management, `wandb` or `mlflow` (optional).

### 5️⃣ Optimization & Real-Time Performance Engineer
- 🎯 **Mission:** Profile the pipeline, reduce latency, optimize memory, test real-time streaming & quantization.
- 💻 **Expected Commits:** `optimization/profiler.py`, `optimization/streaming_audio.py`, `optimization/quantize_model.py`, CPU/edge benchmarks, performance reports.
- 📊 **Contribution:** ~16%
- 🛠️ **Stack:** `torch.profiler`, `onnxruntime`, `pydub`, `sounddevice`, Raspberry Pi / CPU-only tests.

### 6️⃣ Demo, Visualization & Video Production Engineer
- 🎯 **Mission:** Create an interactive demo, export audio/video assets, produce the final video (≥10 min in English).
- 💻 **Expected Commits:** `demo/app.py` (Streamlit/Gradio), `demo/export_demo.py`, `demo/spectrogram_viz.py`, `docs/README.md`, video editing assets.
- 📊 **Contribution:** ~16%
- 🛠️ **Stack:** `streamlit`/`gradio`, `ffmpeg`, `moviepy`, `audacity`/`davinci resolve`, Markdown/Docs.

> ✅ **Important Note:** Role 6 does not "just do video editing". They first code the demo and auto-export tools. The video is a product of the code, not an isolated task.

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
- [ ] Final video `≥ 10 min`, in **English**, fun & fancy
- [ ] Clean, structured source code, hosted on GitHub (URL provided)
- [ ] `submission.txt` file indicating roles + % contribution per member
- [ ] Balanced GitHub history (visible commits for all 6)
- [ ] Zipped folder → Upload to Google Drive → Link sent **in private** (WeChat/Email)

---

## 🚀 Next Steps
1. ✅ Each member **validates their role** in this file
2. 🍴 Fork the base repository → Create dedicated branch
3. 💻 Environment setup (WSL2/Docker/AutoDL) → **First commit within 48h**
4. 🗓️ 30-minute weekly sync (progress + blockers)
5. 🎬 Final video editing 2 weeks before the deadline

> 📩 **In case of doubt or overload**, we will adjust together in advance. The goal is a solid, well-documented project completed without stress.

---
🔗 *GitHub Repo:* [To be completed]  
📅 *Deadline:* First Friday of SHU exam weeks, 23:59  
👥 *Group:* 6 students | *Topic:* 3 — Local Audio Preprocessing for Better ASR