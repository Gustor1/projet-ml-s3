# 📋 Project Roadmap: Remaining Tasks & Deliverables by Role

This document details the checklist of tasks remaining to finalize the Audio Preprocessing & ASR Project (Topic 3), structured by group roles.

---

## 👥 Remaining Tasks by Role

### 1️⃣ Pipeline Architect & DevOps
* **Objective**: Structure core imports, secure model execution inside container boundaries, and implement model caching.
* **Tasks**:
  - [ ] **Model Caching Script**: Write a script to pre-download Hugging Face weights (`openai/whisper-tiny`, `superb/wav2vec2-base-superb-er`, and `distilbert-base-uncased-finetuned-sst-2-english`) into a local directory during Docker build. This ensures the container works 100% offline.
  - [ ] **Pipeline Integration (`main.py`)**: Replace the current placeholder code in [main.py](file:///c:/Users/eliot/projet-ml-s3/main.py) with an end-to-end execution flow calling the `preprocessing` API and `asr` wrappers.
  - [ ] **Configuration File**: Complete `configs/config.yaml` to specify hyperparameters (sampling rates, default noise parameters) and model versions.
  - [ ] **GitHub Actions**: Add a basic CI pipeline to run tests or code format checks automatically.

### 2️⃣ Audio Preprocessing Engineer
* **Objective**: Move local preprocess functions into formal APIs, implement VAD, and write unit tests.
* **Tasks**:
  - [ ] **Modularize Filters**: Move the Wiener filter and Spectral Subtraction implementations out of the demo app and into `preprocessing/denoise.py`.
  - [ ] **VAD Implementation (`preprocessing/vad.py`)**: Integrate a clean Voice Activity Detection wrapper (using libraries like `webrtcvad` or simple energy metrics) to strip silent frames.
  - [ ] **Unit Tests**: Write tests verifying that preprocessors output valid signal formats (correct shape, sample rates, non-null values).

### 3️⃣ ASR Integration & Evaluation Engineer
* **Objective**: Maintain wrapper features and finalize ASR ablation studies.
* **Tasks**:
  - [ ] **ASR Wrapper finalization (`asr/whisper_wrapper.py`)**: Ensure proper support for English, French, and Chinese speech transcription.
  - [ ] **Evaluate Ablation Studies**: Summarize performance and word error rates (WER) across different model sizes (tiny vs. base vs. small vs. medium) on noisy audio.

### 4️⃣ Experimentation & Data Engineer (Eliott)
* **Objective**: Complete scientific report documentation and explore dataset expansion.
* **Tasks**:
  - [x] **Download & Augment Data**: zenodo RAVDESS Actor 01 downloads and noise injection scripts.
  - [x] **SER & Sarcasm Heuristic**: Evaluation script, sarcasm detector logic, and YIN pitch extraction.
  - [x] **Multimodal Fusion Calibration**: Implementation and verification showing +20% relative SER accuracy improvement on RAVDESS (from 35.7% to 42.8%).
  - [ ] **Scientific Expansion**: Expand evaluation runs from Actor 01 (28 files) to a larger subset (e.g., all 24 actors) to increase statistical significance.
  - [ ] **Document Calibration Gains**: Add a short section in `docs/experiment-6-emotions.md` explaining how peak normalization and multimodal fusion solved live close-mic errors.

### 5️⃣ Optimization & Real-Time Performance Engineer
* **Objective**: Profile inference latency, implement model quantization, and verify GPU/CPU metrics.
* **Tasks**:
  - [ ] **Model Quantization (`optimization/quantize_model.py`)**: Convert PyTorch models to INT8 using PyTorch Dynamic Quantization to reduce memory usage and speed up CPU inference.
  - [ ] **Execution Profiling (`optimization/profiler.py`)**: Write scripts to profile memory footprint and CPU utilization for the joint pipeline (ASR + SER + NLP).
  - [ ] **ONNX Runtime (Optional)**: Export models to ONNX to benchmark latency improvements.

### 6️⃣ Demo, Visualization & Video Production Engineer
* **Objective**: Keep the Web dashboard responsive and produce the final presentation video.
* **Tasks**:
  - [x] **Streamlit App Interface**: Finished building [app.py](file:///c:/Users/eliot/projet-ml-s3/demo/app.py) with dual columns, hyperparameter sliders, and pitch-over-spectrogram plots.
  - [ ] **Video Production (MANDATORY)**: Record, edit, and export a $\ge 10$-minute final presentation video in English demonstrating the project goals, experimental insights, and showcasing the Streamlit dashboard in action.

---

## 📦 General Submission Checklist

- [ ] **GitHub Balanced History**: Ensure all 6 members have committed code regularly on their dedicated branches (`feature/role-name`).
- [ ] **Submission File**: Verify that [submission.txt](file:///c:/Users/eliot/projet-ml-s3/submission.txt) is fully complete (Eliott and Boatel are listed; other members must fill in their deliverables).
- [ ] **Presentation Video**: File exported and uploaded (minimum 10 minutes, English).
- [ ] **Final Zip Bundle**: Create zipped project, upload to Drive, and send the link privately as instructed.
