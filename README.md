# 🎙️ Local Audio Preprocessing for Better ASR Performance

> **Topic 3** — Research-grade evaluation of classical DSP speech enhancement for edge ASR and multimodal affective computing.  
> **GitHub Repository:** [https://github.com/Gustor1/projet-ml-s3](https://github.com/Gustor1/projet-ml-s3)  
> **Group:** 6 students | Shanghai University × UTBM

---

## 📖 Overview

A multimodal pipeline that combines **Audio Preprocessing**, **ASR (Whisper)**, **Speech Emotion Recognition (Wav2Vec2)**, **Text Sentiment Analysis (DistilBERT)**, and **Sarcasm Detection** to analyze speech from both verbal and non-verbal perspectives.

This project evaluates whether classical **frequency-domain speech enhancement algorithms** (Wiener filtering, Spectral Subtraction) improve or degrade modern deep learning-based **Automatic Speech Recognition (ASR)** and **Speech Emotion Recognition (SER)** under realistic noise conditions.

We identify a critical **lab-to-real-world gap**: while Wiener filtering helps under stationary white Gaussian noise, it **degrades** ASR accuracy under colored noise (pink $1/f$), real urban noise, and babble noise — due to the violation of the stationarity assumption embedded in classical DSP design.

We also design a **parallel routing architecture** and **multimodal fusion calibration engine** to resolve the fundamental conflict between ASR (which benefits from spectral smoothing) and SER (which is destroyed by it).

---

## 🔑 Key Findings

| Experiment | Noise Type | Wiener Filter Effect (5 dB SNR) |
|---|---|---|
| Exp 2 — White Gaussian Noise | Stationary, flat PSD | ✅ **−2.75% WER** (helps) |
| Exp 3 — Pink 1/f Noise | Colored, low-freq heavy | ❌ **+11.13% WER** (degrades) |
| Exp 4 — Real Urban Noise (DEMAND) | Non-stationary, transient | ❌ **+9.31% WER** (degrades) |
| Exp 5 — Babble / Crowd Noise | Competing speech | ❌ **+8.12% WER** + hallucinations |
| Exp 6 — Speech Emotion Recognition | White Noise (5 dB) | ❌ **−21.43% accuracy** (erases prosody) |

**Hallucination Finding**: Under severe babble noise, Whisper's decoder produces hallucinations (WER ≥ 100%). Decoder perplexity (PPL > 10,000) predicts these events with **100% recall**.

**SER Calibration Gain**: A multimodal fusion heuristic (ASR text sentiment + YIN pitch tracking) boosts RAVDESS SER accuracy from 35.71% → **42.86% (+20% relative gain)**.

---

## 🏗️ Pipeline Architecture

```
                         +----------------------------+
                         |     Raw Noisy Audio        |
                         +--------------+-------------+
                                        |
                   +--------------------+--------------------+
                   |                                         |
     +-------------v-------------+           +-------------v-------------+
     |   Wiener Denoising DSP    |           |   Acoustic Calibration    |
     |  (Optimal for ASR Input)  |           |    (Trim & Normalize)     |
     +-------------+-------------+           +-------------+-------------+
                   |                                         |
     +-------------v-------------+           +-------------v-------------+
     |        Whisper ASR        |           |        Wav2Vec2 SER       |
     +-------------+-------------+           +-------------+-------------+
                   |                                         |
     +-------------v-------------+                           |
     |   DistilBERT Sentiment    |                           |
     +-------------+-------------+                           |
                   |                                         |
                   +--------------------+--------------------+
                                        |
                         +--------------v-------------+
                         |  Multimodal Fusion Engine  |
                         |  (YIN Pitch + ASR + SER)   |
                         +--------------+-------------+
                                        |
                         +--------------v-------------+
                         |  Calibrated Affect Output  |
                         +----------------------------+
```

**Key design decision**: Two parallel preprocessing routes because classical DSP filters (Wiener, Spectral Subtraction) help ASR transcription but destroy the prosodic cues that SER relies on. See [Experiment 6](docs/experiment-6-emotions.md) for the evidence.

---

## 🚀 Quick Start

### Install (Ubuntu / WSL2 / macOS)
```bash
# Clone and setup
git clone https://github.com/Gustor1/projet-ml-s3.git
cd projet-ml-s3
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the full multimodal pipeline on an audio file
python main.py --audio data/emotion_samples/03-01-05-02-01-01-01.wav

# Override preprocessing method
python main.py --audio recording.wav --method spectral_subtraction

# Save results as JSON
python main.py --audio recording.wav --output results/output.json
```

### Install (Windows)
```powershell
git clone https://github.com/Gustor1/projet-ml-s3.git
cd projet-ml-s3

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

### 🐳 Docker (Offline Execution)

```bash
# Build image (pre-downloads all 3 HuggingFace models: ~1.2GB)
docker build -t projet-ml-s3 .

# Run pipeline inside container (no internet needed)
docker run --rm projet-ml-s3 python main.py --audio data/emotion_samples/03-01-05-02-01-01-01.wav
```

---

## 🧪 Testing & CI

```bash
# Run unit tests
pytest tests/ -v

# Run linter
flake8 . --max-line-length=120
```

GitHub Actions automatically runs lint + compile + test on every push to `main` and `feature/**` branches.

---

## 🔬 Experiments

| # | Experiment | Script | Results |
|---|---|---|---|
| 1 | Baseline ASR (Clean Audio) | `experiments/baseline_wer.py` | `results/baseline.csv` |
| 2 | White Gaussian Noise + FFT Debug | `experiments/compare_preprocessing.py` | `results/preprocessing_comparison.csv` |
| 3 | Pink 1/f Noise Evaluation | `experiments/compare_preprocessing.py` | `results/pink_noise_comparison.csv` |
| 4 | Real Urban Noise (DEMAND DB) | `experiments/compare_preprocessing.py` | `results/urban_noise_comparison.csv` |
| 5 | Babble Noise + Hallucination Analysis | `experiments/compare_preprocessing.py` | `results/babble_noise_comparison.csv` |
| 6 | Speech Emotion Recognition (SER) + Sarcasm | `experiments/evaluate_emotion_robustness.py` | `results/emotion_robustness.csv` |
| — | Cross-Modal ASR→NLP Ablation | `asr/cross_modal_ablation.py` | `results/cross_modal_ablation_summary.csv` |
| — | Joint Pipeline Profiling | `optimization/profiler.py` | `results/profiling_summary.csv` |

---

## 🖥️ Interactive Demo (Streamlit)

```bash
streamlit run demo/app.py
```

The dashboard provides:
- Real-time microphone recording + ASR transcription
- Live spectrogram with pitch ($F_0$) overlay
- Preprocessing mode selector (None / Wiener / Spectral Subtraction)
- SER emotion classification + sarcasm detection alert

---

## 📁 Repository Structure

```
projet-ml-s3/
├── main.py                 # Pipeline entry point (ASR + SER + Sarcasm)
├── configs/config.yaml     # Global configuration (models, preprocessing, pitch, etc.)
├── Dockerfile              # Production container with model caching
├── requirements.txt        # Python dependencies
├── asr/                    # ASR model wrappers (Whisper, Wav2Vec2)
│   ├── whisper_wrapper.py
│   ├── wav2vec_wrapper.py
│   ├── evaluator.py
│   ├── benchmark.py
│   ├── ablation_study.py
│   └── cross_modal_ablation.py
├── preprocessing/          # DSP algorithms (Wiener, Spectral Subtraction, VAD)
├── experiments/            # Scientific experiment scripts
├── optimization/           # Profiling + quantization
├── demo/app.py             # Streamlit interactive dashboard
├── scripts/                # Data download, augmentation, visualization
├── tests/                  # Unit tests (pytest)
├── docs/                   # Technical reports, experiment logs, journal
│   ├── final_report_data_engineer.md
│   ├── insights.md
│   ├── tradeoffs.md
│   ├── experiment-1-baseline.md
│   ├── experiment-2-final-comparison.md
│   ├── experiment-3-pink-noise.md
│   ├── experiment-4-urban-noise.md
│   ├── experiment-5-babble-noise.md
│   ├── experiment-6-emotions.md
│   ├── role3-asr-integration-report.md
│   ├── role5-profiling-report.md
│   └── journal/
├── results/                # CSV experiment outputs
├── visuals/                # Generated charts and figures
└── notebooks/              # Interactive Jupyter demos
```

---

## 👥 Team Roles

| Role | Member | Focus |
|------|--------|-------|
| 1. Pipeline Architect & DevOps | Elio | `main.py`, `config.yaml`, Docker, CI/CD |
| 2. Audio Preprocessing Engineer | — | Denoise APIs, VAD, parallel stream routing |
| 3. ASR Integration & Evaluation | Bilel | Whisper/Wav2Vec2 wrappers, WER/CER benchmarks |
| 4. Experimentation & Data | Eliott | 6 experiments, SER + sarcasm pipeline, data augmentation |
| 5. Optimization & Performance | Bilel & Elio | Model quantization, profiling, ONNX |
| 6. Demo & Video Production | — | Streamlit app, presentation video |

---

## 📚 Key References

1. Radford et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision.* ICML.
2. Gong et al. (2023). *Whisper-AT: Noise-Robust ASR as General Audio Tagger.* Interspeech.
3. Baevski et al. (2020). *wav2vec 2.0: Self-Supervised Speech Representations.* NeurIPS.
4. Boll (1979). *Suppression of acoustic noise using spectral subtraction.* IEEE TASLP.
5. Evans et al. (2005). *On the Fundamental Limitations of Spectral Subtraction.* EUSIPCO.
6. Tsao et al. (2019). *The impact of speech enhancement on SER.* IEEE SPL.
7. Schröter et al. (2022). *DeepFilterNet.* Interspeech.

See [`docs/final_report_data_engineer.md`](docs/final_report_data_engineer.md) for the full bibliography (14 references).

---

## 📚 Documentation

- [Pipeline Architecture Report](docs/pipeline-architecture-report.md) — Design decisions, trade-offs, limitations
- [Experiment Reports](docs/) — 6 experiments across 4 noise types
- [Engineering Insights](docs/insights.md) — 12 curated findings with academic references
- [Development Journal](docs/journal/) — Iterative progress logs
- [Submission Details](submission.txt) — Team contributions & deliverables

---

## 📄 License

Academic project — SHU S3 Machine Learning Course, 2026.
