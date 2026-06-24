# 🎙️ Local Audio Preprocessing for Better ASR Performance

> **Topic 3** — Research-grade evaluation of classical DSP speech enhancement for edge ASR and multimodal affective computing.  
> **GitHub Repository:** [https://github.com/Gustor1/projet-ml-s3](https://github.com/Gustor1/projet-ml-s3)  
> **Group:** 6 students | SHU S3 Machine Learning Project

---

## 📖 Overview

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

## 🏗️ Architecture

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

## 🛠️ Setup

### Requirements
- Python 3.9+
- GPU optional (CPU inference supported, RTF ≈ 0.4x on standard laptop)

### Install (Ubuntu / WSL2 / macOS)
```bash
git clone https://github.com/Gustor1/projet-ml-s3.git
cd projet-ml-s3

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Install (Windows)
```powershell
git clone https://github.com/Gustor1/projet-ml-s3.git
cd projet-ml-s3

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

### Docker
```bash
docker build -t asr-preprocessing .
docker run --rm asr-preprocessing python experiments/baseline_wer.py
```

---

## 🚀 Running Experiments

```bash
# 1. Download data (LibriSpeech + RAVDESS)
python scripts/download_real_librispeech.py
python scripts/download_emotion_samples.py

# 2. Generate noise augmentations
python scripts/augment_audio.py          # White noise
python scripts/augment_pink_noise.py     # Pink noise
python scripts/augment_urban_noise.py    # Urban noise
python scripts/augment_babble_noise.py   # Babble noise

# 3. Run experiments
python experiments/baseline_wer.py
python experiments/compare_preprocessing.py
python experiments/evaluate_emotion_robustness.py
python experiments/sarcasm_detector.py

# 4. Generate visualizations
python scripts/generate_all_visuals.py
```

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
├── asr/                        # ASR wrappers (Whisper, Wav2Vec2) + evaluators
│   ├── whisper_wrapper.py
│   ├── wav2vec_wrapper.py
│   ├── evaluator.py
│   ├── benchmark.py
│   ├── ablation_study.py
│   └── cross_modal_ablation.py
├── preprocessing/              # DSP algorithms (Wiener, Spectral Subtraction, VAD)
├── experiments/                # Scientific experiment scripts
├── optimization/               # Profiling + quantization
├── demo/                       # Streamlit dashboard
├── scripts/                    # Data pipeline (download, augment, visualize)
├── results/                    # CSV metrics (WER, CER, SER accuracy, PPL, latency)
├── visuals/                    # PNG charts and spectrograms
├── docs/                       # Research reports, experiment logs, insights
│   ├── final_report_data_engineer.md   # Main research report (Role 4)
│   ├── insights.md                     # 12 curated scientific insights
│   ├── tradeoffs.md                    # Engineering trade-offs analysis
│   ├── experiment-1-baseline.md
│   ├── experiment-2-final-comparison.md
│   ├── experiment-3-pink-noise.md
│   ├── experiment-4-urban-noise.md
│   ├── experiment-5-babble-noise.md
│   ├── experiment-6-emotions.md
│   ├── role3-asr-integration-report.md
│   ├── role5-profiling-report.md
│   └── journal/                        # Development journal (day-by-day)
├── configs/                    # YAML configuration
├── data/                       # Audio datasets (gitignored)
├── main.py
├── requirements.txt
├── Dockerfile
└── submission.txt
```

---

## 👥 Team

| Member | Role | Contribution |
|---|---|---|
| **Eliott** | Data & Experimentation Engineer (Role 4) | 17% |
| **Bilel** | ASR Integration & Evaluation (Role 3) + Profiling (Role 5) | 25% |
| TBD | Pipeline Architect & DevOps (Role 1) | ~17% |
| TBD | Audio Preprocessing Engineer (Role 2) | ~17% |
| TBD | Optimization & Real-Time (Role 5) | ~8% |
| TBD | Demo & Video Production (Role 6) | ~16% |

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

## 📄 License

Academic project — SHU S3 Machine Learning Course, 2026.
