# 🎙️ Local Audio Preprocessing for Better ASR Performance

A multimodal pipeline that combines **Audio Preprocessing**, **ASR (Whisper)**, **Speech Emotion Recognition (Wav2Vec2)**, **Text Sentiment Analysis (DistilBERT)**, and **Sarcasm Detection** to analyze speech from both verbal and non-verbal perspectives.

> **Topic 3** — Group of 6 students | Shanghai University × UTBM

## 🚀 Quick Start

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

## 🏗️ Pipeline Architecture

```
Audio File (.wav)
    │
    ├─── Denoised Stream ──► Wiener / SpecSub ──► Whisper ASR ──► Text
    │                                                                │
    │                                                  DistilBERT Sentiment
    │                                                                │
    ├─── Normalized Stream ──► Trim + Normalize ──► Wav2Vec2 SER    │
    │                                                    │           │
    │                                                    ▼           ▼
    │                                             Vocal Emotion  Text Sentiment
    │                                                    │           │
    └──────────────────────────────────────► Sarcasm Detector ◄──────┘
                                                    │
                                              Structured Results (JSON)
```

**Key design decision**: Two parallel preprocessing routes because classical DSP filters (Wiener, Spectral Subtraction) help ASR transcription but destroy the prosodic cues that SER relies on. See [Experiment 6](docs/experiment-6-emotions.md) for the evidence.

## 🐳 Docker (Offline Execution)

```bash
# Build image (pre-downloads all 3 HuggingFace models: ~1.2GB)
docker build -t projet-ml-s3 .

# Run pipeline inside container (no internet needed)
docker run projet-ml-s3 python main.py --audio data/emotion_samples/03-01-05-02-01-01-01.wav
```

## 🧪 Testing & CI

```bash
# Run unit tests
pytest tests/ -v

# Run linter
flake8 . --max-line-length=120
```

GitHub Actions automatically runs lint + compile + test on every push to `main` and `feature/**` branches.

## 📁 Project Structure

```
projet-ml-s3/
├── main.py                 # Pipeline entry point (ASR + SER + Sarcasm)
├── configs/config.yaml     # Global configuration (models, preprocessing, pitch, etc.)
├── Dockerfile              # Production container with model caching
├── requirements.txt        # Python dependencies
├── asr/                    # ASR model wrappers (Whisper, Wav2Vec2)
├── preprocessing/          # Audio preprocessing modules (denoise, VAD)
├── experiments/            # Scientific experiments & evaluation scripts
├── demo/app.py             # Streamlit interactive dashboard
├── optimization/           # Model quantization & profiling
├── scripts/                # Data download, augmentation, visualization
├── tests/                  # Unit tests (pytest)
├── docs/                   # Technical reports, experiment logs, journal
├── results/                # CSV experiment outputs
├── visuals/                # Generated charts and figures
└── notebooks/              # Interactive Jupyter demos
```

## 👥 Team Roles

| Role | Member | Focus |
|------|--------|-------|
| 1. Pipeline Architect & DevOps | Elio | `main.py`, `config.yaml`, Docker, CI/CD |
| 2. Audio Preprocessing Engineer | — | Denoise APIs, VAD, parallel stream routing |
| 3. ASR Integration & Evaluation | Bilel | Whisper/Wav2Vec2 wrappers, WER/CER benchmarks |
| 4. Experimentation & Data | Eliott | 6 experiments, SER + sarcasm pipeline, data augmentation |
| 5. Optimization & Performance | — | Model quantization, profiling, ONNX |
| 6. Demo & Video Production | — | Streamlit app, presentation video |

## 📊 Key Findings

- **Preprocessing is context-dependent**: Wiener filter helps ASR on white noise at 5dB (-2.75% WER) but *degrades* performance on realistic noise types (pink +11.1%, urban +5.0%, babble +8.1%)
- **Classical DSP destroys SER**: Wiener filtering drops emotion recognition accuracy from 39% to 18% by erasing prosodic cues
- **Dual routing is the solution**: Feed denoised audio to ASR and normalized audio to SER
- **Babble noise triggers hallucinations**: At 5dB SNR, 3.3% of Whisper inferences produce fabricated text (WER > 100%)

See [docs/insights.md](docs/insights.md) for the full analysis (12 insights with citations).

## 📚 Documentation

- [Pipeline Architecture Report](docs/pipeline-architecture-report.md) — Design decisions, trade-offs, limitations
- [Experiment Reports](docs/) — 6 experiments across 4 noise types
- [Engineering Insights](docs/insights.md) — 12 curated findings with academic references
- [Development Journal](docs/journal/) — Iterative progress logs
- [Submission Details](submission.txt) — Team contributions & deliverables
