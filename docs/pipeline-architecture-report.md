# 🏗️ Pipeline Architecture & DevOps — Technical Report

## Overview

This document describes the engineering work done for **Role 1: Pipeline Architect & DevOps** on the multimodal Audio Preprocessing + ASR + SER + Sarcasm Detection pipeline.

The goal was to take the scattered modules (Whisper wrapper, emotion evaluator, sarcasm detector, Streamlit demo) and tie them together into a single, reproducible, config-driven pipeline that works both locally and inside a Docker container without internet.

---

## 1. The Config Problem (and Why It Matters)

When I first looked at the project, `configs/config.yaml` had 26 lines. It told the pipeline to "enable denoising" but never said *which* method, never specified *which* models to load, and had no pitch estimation parameters even though the demo app uses YIN pitch tracking extensively.

This created a silent dependency problem: every script hardcoded its own defaults. `demo/app.py` uses `openai/whisper-tiny`, `experiments/sarcasm_detector.py` also uses `whisper-tiny` but with different options, and the config says `openai/whisper-small`. Which one is it? Nobody knows until something crashes.

### What I did
Extended `config.yaml` to 95+ lines covering:
- **All three model identifiers** (ASR, SER, NLP) — single source of truth
- **Preprocessing method selection** with per-method hyperparameters (Wiener window size, SpecSub alpha/beta)
- **VAD sensitivity** (0-3 scale matching WebRTC conventions)
- **SNR threshold** for conditional preprocessing activation
- **YIN pitch parameters** (fmin/fmax matching human vocal range 75-400Hz)
- **Multimodal fusion calibration** thresholds (pitch high/low boundaries)
- **Sarcasm detection toggle** — can be disabled for pure ASR workloads

### What I learned
The hardest part wasn't writing the YAML — it was *reading other people's code* to find every magic number that should be a config parameter. Eliott's `fuse_modalities()` function in `demo/app.py` uses `180.0` and `130.0` Hz as pitch thresholds, `0.25` as a boost factor. Those are now `fusion.pitch_high_threshold`, `fusion.pitch_low_threshold`, and `fusion.text_boost_factor` in the config. If someone wants to tune the sarcasm detector's sensitivity, they change YAML instead of hunting through 740 lines of Streamlit code.

---

## 2. The `main.py` Rewrite

### Before
```python
def main(config_path: str):
    config = load_config(config_path)
    print(f"[INFO] Loaded config for project: {config['project']['name']}")
    print("[INFO] Placeholder main() finished.")
```

### After
The pipeline now runs the full multimodal sequence in ~260 lines:

1. **Load audio** — reads any WAV file, converts stereo to mono
2. **Route 1 (Denoised)**: Apply Wiener or SpecSub → feed to Whisper ASR → get transcription → run DistilBERT sentiment
3. **Route 2 (Normalized)**: Trim silence + peak normalize → feed to Wav2Vec2 SER → get vocal emotion
4. **Cross-modal check**: Compare text sentiment vs. vocal emotion → sarcasm detection
5. **Output**: Structured JSON results + human-readable report to stdout

### Why dual routing?
This was the biggest design question. Eliott's Experiment 6 conclusively showed that Wiener filtering *destroys* SER accuracy (from 39.29% to 17.86% under white noise at 5dB). But Wiener *helps* ASR in the same conditions (-2.75% WER). So we can't feed both models the same preprocessed audio.

The solution: two separate preprocessing routes.
- **Denoised stream**: Classical DSP filters clean noise for Whisper. These filters damage pitch and prosody, but Whisper doesn't care — it only needs the text.
- **Normalized stream**: Trim silence margins + peak normalize for Wav2Vec2. No spectral manipulation, just amplitude standardization. This preserves the emotional cues that SER relies on.

This is documented in the project roadmap as "Parallel Stream Routing" and it's the core architectural insight of the pipeline.

### What didn't work
My first attempt imported from `preprocessing/denoise.py` — which doesn't exist yet (Role 2 hasn't delivered it). I burned 20 minutes trying to make the pipeline depend on modules that aren't written. Lesson learned: embed the functions directly and refactor later. Yes, it duplicates code from `demo/app.py`, but at least the pipeline runs today.

---

## 3. Docker & Offline Model Caching

### The problem
The Dockerfile was minimal: install pip packages, copy source, run `main.py`. But HuggingFace `transformers` downloads model weights *lazily* — the first time you call `pipeline("automatic-speech-recognition", model="openai/whisper-tiny")`, it fetches ~150MB from the Hub. In a Docker container without internet, this silently fails.

### The fix
`scripts/cache_models.py` forces the download during image build:

```python
MODELS_TO_CACHE = {
    "ASR": ("openai/whisper-tiny", "automatic-speech-recognition"),
    "SER": ("superb/wav2vec2-base-superb-er", "audio-classification"),
    "NLP": ("distilbert-base-uncased-finetuned-sst-2-english", "sentiment-analysis"),
}
```

The script creates a `transformers.pipeline` for each model (which triggers the download), then exits. If any model fails, it returns exit code 1 and the Docker build fails. No silent offline failures.

The Dockerfile also adds `libsndfile1` and `ffmpeg` as system dependencies — without these, `soundfile` and `librosa` crash at runtime with cryptic import errors.

### Image size trade-off
Pre-caching three models adds ~1.2GB to the Docker image. That's significant, but the alternative (downloading at runtime) is worse for reproducibility and deployment. The models are baked into a Docker layer, so rebuilds that don't touch `cache_models.py` skip the download entirely.

---

## 4. CI Pipeline Design

### Before
The CI workflow only ran `python -m compileall .` — it checked that Python files have valid syntax but nothing else. No linting, no tests.

### After
Three parallel jobs:

| Job | Tool | Purpose | Blocking? |
|-----|------|---------|-----------|
| `lint` | flake8 | Syntax errors (E9, F63, F7, F82) | ✅ Yes — critical errors block merge |
| | flake8 | Style warnings (line length, complexity) | ❌ No — advisory only (`--exit-zero`) |
| `compile` | py_compile | Python bytecode compilation | ✅ Yes |
| `test` | pytest | Unit tests in `tests/` | ✅ Yes — runs after lint + compile pass |

The linting is deliberately split into two passes. The first pass catches real bugs (undefined names, syntax errors) and blocks the PR. The second pass reports style issues (lines > 120 chars, McCabe complexity > 10) but doesn't fail the build. This avoids the "CI is red because of a trailing whitespace" frustration while still catching actual defects.

### 16 unit tests
I wrote tests for the things that *don't* need GPU or model downloads:
- Config YAML structure validation (file exists, all sections present, valid ranges)
- Preprocessing functions (output shape, normalization bounds, edge cases)
- Sarcasm detection logic (all mismatch combinations)
- Config loader (parsing works, returns dict)

Tests that would require downloading HuggingFace models (full pipeline integration tests) are out of scope for CI — they'd take 5+ minutes and need 2GB of downloads.

---

## 5. Honest Assessment: Limitations

1. **Code duplication**: Preprocessing functions exist in three places (`main.py`, `demo/app.py`, `experiments/evaluate_emotion_robustness.py`). This is technical debt that should be resolved when Role 2 delivers `preprocessing/denoise.py`.

2. **No VAD in the pipeline**: The config has `vad.enabled: true` and sensitivity parameters, but `main.py` doesn't actually implement VAD (it's listed as a Role 2 deliverable). The parameter is there for when the module arrives.

3. **Single-file inference only**: `main.py` processes one audio file at a time. Batch processing (directory of WAV files → CSV of results) would be a useful extension.

4. **No GPU memory management**: The pipeline loads all three models into memory simultaneously (~1.5GB VRAM). On edge devices with limited GPU, sequential loading/unloading would be more efficient, but adds latency.

5. **Test coverage is partial**: The 16 tests cover preprocessing logic and config validation, but there are no integration tests that actually run the full pipeline end-to-end. Those would require model downloads and a test audio file.

---

## 6. Reproducibility

```bash
# Run the pipeline
python main.py --audio data/emotion_samples/03-01-05-02-01-01-01.wav --config configs/config.yaml

# Override preprocessing method
python main.py --audio recording.wav --method spectral_subtraction

# Save results as JSON
python main.py --audio recording.wav --output results/pipeline_output.json

# Run unit tests
pytest tests/ -v

# Build Docker image (will cache all models)
docker build -t projet-ml-s3 .

# Run pipeline in Docker
docker run projet-ml-s3 python main.py --audio data/emotion_samples/03-01-05-02-01-01-01.wav
```
