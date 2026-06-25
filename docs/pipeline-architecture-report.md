# 🏗️ Pipeline Architecture & DevOps — Technical Report

## Overview

This document describes the engineering work done for **Role 1: Pipeline Architect & DevOps** on the multimodal Audio Preprocessing + ASR + SER + Sarcasm Detection pipeline. Beyond implementation, it explains **why** each design decision was made, what alternatives were considered, and how the choices are grounded in existing research.

---

## 1. Research Context & Hypothesis

### 1.1 Problem Statement
Modern voice interfaces require more than transcription accuracy. Detecting **vocal emotions** alongside text opens up applications in customer service analytics [1], mental health monitoring [2], and deception/sarcasm detection [3]. However, integrating multiple models (ASR, NLP, SER) into a single pipeline introduces engineering trade-offs that are rarely discussed in the literature, which tends to evaluate models in isolation.

**Research hypothesis**: Classical DSP preprocessing (Wiener filtering, spectral subtraction) improves ASR accuracy but degrades SER performance. A **dual-route architecture** — feeding denoised audio to ASR and normalized audio to SER — can preserve the benefits of both without the destructive interference documented by Tsao et al. (2019) [4].

### 1.2 Related Work

| System | Architecture | Limitation addressed by our work |
|--------|-------------|----------------------------------|
| Whisper (Radford et al., 2022) [5] | End-to-end encoder-decoder ASR | Does not include emotion recognition |
| Whisper-AT (Gong et al., 2023) [6] | Whisper + audio tagging | Shows Whisper encodes noise type — preprocessing removes this conditioning signal |
| SUPERB benchmark (Yang et al., 2021) [7] | Multi-task evaluation including SER | Evaluates models separately, not in a joint pipeline |
| EmotionPrompt (Li et al., 2023) [8] | LLM-based emotion detection from text | Misses non-verbal cues entirely |
| Multimodal sentiment (Poria et al., 2017) [9] | Text + audio + video fusion | Uses complex attention fusion; our heuristic is simpler but interpretable |

Our contribution is the **engineering integration** of these capabilities into a reproducible, config-driven pipeline with empirically-grounded preprocessing routing.

---

## 2. Model Selection & Justification

### 2.1 ASR: Why Whisper-tiny?

We evaluated three ASR options for the pipeline:

| Model | Parameters | WER (LibriSpeech test-clean) | Latency (CPU, 3s audio) | Multilingual | Source |
|-------|-----------|------------------------------|--------------------------|-------------|--------|
| **Whisper-tiny** | 39M | ~7.6% [5] | ~1.5-3s | ✅ 99 languages | OpenAI |
| Wav2Vec2-base | 95M | ~6.1% [10] | ~1-2s | ❌ English only | Meta |
| Faster-Whisper (CTranslate2) | 39M (quantized) | ~7.6% | ~0.5-1s | ✅ | Community |
| WhisperX (Bain et al., 2023) [11] | 39M+ | ~7.6% + alignment | ~2-4s | ✅ | Research |

**Decision**: Whisper-tiny was chosen for three reasons:
1. **Multilingual support out of the box** — the project specification requires a general-purpose pipeline, and Whisper-tiny handles 99 languages natively. Wav2Vec2-base is English-only without fine-tuning.
2. **Consistent API with HuggingFace** — all three models (ASR, NLP, SER) use the `transformers.pipeline()` interface, simplifying integration and caching. Faster-Whisper requires a separate CTranslate2 runtime.
3. **Baseline for ablation** — Eliott's cross-modal ablation study (Experiment 12 in `docs/insights.md`) compares tiny/base/small, showing that upgrading from tiny to small reduces sarcasm false positives from 10.71% to 0%. Whisper-tiny is the *worst case* — if the pipeline works with tiny, it works with any larger model.

**Why not Faster-Whisper?** Faster-Whisper (Bain et al., 2023) [11] uses CTranslate2 for 3-5x CPU speedup via INT8 quantization and optimized kernels. This is the **recommended production choice** (see `docs/optimization-report.md`). However, for this research project, we prioritize *consistency* over speed: using the same `transformers` framework across all three models enables fair profiling comparisons and uniform quantization experiments.

**Why not WhisperX?** WhisperX adds forced phoneme alignment using wav2vec2-based alignment models, providing word-level timestamps. Our pipeline doesn't require timestamps — it processes entire utterances. Adding WhisperX would double the model loading time (Whisper + alignment model) for a feature we don't use.

### 2.2 SER: Why `superb/wav2vec2-base-superb-er`?

| Model | Training Data | Emotions | Accuracy (IEMOCAP) | Source |
|-------|--------------|----------|--------------------|----|
| **wav2vec2-base-superb-er** | IEMOCAP [12] | 4 (neu/hap/sad/ang) | ~63.4% [7] | SUPERB benchmark |
| HuBERT-base-superb-er | IEMOCAP | 4 | ~64.9% [7] | SUPERB benchmark |
| Emotion2Vec (Ma et al., 2024) [13] | 40k hours | 9 | ~87% (weighted F1) | Recent SOTA |
| SpeechBrain ECAPA-TDNN | IEMOCAP | 4 | ~65% | SpeechBrain |

**Decision**: We chose `wav2vec2-base-superb-er` because:
1. **SUPERB benchmark standardization** — it's the reference model from the SUPERB benchmark [7], making our results directly comparable to published baselines.
2. **Raw waveform input** — unlike older SER models (OpenSMILE + SVM [14]), Wav2Vec2 processes raw audio at 16kHz without handcrafted features, making it compatible with our dual-route architecture.
3. **Known cross-corpus limitations** — the model was trained on IEMOCAP (spontaneous conversation) and we test on RAVDESS (acted speech). The expected accuracy drop from ~63% to ~35% is well-documented in cross-corpus SER literature [15] and is part of what we investigate.

**Why not Emotion2Vec?** Emotion2Vec (Ma et al., 2024) [13] achieves state-of-the-art results but was published after our project started, and its model weights require a separate `funasr` framework. Integration would break our uniform `transformers.pipeline()` architecture.

**Why not HuBERT?** HuBERT-base-superb-er achieves marginally better accuracy (+1.5%) but uses the same architecture and inference latency. We chose Wav2Vec2 because it has more community support and documentation.

### 2.3 NLP: Why DistilBERT-SST2?

| Model | Parameters | Accuracy (SST-2) | Latency (CPU) | Source |
|-------|-----------|-------------------|---------------|--------|
| **DistilBERT-SST2** | 66M | 91.3% [16] | ~10-30ms | HuggingFace |
| BERT-base-SST2 | 110M | 93.5% | ~30-60ms | Google |
| RoBERTa-large-SST2 | 355M | 96.4% | ~100-200ms | Meta |
| GPT-3.5 (zero-shot) | 175B | ~95% | ~500ms+ (API) | OpenAI |

**Decision**: DistilBERT-SST2 because:
1. **Speed is critical** — sentiment analysis runs on every transcription. At ~10-30ms per inference, DistilBERT adds negligible latency to the pipeline. RoBERTa-large would add 100ms+.
2. **Distillation preserves accuracy** — Sanh et al. (2019) [16] showed that DistilBERT retains 97% of BERT's accuracy while being 60% smaller and 2x faster. For binary sentiment (positive/negative), the 2% accuracy gap is irrelevant to sarcasm detection.
3. **INT8 quantization candidate** — DistilBERT's pure-encoder architecture with dense linear layers makes it the ideal target for dynamic quantization (see Role 5 report), achieving ~45-50% size reduction.
4. **Offline execution** — unlike GPT-3.5, DistilBERT runs fully locally without API calls, which is required for our Docker offline deployment.

---

## 3. The Dual-Route Architecture

### 3.1 Why Not a Single Preprocessing Pipeline?

The naive approach is to denoise the audio once and feed it to both ASR and SER. Our experiments demonstrate why this fails:

| Condition (5dB SNR, White Noise) | ASR (WER) | SER (Accuracy) |
|----------------------------------|-----------|----------------|
| No preprocessing | 27.47% | 39.29% |
| Wiener filter | **24.72%** ✅ (-2.75%) | **17.86%** ❌ (-21.43%) |
| Spectral subtraction | 42.11% ❌ | **53.57%** ✅ (+14.28%) |

**Key finding**: There is no single preprocessing method that benefits both tasks simultaneously. This confirms the "enhancement-distortion trade-off" described by Tsao et al. (2019) [4] — speech enhancement that improves intelligibility (ASR) destroys the fine acoustic cues (pitch micro-variations, jitter, shimmer) that SER relies on.

### 3.2 Architecture Design

```
Audio File (.wav)
    │
    ├─── [Route 1: Denoised] ──► Wiener / SpecSub ──► Whisper ASR ──► Text
    │         Rationale: Classical DSP removes                          │
    │         stationary noise. Whisper's attention              DistilBERT
    │         mechanism benefits from cleaner                    Sentiment
    │         spectral features [6].                                    │
    │                                                                   │
    ├─── [Route 2: Normalized] ──► Trim + Peak Norm ──► Wav2Vec2 SER   │
    │         Rationale: No spectral manipulation.                      │
    │         Preserves pitch (F0), jitter, shimmer,                    │
    │         and energy dynamics that encode                           │
    │         emotion [1].                                              │
    │                                                                   │
    └────────────────────────────────────────────► Sarcasm Detector ◄───┘
              Compares text sentiment vs. vocal emotion.
              Mismatch = potential sarcasm [3].
```

### 3.3 Literature Support for Dual Routing

The dual-route approach is supported by two key findings:

1. **Gong et al. (2023) [6]** showed that Whisper's intermediate representations encode background noise type. Classical DSP preprocessing removes this noise information, depriving Whisper of the acoustic context it uses for conditioned transcription. This explains why Wiener filtering provides only modest ASR gains (~2.75%) — Whisper already handles noise internally.

2. **Busso et al. (2005) [1]** demonstrated that SER relies on F0 contour, energy envelope, and spectral tilt — all of which are distorted by Wiener filtering's frequency-domain smoothing. Our empirical observation (SER accuracy drops from 39.29% to 17.86% with Wiener) confirms this mechanism.

---

## 4. Configuration Management

### 4.1 Why a Centralized YAML Config?

The original project had hardcoded defaults scattered across 8+ files. Each script used slightly different model names, thresholds, and preprocessing parameters. This violates the **single source of truth** principle and makes experiments non-reproducible — you can't tell which configuration produced which results.

Our `config.yaml` (95+ parameters) solves this by centralizing:
- **Model identifiers**: Changing `asr.model_name: "openai/whisper-base"` automatically updates the entire pipeline
- **Preprocessing hyperparameters**: Wiener window size, SpecSub oversubtraction factor (α), spectral floor (β)
- **Pitch estimation**: YIN algorithm bounds matching the human vocal range (fmin=75Hz male fundamental, fmax=400Hz female fundamental) [17]
- **Sarcasm thresholds**: Multimodal fusion calibration parameters derived from Eliott's RAVDESS experiments

### 4.2 Design Alternatives Considered

| Approach | Advantage | Disadvantage | Our choice |
|----------|-----------|-------------|------------|
| **YAML config file** | Human-readable, Git-trackable, hierarchical | No type validation | ✅ Selected |
| Python dataclasses | Type-safe, IDE support | Not editable by non-programmers | Considered |
| argparse only | No config file needed | Doesn't scale to 95+ parameters | Rejected |
| Hydra (Facebook) | Powerful, composable | Heavy dependency for a research project | Rejected |
| Environment variables | Container-friendly | Flat namespace, no nesting | Rejected |

---

## 5. Docker & Offline Execution

### 5.1 The Lazy Loading Problem

HuggingFace `transformers` downloads model weights on first use via `huggingface_hub`. In a Docker container with `--network=none`, this silently fails with a cryptic `OSError: Can't load tokenizer`. Our caching script (`scripts/cache_models.py`) forces the download during `docker build` by instantiating each pipeline:

```python
MODELS_TO_CACHE = {
    "ASR": ("openai/whisper-tiny", "automatic-speech-recognition"),
    "SER": ("superb/wav2vec2-base-superb-er", "audio-classification"),
    "NLP": ("distilbert-base-uncased-finetuned-sst-2-english", "sentiment-analysis"),
}
```

### 5.2 Image Size Analysis

| Component | Size | Justification |
|-----------|------|---------------|
| Python 3.10-slim base | ~120 MB | Minimal Debian without dev tools |
| System deps (ffmpeg, libsndfile1) | ~50 MB | Required by librosa and soundfile for audio I/O |
| PyTorch CPU | ~350 MB | Core ML framework, CPU-only to avoid CUDA bloat |
| Whisper-tiny weights | ~150 MB | ASR model |
| Wav2Vec2-SER weights | ~360 MB | SER model |
| DistilBERT-SST2 weights | ~255 MB | NLP model |
| Python packages + source | ~100 MB | transformers, librosa, scipy, etc. |
| **Total** | **~1.4 GB** | |

**Alternative**: Using `openai/whisper-large-v3` (1.55B params) would increase the image to ~6GB. The tiny model keeps the image under 1.5GB, enabling deployment on resource-constrained environments.

---

## 6. CI/CD Pipeline

### 6.1 Three-Job Architecture

| Job | Tool | Purpose | Blocking? |
|-----|------|---------|-----------| 
| `lint` | flake8 | Critical syntax errors (E9, F63, F7, F82) | ✅ Blocks PR |
| | flake8 | Style warnings (line length, McCabe complexity) | ❌ Advisory (`--exit-zero`) |
| `compile` | py_compile | Python bytecode compilation | ✅ Blocks PR |
| `test` | pytest | Unit tests (27 tests total across 3 test files) | ✅ Blocks PR |

The deliberate split between blocking and non-blocking lint passes avoids the common problem of CI fatigue — where developers ignore CI because it's always red due to trivial style violations [18].

---

## 7. Limitations & Future Work

1. **Code duplication**: Preprocessing functions exist in `main.py`, `demo/app.py`, and `experiments/evaluate_emotion_robustness.py`. A shared `preprocessing/` module would eliminate this, pending Role 2 delivery.

2. **No end-to-end integration tests**: Our 27 unit tests cover individual functions but not the full pipeline (which requires model downloads and ~2GB of disk space). A CI environment with cached models would enable this.

3. **Sequential model loading**: All three models load simultaneously (~765MB RAM). On edge devices, loading models one-at-a-time and releasing memory between stages would reduce peak RAM to ~360MB (largest single model), at the cost of increased latency.

4. **Sarcasm detection is heuristic-based**: The cross-modal mismatch check (positive text + angry voice = sarcasm) is a simple rule. A learned classifier trained on multimodal sarcasm datasets (e.g., MUStARD [3]) would be more robust, but requires labeled data we don't have.

5. **VAD not implemented**: The config includes VAD parameters, but the module depends on Role 2's delivery of `preprocessing/vad.py`.

---

## References

[1] C. Busso et al., "Analysis of emotion recognition using acoustic features in a multidimensional space," *Proc. Interspeech*, 2005.
[2] N. Cummins et al., "A review of depression and suicide risk assessment using speech analysis," *Speech Communication*, vol. 71, pp. 10–49, 2015.
[3] S. Castro et al., "Towards Multimodal Sarcasm Detection (An Obviously Perfect Paper)," *Proc. ACL*, pp. 4619–4629, 2019. (MUStARD dataset)
[4] Y. Tsao et al., "The impact of speech enhancement on speech emotion recognition," *IEEE Signal Process. Lett.*, vol. 26, no. 12, pp. 1803–1807, 2019.
[5] A. Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision," *Proc. ICML*, 2022.
[6] Y. Gong et al., "Whisper-AT: Noise-Robust Automatic Speech Recognizers are Also Strong General Audio Event Taggers," *Proc. Interspeech*, pp. 2798–2802, 2023.
[7] S. Yang et al., "SUPERB: Speech processing Universal PERformance Benchmark," *Proc. Interspeech*, pp. 1194–1198, 2021.
[8] C. Li et al., "EmotionPrompt: Leveraging Psychology for Large Language Models Enhancement via Emotional Stimulus," *arXiv:2307.11760*, 2023.
[9] S. Poria et al., "A review of affective computing: From unimodal analysis to multimodal fusion," *Information Fusion*, vol. 37, pp. 98–125, 2017.
[10] A. Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," *Proc. NeurIPS*, 2020.
[11] M. Bain et al., "WhisperX: Time-Accurate Speech Transcription of Long-Form Audio," *Proc. Interspeech*, 2023.
[12] C. Busso et al., "IEMOCAP: Interactive emotional dyadic motion capture database," *Lang. Resour. Eval.*, vol. 42, no. 4, pp. 335–359, 2008.
[13] Z. Ma et al., "emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation," *arXiv:2312.15185*, 2024.
[14] B. Schuller et al., "The INTERSPEECH 2009 Emotion Challenge," *Proc. Interspeech*, 2009.
[15] S. Latif et al., "Cross-corpus speech emotion recognition: An overview and directions," *IEEE Trans. Affect. Comput.*, 2021.
[16] V. Sanh et al., "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter," *arXiv:1910.01108*, 2019.
[17] D. Talkin, "A Robust Algorithm for Pitch Tracking (RAPT)," in *Speech Coding and Synthesis*, Elsevier, 1995, pp. 495–518.
[18] M. Hilton et al., "Usage, costs, and benefits of continuous integration in open-source projects," *Proc. ASE*, pp. 426–437, 2016.
