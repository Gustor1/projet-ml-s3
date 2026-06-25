# 📅 2026-06-21 — Role 1: Pipeline Integration & Docker Offline Deployment

## 🎯 Objective
Finalize the project's core infrastructure: make `main.py` actually run the full multimodal pipeline instead of being a placeholder, complete the `config.yaml` with all model paths and tuning parameters, and make the Docker container work 100% offline.

## 🧠 What Worked

### Config-driven architecture
The decision to centralize *all* model names, preprocessing parameters, pitch bounds, and sarcasm thresholds into a single `config.yaml` paid off. It means anyone on the team can tweak the pipeline behavior without touching Python code. For example, switching from Wiener to spectral subtraction is just one line:

```yaml
preprocessing:
  method: "spectral_subtraction"  # was "wiener"
```

This feels obvious in retrospect, but the original config only had `enable_denoise: true/false` — it didn't specify *which* method, *which* model, or *what* thresholds. That made it impossible for `main.py` to actually do anything without hardcoded defaults scattered everywhere.

### Dual-route preprocessing
The biggest design decision was implementing **two parallel audio streams** in the pipeline:
- **Denoised stream → ASR**: Wiener/SpecSub cleans the audio for Whisper transcription
- **Normalized stream → SER**: Trim silence + peak normalize for Wav2Vec2 emotion classification

This came directly from Eliott's findings in Experiment 6 (Insight 11): Wiener filtering destroys prosodic cues that SER needs, so we can't just denoise everything and feed both models the same audio. The parallel routing is the engineering fix for a fundamental scientific trade-off.

### HuggingFace model caching
Getting the Docker container to work offline was harder than expected. The naive approach (just `pip install transformers` and hope for the best) fails because HuggingFace downloads models lazily at first inference. The solution: a dedicated `scripts/cache_models.py` that creates a `transformers.pipeline` for each model during Docker build. This forces the download into a cached layer. If any model fails to download, the script exits with code 1 and the Docker build fails — no silent offline failures.

## ❌ What Didn't Work / Pain Points

### First attempt at main.py
My first version of `main.py` tried to import from `preprocessing/denoise.py` and `preprocessing/vad.py` directly. Problem: those modules don't exist yet (that's Role 2's job). I wasted time trying to make it depend on files that haven't been written. The fix was to embed the preprocessing functions directly into `main.py` — yes, it's code duplication with `demo/app.py`, but it means the pipeline runs standalone without waiting for Role 2 to deliver their modules. We can refactor later to import from `preprocessing/` when those files land.

### GitHub Actions workflow scope
Pushing the updated `ci.yml` required a GitHub token with the `workflow` scope. My first token didn't have it, so the push was rejected for the CI file specifically while everything else went through fine. This is a GitHub security feature I didn't know about — Personal Access Tokens need *explicit* workflow permissions to modify anything under `.github/workflows/`. Had to regenerate the token.

### Spectral subtraction edge cases
While writing the preprocessing functions in `main.py`, I noticed that spectral subtraction can produce slightly-above-1.0 values after peak normalization due to floating point precision (`1.0000001`). Added an epsilon guard in the tests to avoid flaky assertions. Small detail but the kind of thing that would break CI at 2 AM.

## 🔍 Design Decisions & Trade-offs

| Decision | Why | Alternative Considered |
|----------|-----|----------------------|
| Embed preprocessing in `main.py` | Role 2 modules don't exist yet | Import from `preprocessing/` — blocked on other team |
| Use `transformers.pipeline` directly | Simpler, consistent with `demo/app.py` | Use our own `asr/whisper_wrapper.py` — adds import complexity |
| Exit with structured JSON output | Machine-readable for downstream tooling | Pretty-print only — less useful for automation |
| Config defaults as fallbacks | Pipeline works even with minimal config | Crash on missing keys — more rigid but explicit |

## 📊 Current Pipeline Architecture

```
Audio File (.wav)
    │
    ├─── [Denoised Stream] ──► Wiener / SpecSub ──► Whisper ASR ──► Text
    │                                                                  │
    │                                                    DistilBERT Sentiment
    │                                                                  │
    ├─── [Normalized Stream] ──► Trim + Normalize ──► Wav2Vec2 SER    │
    │                                                      │           │
    │                                                      ▼           ▼
    │                                               Vocal Emotion  Text Sentiment
    │                                                      │           │
    └──────────────────────────────────────────────► Sarcasm Detector ◄─┘
                                                           │
                                                    Structured Results (JSON)
```

## ✅ Verification
- All Python files compile cleanly (`py_compile`)
- `main.py --help` displays correct CLI usage
- Config YAML parses without errors
- 16 unit tests covering preprocessing functions and sarcasm detection logic
- 7 commits pushed to `feature/pipeline-architect`

## 🔮 Next Steps
- When Role 2 delivers `preprocessing/denoise.py` and `preprocessing/vad.py`, refactor `main.py` to import from there instead of duplicating
- When Role 5 delivers quantized models, update `config.yaml` model paths and add ONNX runtime option
- Create a Pull Request to merge `feature/pipeline-architect` into `main`
