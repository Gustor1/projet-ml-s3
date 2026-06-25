# 🎬 Video Presentation Script — Elio (Role 1: Pipeline Architect & DevOps)

This script contains the precise spoken lines (in English) and visual instructions for Elio's part in the final presentation video (≥ 10 minutes).

---

## 📌 Pipeline Architecture & DevOps (Role 1)
**Estimated Duration:** ~3:00 minutes

### 🎥 Visuals on Screen
* **0:00 - 0:30:** Show the overall system architecture diagram (from [pipeline-architecture-report.md](file:///c:/Users/fabre/.gemini/antigravity-ide/scratch/projet-ml-s3/docs/pipeline-architecture-report.md)).
* **0:30 - 1:15:** Scroll through [main.py](file:///c:/Users/fabre/.gemini/antigravity-ide/scratch/projet-ml-s3/main.py) highlighting the imports and the `run_pipeline` function. Show the dual-route routing logic.
* **1:15 - 1:45:** Open and highlight sections of [configs/config.yaml](file:///c:/Users/fabre/.gemini/antigravity-ide/scratch/projet-ml-s3/configs/config.yaml).
* **1:45 - 2:30:** Display the [Dockerfile](file:///c:/Users/fabre/.gemini/antigravity-ide/scratch/projet-ml-s3/Dockerfile) and highlight the model caching step (`scripts/cache_models.py`).
* **2:30 - 3:00:** Open [.github/workflows/ci.yml](file:///c:/Users/fabre/.gemini/antigravity-ide/scratch/projet-ml-s3/.github/workflows/ci.yml) and show the pytest results on the GitHub Action interface or local terminal.

### 🗣️ Spoken Script (English)

> "Hi everyone, I am Elio, and for this project, I was responsible for **Role 1: Pipeline Architect and DevOps**."
>
> *(Point to the Architecture Diagram)*
> "My main challenge was to integrate the deliverables from all team members into a single, cohesive, and reproducible execution flow. 
> 
> To achieve this, I built the entry point in `main.py` using a **dual-route preprocessing architecture**. 
> Our early experiments showed a critical trade-off: while ASR models like Whisper benefit from denoising under severe noise, classical DSP filters like the Wiener filter heavily distort the prosodic, non-verbal cues that Speech Emotion Recognition models rely on.
> 
> Therefore, the pipeline splits the audio stream: 
> Route 1 applies denoising before sending the audio to Whisper ASR. 
> Route 2 skips the frequency filters, applying only silence trimming and peak normalization before routing the audio to Wav2Vec2 for emotion classification. 
> We then fuse the outputs downstream to perform sarcasm detection."
>
> *(Point to config.yaml)*
> "To ensure our scientific benchmarks are completely reproducible, I designed a project-wide configuration specification in `configs/config.yaml`. 
> With over 95 parameters, this single file serves as the absolute source of truth—controlling everything from model weight paths to Pitch estimation YIN bounds and sarcasm detection thresholds."
>
> *(Point to Dockerfile & cache_models.py)*
> "From a DevOps perspective, I built a production-ready `Dockerfile`. To enable fully offline, edge-capable execution, I wrote a custom caching script that pre-downloads and packages our three Hugging Face models—Whisper, Wav2Vec2, and DistilBERT—directly during the Docker image build.
> 
> Finally, to enforce code quality across our collaborative development, I set up a 3-job CI/CD pipeline using GitHub Actions. It runs Python compilation checks, strict Flake8 linting, and executes our suite of 16 unit tests, ensuring that no breaking commits reach our main branch."
