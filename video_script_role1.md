# 🎬 Video Script — Role 1: Pipeline Architect & DevOps

> **Target duration:** ~2 min (of a total ≥10 min group video)  
> **Language:** English  
> **Tone:** Engineering-focused, practical and data-driven  
> **Format conventions:**  
> - `[VISUAL: ...]` → screen / slide to show at that moment  
> - `[PAUSE]` → beat/pause for emphasis  
> - `⏱ ~X s` → estimated speaking time for the segment  

---

## 🎬 SEGMENT 1 — Introduction & Dual-Route Architecture
> ⏱ ~40 s

---

`[VISUAL: visuals/slide_r1_architecture.png — System Architecture Diagram]`

---

> This section covers **Pipeline Architecture and DevOps**.
>
> The primary challenge of this project was integrating independent deliverables—ASR, Speech Emotion Recognition, and Sentiment Analysis—into a single, reproducible execution flow.
>
> [PAUSE]
>
> To achieve this, we built a unified entry point utilizing a **dual-route preprocessing architecture**. 
> 
> Experimental results revealed a fundamental trade-off: ASR models like Whisper require denoising to handle severe noise, whereas classical DSP filters destroy the prosodic, non-verbal cues that emotion models depend on. 
>
> Therefore, the pipeline actively splits the audio stream. Route 1 applies frequency filtering before routing to Whisper. Route 2 bypasses these filters, applying only silence trimming and peak normalization before routing the raw audio to Wav2Vec2. The streams are then fused downstream for sarcasm detection.

---

## 🎬 SEGMENT 2 — Configuration as a Single Source of Truth
> ⏱ ~30 s

---

`[VISUAL: configs/config.yaml — Screen recording scrolling through the YAML configuration]`

---

> To guarantee scientific reproducibility across our benchmarks, we centralized all system parameters.
>
> We engineered a project-wide configuration specification utilizing YAML. With over 95 parameters, this single file acts as the absolute source of truth. 
>
> [PAUSE]
>
> It controls everything from model weight paths to signal processing thresholds, pitch estimation boundaries, and multimodal fusion weights, allowing researchers to tweak experiments without altering any code.

---

## 🎬 SEGMENT 3 — DevOps: Offline Execution & CI/CD
> ⏱ ~40 s

---

`[VISUAL: Dockerfile and GitHub Actions Interface — Split screen or transitioning slides]`

---

> From a DevOps perspective, the pipeline is fully containerized. 
>
> We developed a production-ready Dockerfile coupled with a custom caching script. This pre-downloads all three Hugging Face models—amounting to over 1.2 gigabytes—directly during the image build. This guarantees that the final container can execute fully offline, satisfying strict edge-deployment requirements.
>
> [PAUSE]
>
> Finally, to enforce rigorous code quality during collaborative development, we implemented a 3-job CI/CD pipeline via GitHub Actions. It enforces strict Python linting, executes a suite of 16 unit tests, and validates configuration schemas, ensuring zero breaking changes reach the main branch.

---

## 🎬 SEGMENT 4 — Conclusion
> ⏱ ~10 s

---

`[VISUAL: visuals/slide_r1_conclusion.png — 3 key takeaways: Dual-Routing, Reproducibility, CI/CD]`

---

> By establishing a robust architecture, centralized configurations, and strict DevOps practices, we provided a highly stable foundation upon which the entire multimodal project was successfully built.

---

# 📸 Visual Checklist — Role 1

| # | Segment | File / Action | Status | Description |
|---|---|---|:---:|---|
| 1 | Seg 1 — architecture | `visuals/slide_r1_architecture.png` | ⚠️ **To generate** | Diagram of the dual-route preprocessing |
| 2 | Seg 2 — configuration | `configs/config.yaml` | ✅ **Ready** | Screen recording scrolling through parameters |
| 3 | Seg 3 — devops | `Dockerfile` & GitHub UI | ✅ **Ready** | Show Dockerfile caching & CI green checks |
| 4 | Seg 4 — conclusion | `visuals/slide_r1_conclusion.png` | ⚠️ **To generate** | Summary of DevOps achievements |

---

## 🎥 Optional Screen Recording

To make the video dynamic, you can run and record:
1. `cat configs/config.yaml` or show it in the IDE to highlight the structure.
2. `pytest tests/ -v` running successfully in the terminal.
3. The GitHub Actions tab showing a successful workflow run.

## 📦 Delivery Checklist
- [ ] Record narration in English following the script above.
- [ ] Add the slide visuals to the video editor.
- [ ] Optional: overlay screen recordings of the terminal logs.
