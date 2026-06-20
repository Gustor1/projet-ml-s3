# 👤 Role: Eliott — Data & Experimentation Engineer
> Project: Local Audio Preprocessing for Better ASR Performance (Sujet 3)  
> Team: 6 students | Ref: `Project.md`, `role.md`

## 🎯 Core Mission
Design data pipelines, run ablation studies, compute ASR metrics (WER/CER/latency), and extract engineering trade-offs. Prove whether preprocessing actually improves downstream ASR.

##  Key Deliverables
- `scripts/` : dataset download, noise augmentation, emotion samples download & noise simulation
- `experiments/` : baseline, preprocessing comparison, SNR sweep, SER evaluation & sarcasm detection
- `results/` : CSV metrics (WER, CER, Perplexity, SER robustness), logs
- `visuals/` : plots for analysis & video (ASR degradation, SER accuracy bar charts)
- `docs/` : curated trade-offs, scientific insights, and experiment reports (including experiment-6-emotions.md)

## 🛠️ Tech & AI Setup
- IDE: VS Code + Continue + Cline
- Models (NVIDIA Build): `deepseek-v4-flash` (coding), `qwen3.5-122b` (reasoning/insights)
- Stack: Python, librosa, jiwer, transformers, matplotlib, pandas
- GPU: AutoDL or local WSL2/Docker

## 🔄 Workflow Principle
1. Work on `main` or task branches (`feat/xxx`). Merge frequently.
2. Commit at logical milestones (script done, experiment run, insight added). Let Cline suggest commit messages.
3. Validate AI output locally. Never blind copy-paste.
4. Document insights in `docs/insights.md` as we go.

## 📖 How to Use This File with AI
- Type `@eliott-data` in Continue/Cline chat to inject this context.
- Combine with `@codebase` for full repo awareness.
- AI should: generate modular code, respect existing structure, prioritize reproducibility, and flag trade-offs.
- AI should NOT: hardcode paths/keys, skip local validation, or ignore team conventions.