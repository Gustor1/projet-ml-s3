# 🎙️ Répartition des Rôles — Projet ASR & Prétraitement Audio (Sujet 3)
> **Sujet choisi :** `Local Audio Preprocessing for Better ASR Performance`  
> **Groupe :** 6 étudiants | **Vidéo finale :** ≥ 10 min, en anglais, fun & fancy  
> **Règle d'or du prof :** L'historique GitHub sera vérifié. Chaque membre doit avoir des commits techniques réguliers et visibles.

---

## 📌 Contexte & Objectif du Projet
Construire un pipeline complet qui **prétraite l'audio localement** (réduction de bruit, VAD, enhancement, écho, etc.) pour **améliorer significativement la performance d'un modèle ASR** (Whisper, Wav2Vec, etc.).  
Le projet doit aller au-delà de l'usage basique d'API : il faut montrer des **insights techniques**, des **ablation studies**, des **trade-offs d'ingénierie**, et une **démo fonctionnelle**.

---

## 👥 Les 6 Rôles & Missions Techniques

### 1️⃣ Pipeline Architect & DevOps
- 🎯 **Mission :** Structurer le projet, assurer l'intégration des modules, gérer l'environnement et la CI/CD.
- 💻 **Commits attendus :** `main.py`, `config.yaml`, `utils/`, `requirements.txt`, `Dockerfile`, scripts d'installation, GitHub Actions.
- 📊 **Contribution :** ~17%
- 🛠️ **Stack :** Python, YAML, Docker/WSL, GitHub, CLI tools.

### 2️⃣ Audio Preprocessing Engineer
- 🎯 **Mission :** Développer les algorithmes de prétraitement audio (débruitage, VAD, annulation d'écho, beamforming, enhancement).
- 💻 **Commits attendus :** `preprocessing/denoise.py`, `preprocessing/vad.py`, `preprocessing/beamform.py`, `preprocessing/pipeline.py`, tests unitaires.
- 📊 **Contribution :** ~17%
- 🛠️ **Stack :** `librosa`, `noisereduce`, `webrtcvad`, `demucs`, `scipy.signal`.

### 3️⃣ ASR Integration & Evaluation Engineer
- 🎯 **Mission :** Intégrer les modèles ASR, calculer le WER/CER, comparer les performances avec/sans prétraitement.
- 💻 **Commits attendus :** `asr/whisper_wrapper.py`, `asr/wav2vec_wrapper.py`, `asr/evaluator.py`, scripts de benchmark, support multi-langue (EN/FR/ZH).
- 📊 **Contribution :** ~17%
- 🛠️ **Stack :** `transformers`, `jiwer`, `openai-whisper`, `pyannote` (si besoin).

### 4️⃣ Experimentation & Data Engineer
- 🎯 **Mission :** Préparer les datasets, concevoir les expériences, lancer les ablation studies, générer les métriques & graphiques.
- 💻 **Commits attendus :** `experiments/ablation_study.py`, `experiments/plot_results.py`, scripts de data loading, gestion des logs/CSV, notebooks d'analyse.
- 📊 **Contribution :** ~17%
- 🛠️ **Stack :** `matplotlib`, `seaborn`, `pandas`, AutoDL/GPU management, `wandb` ou `mlflow` (optionnel).

### 5️⃣ Optimization & Real-Time Performance Engineer
- 🎯 **Mission :** Profiler le pipeline, réduire la latence, optimiser la mémoire, tester le streaming temps réel & la quantization.
- 💻 **Commits attendus :** `optimization/profiler.py`, `optimization/streaming_audio.py`, `optimization/quantize_model.py`, benchmarks CPU/edge, rapports de performance.
- 📊 **Contribution :** ~16%
- 🛠️ **Stack :** `torch.profiler`, `onnxruntime`, `pydub`, `sounddevice`, Raspberry Pi / CPU-only tests.

### 6️⃣ Demo, Visualization & Video Production Engineer
- 🎯 **Mission :** Créer une démo interactive, exporter les assets audio/vidéo, produire la vidéo finale ≥10 min en anglais.
- 💻 **Commits attendus :** `demo/app.py` (Streamlit/Gradio), `demo/export_demo.py`, `demo/spectrogram_viz.py`, `docs/README.md`, assets de montage.
- 📊 **Contribution :** ~16%
- 🛠️ **Stack :** `streamlit`/`gradio`, `ffmpeg`, `moviepy`, `audacity`/`davinci resolve`, Markdown/Docs.

> ✅ **Note importante :** Le rôle 6 ne fait pas "que du montage vidéo". Il code d'abord les outils de démo et d'export automatique. La vidéo est un produit dérivé du code, pas une tâche isolée.

---

## 🛠️ Règles GitHub & Workflow (OBLIGATOIRE)
Le professeur vérifiera l'historique des commits. Pour que tout soit validé :

| Règle | Détail |
|-------|--------|
| 🌿 Branches | Chaque membre travaille sur sa branche : `feature/role-name` |
| 📝 Commits | Minimum **2 à 3 commits/semaine/personne** (même petits : `fix:`, `docs:`, `test:`) |
| 🔀 Merge | Toutes les intégrations passent par des **Pull Requests** avec review |
| 🤖 IA | Utilisation de Copilot/Cursor/Qwen fortement encouragée, mais **documenter les insights & trade-offs** |
| 📁 Structure | Respecter l'arborescence définie. Pas de fichiers en dehors des dossiers assignés |
| 📊 Traçabilité | Chaque issue GitHub = une tâche assignée à une personne |

---

## 📦 Checklist de Submission Finale
- [ ] Vidéo finale `≥ 10 min`, en **anglais**, fun & fancy
- [ ] Code source propre, structuré, hébergé sur GitHub (URL fournie)
- [ ] Fichier `submission.txt` avec rôle + % de contribution par membre
- [ ] Historique GitHub équilibré (commits visibles pour les 6)
- [ ] Dossier zippé → Upload Google Drive → Lien envoyé **en privé** (WeChat/Email)

---

## 🚀 Prochaines Étapes
1. ✅ Chaque membre **valide son rôle** dans ce fichier
2. 🍴 Fork du repo de base → Création de la branche dédiée
3. 💻 Setup environnement (WSL2/Docker/AutoDL) → **Premier commit sous 48h**
4. 🗓️ Point hebdomadaire de 30 min (avancement + blocages)
5. 🎬 Montage vidéo final 2 semaines avant deadline

> 📩 **En cas de doute ou de surcharge**, on réajuste ensemble en advance. L'objectif est un projet solide, bien documenté, et validé sans stress.

---
🔗 *Repo GitHub :* [À compléter]  
📅 *Deadline :* Premier vendredi de la semaine d'examens SHU, 23:59  
👥 *Groupe :* 6 étudiants | *Sujet :* 3 — Local Audio Preprocessing for Better ASR