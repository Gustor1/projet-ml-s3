# ASR Integration & Evaluation — Rôle 3

## Responsable
Boatel

## Mission
Intégrer les modèles ASR (Whisper, Wav2Vec2) et mesurer l'impact du
preprocessing audio sur le WER/CER, avec support multi-langue EN/FR/ZH.

## Structure
- `whisper_wrapper.py` — Wrapper Whisper (tiny / base / small / medium)
- `wav2vec_wrapper.py` — Wrapper Wav2Vec2 via HuggingFace
- `evaluator.py` — Calcul WER/CER avec jiwer, export CSV
- `benchmark.py` — Pipeline de benchmark avant/après preprocessing

## Stack
openai-whisper, transformers, jiwer, soundfile, pandas
