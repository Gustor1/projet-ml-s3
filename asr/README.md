# ASR Integration & Evaluation — Rôle 3

## Responsable
Boatel

## Mission
Intégrer les modèles ASR (Whisper, Wav2Vec2) et mesurer l'impact du
preprocessing audio sur le WER/CER, avec support multi-langue EN/FR/ZH.
Analyser comment les erreurs de transcription ASR se propagent dans la
détection de sentiment et de sarcasme (étude d'ablation cross-modale).

## Structure
- `base_asr.py` — Classe abstraite `BaseASR` avec `transcribe_batch` error-safe
- `whisper_wrapper.py` — Wrapper Whisper (tiny / base / small / medium), GPU-aware, EN/FR/ZH
- `wav2vec_wrapper.py` — Wrapper Wav2Vec2 via HuggingFace, GPU-aware, EN/FR/ZH
- `evaluator.py` — Calcul WER/CER avec jiwer, export CSV horodaté
- `benchmark.py` — Pipeline de benchmark raw vs. preprocessed avec RTF
- `ablation_study.py` — Comparaison WER/CER/RTF par taille de modèle Whisper
- `cross_modal_ablation.py` — **Étude cross-modale** : impact des erreurs ASR sur le sentiment (DistilBERT) et la détection de sarcasme (mismatch voix/texte)

## Cross-Modal Ablation Study

### Méthodologie
Pour chaque taille de modèle Whisper (tiny, base, small) :
1. Transcription ASR de fichiers audio RAVDESS
2. Analyse sentiment texte avec DistilBERT (sur transcription ASR **et** texte de référence)
3. Détection émotion vocale avec Wav2Vec2-SER
4. Détection de sarcasme par mismatch voix/texte
5. Comparaison des verdicts sarcasme ASR vs. ground-truth

### Métriques
| Métrique | Description |
|---|---|
| Sentiment Flip Rate | % de cas où le sentiment change entre texte ASR et texte de référence |
| Sarcasm False Positive Rate | % de faux sarcasmes déclenchés par des erreurs ASR |
| Sarcasm False Negative Rate | % de sarcasmes manqués à cause d'erreurs ASR |
| Agreement Rate | % de verdicts sarcasme identiques entre pipeline ASR et ground-truth |

### Usage
```bash
python -m asr.cross_modal_ablation --metadata data/emotion_metadata.json
python -m asr.cross_modal_ablation --models tiny base small --output results
```

## Stack
openai-whisper, transformers, jiwer, soundfile, pandas, torch
