#!/usr/bin/env python3
"""
experiments/baseline_wer.py (v2 - soundfile only, no torchaudio)
Calcule le WER baseline avec Whisper sur LibriSpeech test-clean.
Compatible Windows PowerShell + CPU.
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_audio_wav(wav_path, target_sr=16000):
    """Charge un WAV avec soundfile (pas de torchaudio/torchcodec)."""
    audio, sr = sf.read(str(wav_path), dtype='float32')
    
    # Convertir stereo -> mono si besoin
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Resample si nécessaire (simple decimation/interpolation)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    return audio, target_sr

def run_baseline(metadata_path, output_path, model_size="base"):
    logger.info(f"📂 Chargement métadonnées: {metadata_path}")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    logger.info(f"🤖 Chargement Whisper {model_size} (premier run ~150MB)...")
    asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=f"openai/whisper-{model_size}",
    device="cpu",
    chunk_length_s=30,
    stride_length_s=5,
    ignore_warning=True  # ← Ajoute ceci pour supprimer le warning
)

    results = []
    total_files = len(metadata)
    
    for i, entry in enumerate(metadata):
        wav_path = Path(entry["file_path"])
        if not wav_path.exists():
            logger.warning(f"⚠️  Fichier manquant: {wav_path}")
            continue

        logger.info(f"[{i+1}/{total_files}] {wav_path.name}...")
        try:
            # Charger avec soundfile (PAS librosa.load qui utilise torchaudio)
            audio, sr = load_audio_wav(wav_path, target_sr=16000)
            
            start_time = time.time()
            # ✅ NOUVEAU
            asr_output = asr_pipeline(audio, generate_kwargs={"language": "en"})
            latency_ms = (time.time() - start_time) * 1000
            
            prediction = asr_output["text"].strip().lower()
            reference = entry["transcription"].strip().lower()
            
            wer = jiwer.wer(reference, prediction)
            
            results.append({
                "file_name": wav_path.name,
                "duration_s": entry["duration"],
                "reference": reference,
                "prediction": prediction,
                "wer": round(wer, 4),
                "latency_ms": round(latency_ms, 2)
            })
            logger.info(f"  ✅ WER: {wer:.2%} | Latence: {latency_ms:.0f}ms")
            
        except Exception as e:
            logger.error(f"❌ Erreur sur {wav_path.name}: {e}")
            results.append({
                "file_name": wav_path.name,
                "duration_s": entry["duration"],
                "reference": "",
                "prediction": "",
                "wer": -1,
                "latency_ms": -1
            })

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "duration_s", "reference", "prediction", "wer", "latency_ms"])
        writer.writeheader()
        writer.writerows(results)
    
    logger.info(f"💾 Résultats: {output_file}")
    
    valid_wers = [r["wer"] for r in results if r["wer"] >= 0]
    if valid_wers:
        avg_wer = sum(valid_wers) / len(valid_wers)
        avg_latency = sum(r["latency_ms"] for r in results if r["latency_ms"] > 0) / len(valid_wers)
        logger.info("=" * 50)
        logger.info(f"📊 BASELINE TERMINÉE")
        logger.info(f"   • Fichiers: {len(valid_wers)}/{total_files}")
        logger.info(f"   • WER moyen: {avg_wer:.2%}")
        logger.info(f"   • Latence moyenne: {avg_latency:.0f}ms")
        logger.info("=" * 50)
    else:
        logger.error(" Aucun résultat valide")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default="data/librispeech_metadata.json")
    parser.add_argument("--output", type=str, default="results/baseline.csv")
    parser.add_argument("--model", type=str, default="base", choices=["tiny", "base", "small"])
    args = parser.parse_args()
    run_baseline(args.metadata, args.output, args.model)

if __name__ == "__main__":
    main()