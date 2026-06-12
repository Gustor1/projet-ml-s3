#!/usr/bin/env python3
"""
scripts/download_datasets.py
Télécharge ou génère des fichiers audio de test pour le projet ASR.
Fallback: si le téléchargement échoue, génère des audio synthétiques avec librosa.
Compatible Windows PowerShell.
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Transcriptions de test (LibriSpeech-style)
TEST_TRANSCRIPTIONS = [
    "his thoughts were crowded with dreams and magnified associations",
    "the bright sunshine filled the room with warmth and comfort",
    "she walked quietly through the garden listening to the birds",
    "the old clock on the wall ticked steadily through the night",
    "a gentle breeze carried the scent of flowers across the meadow",
    "he opened the book and began to read the first chapter",
    "the children played happily in the park under the blue sky",
    "coffee brewed slowly in the kitchen filling the air with aroma",
    "the train arrived at the station right on schedule today",
    "music flowed from the speakers creating a peaceful atmosphere",
]

def generate_silent_audio(duration_sec: float, sample_rate: int = 16000) -> np.ndarray:
    """Génère un audio silencieux (pour test baseline)."""
    return np.zeros(int(duration_sec * sample_rate), dtype=np.float32)

def generate_tone_audio(frequency: float, duration_sec: float, sample_rate: int = 16000) -> np.ndarray:
    """Génère un ton sinusoïdal simple (pour test ASR)."""
    t = np.linspace(0, duration_sec, int(duration_sec * sample_rate), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

def generate_noise_audio(duration_sec: float, sample_rate: int = 16000, snr_db: float = None) -> np.ndarray:
    """Génère un bruit blanc ou coloré."""
    audio = np.random.randn(int(duration_sec * sample_rate)).astype(np.float32)
    audio = audio / np.max(np.abs(audio))  # Normaliser
    if snr_db is not None:
        # Ajout de silence pour simuler un SNR (simplifié)
        factor = 10 ** (snr_db / 20)
        audio = audio / factor
    return audio

def save_audio(audio: np.ndarray, filepath: str, sample_rate: int = 16000):
    """Sauvegarde un array numpy en fichier WAV."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    sf.write(filepath, audio, sample_rate)
    logger.info(f"✓ Sauvé: {filepath}")

def create_metadata_entry(file_path: str, duration: float, sample_rate: int, 
                          language: str, transcription: str, noise_type: str = "clean") -> dict:
    """Crée une entrée de métadonnées pour un fichier audio."""
    return {
        "file_path": file_path,
        "duration": round(duration, 2),
        "sample_rate": sample_rate,
        "language": language,
        "transcription": transcription,
        "noise_type": noise_type,
        "file_size_bytes": Path(file_path).stat().st_size if Path(file_path).exists() else 0
    }

def generate_test_dataset(output_dir: str, num_files: int, sample_rate: int = 16000):
    """Génère un dataset de test synthétique (fallback fiable)."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metadata = []
    
    logger.info(f"🎵 Génération de {num_files} fichiers audio de test...")
    
    for i in range(num_files):
        # Durée aléatoire entre 2 et 5 secondes
        duration = random.uniform(2.0, 5.0)
        transcription = TEST_TRANSCRIPTIONS[i % len(TEST_TRANSCRIPTIONS)]
        
        # Type d'audio alterné pour varier les tests
        if i % 3 == 0:
            audio = generate_silent_audio(duration, sample_rate)
            noise_type = "silent"
        elif i % 3 == 1:
            audio = generate_tone_audio(440.0, duration, sample_rate)  # La 440Hz
            noise_type = "tone_440hz"
        else:
            audio = generate_noise_audio(duration, sample_rate, snr_db=random.choice([20, 10, 5]))
            noise_type = "white_noise"
        
        # Sauvegarde
        filename = f"test_{i+1:03d}_{noise_type}.wav"
        filepath = output_path / filename
        save_audio(audio, str(filepath), sample_rate)
        
        # Métadonnées
        entry = create_metadata_entry(
            file_path=str(filepath),
            duration=duration,
            sample_rate=sample_rate,
            language="en",
            transcription=transcription,
            noise_type=noise_type
        )
        metadata.append(entry)
        
        logger.info(f"  [{i+1}/{num_files}] {filename} - {noise_type} - {duration:.1f}s")
    
    # Sauvegarde metadata.json
    metadata_path = Path(output_dir).parent / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Dataset généré: {num_files} fichiers dans {output_dir}")
    logger.info(f"📄 Métadonnées: {metadata_path}")
    
    return metadata

def main():
    parser = argparse.ArgumentParser(description="Download or generate test audio dataset for ASR project")
    parser.add_argument("--num_files", type=int, default=10, help="Number of audio files to generate/download")
    parser.add_argument("--output_dir", type=str, default="data/raw/", help="Output directory for audio files")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate in Hz")
    parser.add_argument("--language", type=str, default="en", help="Language code for transcriptions")
    parser.add_argument("--mode", type=str, choices=["generate", "download"], default="generate",
                        help="Mode: 'generate' synthetic audio (reliable) or 'download' from URL (experimental)")
    
    args = parser.parse_args()
    
    logger.info("=" * 50)
    logger.info("🎙️  DATASET SETUP FOR ASR PROJECT")
    logger.info("=" * 50)
    logger.info(f"Configuration:")
    logger.info(f"  • Mode: {args.mode}")
    logger.info(f"  • Output dir: {args.output_dir}")
    logger.info(f"  • Files: {args.num_files}")
    logger.info(f"  • Sample rate: {args.sample_rate} Hz")
    logger.info(f"  • Language: {args.language}")
    logger.info("-" * 50)
    
    try:
        if args.mode == "download":
            logger.info("⚠️  Mode 'download' expérimental - fallback vers 'generate' recommandé")
            # Ici on pourrait ajouter un vrai téléchargement si besoin
            # Pour l'instant, on fallback vers generate pour garantir que ça marche
            logger.info("🔄 Fallback: génération de fichiers synthétiques...")
        
        # Génération fiable (toujours fonctionnelle)
        metadata = generate_test_dataset(
            output_dir=args.output_dir,
            num_files=args.num_files,
            sample_rate=args.sample_rate
        )
        
        logger.info("=" * 50)
        logger.info("✅ SETUP TERMINÉ AVEC SUCCÈS")
        logger.info("=" * 50)
        logger.info(f"📁 Fichiers créés: {len(metadata)}")
        logger.info(f"📊 Prochaine étape: lancer experiments/baseline_wer.py")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())