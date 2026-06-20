#!/usr/bin/env python3
"""
scripts/augment_babble_noise.py
Simule un bruit de foule (babble) en superposant aléatoirement 
d'autres fichiers audio du dataset en arrière-plan.
"""
import argparse
import json
import logging
import numpy as np
import soundfile as sf
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_babble(target_audio_len, all_audios, seed):
    """Génère un bruit de foule en mélangeant 3 à 5 autres fichiers audio."""
    random.seed(seed)
    np.random.seed(seed)
    
    babble = np.zeros(target_audio_len, dtype=np.float32)
    num_speakers = random.randint(3, 5)
    
    for _ in range(num_speakers):
        # Choisir un fichier aléatoire (différent du target, géré par l'appelant)
        other_audio = random.choice(all_audios)
        other_len = len(other_audio)
        
        # Prendre un segment aléatoire de la même longueur que le target
        if other_len >= target_audio_len:
            start = random.randint(0, other_len - target_audio_len)
            segment = other_audio[start:start + target_audio_len]
        else:
            # Si le fichier est trop court, on le répète
            repeats = target_audio_len // other_len + 1
            segment = np.tile(other_audio, repeats)[:target_audio_len]
            
        # Ajouter avec un volume aléatoire pour varier les voix
        volume = random.uniform(0.5, 1.5)
        babble += segment * volume
        
    return babble

def add_babble(signal, babble, snr_db):
    """Ajoute le babble au signal avec le SNR spécifié."""
    signal_power = np.mean(signal ** 2)
    babble_power = np.mean(babble ** 2)
    
    if signal_power == 0 or babble_power == 0:
        return signal
        
    target_babble_power = signal_power / (10 ** (snr_db / 10.0))
    scale_factor = np.sqrt(target_babble_power / babble_power)
    
    noisy_signal = signal + babble * scale_factor
    return np.clip(noisy_signal, -1.0, 1.0).astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="Augment audio with babble noise")
    parser.add_argument("--input_metadata", type=str, default="data/librispeech_metadata.json")
    parser.add_argument("--output_dir", type=str, default="data/augmented_babble")
    parser.add_argument("--snr_values", type=str, default="20,10,5")
    args = parser.parse_args()

    snr_list = [float(x) for x in args.snr_values.split(",")]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input_metadata, "r", encoding="utf-8") as f:
        clean_meta = json.load(f)

    # Charger tous les audios en mémoire pour le mélange
    logger.info("Chargement de tous les fichiers audio pour la simulation de foule...")
    all_audios = []
    for entry in clean_meta:
        src_path = Path(entry["file_path"])
        if src_path.exists():
            audio, _ = sf.read(str(src_path), dtype="float32")
            if len(audio.shape) > 1: audio = audio.mean(axis=1)
            all_audios.append(audio)
    
    logger.info(f"✅ {len(all_audios)} fichiers chargés.")

    aug_meta = []
    total_tasks = len(clean_meta) * len(snr_list)
    task_idx = 0

    for entry in clean_meta:
        src_path = Path(entry["file_path"])
        if not src_path.exists(): continue

        signal, sr = sf.read(str(src_path), dtype="float32")
        if len(signal.shape) > 1: signal = signal.mean(axis=1)

        # Filtrer pour ne pas utiliser le fichier target dans le babble
        other_audios = [a for a in all_audios if len(a) != len(signal) or not np.array_equal(a, signal)]

        stem = src_path.stem
        for snr in snr_list:
            task_idx += 1
            
            # Générer le babble spécifique pour ce fichier et ce SNR
            babble = generate_babble(len(signal), other_audios, seed=42 + task_idx)
            noisy_signal = add_babble(signal, babble, snr)
            
            out_name = f"{stem}_babble_snr{int(snr)}dB.wav"
            out_path = out_dir / out_name
            sf.write(str(out_path), noisy_signal, sr)

            aug_meta.append({
                "file_path": str(out_path),
                "original_file": str(src_path),
                "snr_db": snr,
                "noise_type": "babble_crowd",
                "duration": entry.get("duration", len(signal)/sr),
                "sample_rate": sr,
                "transcription": entry["transcription"],
                "language": "en"
            })
            
            if task_idx % 10 == 0:
                logger.info(f"[{task_idx}/{total_tasks}] Processed {out_name}")

    meta_path = Path(args.input_metadata).parent / "augmented_babble_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(aug_meta, f, indent=2, ensure_ascii=False)

    logger.info(f"{'='*60}")
    logger.info(f"✅ Done. {len(aug_meta)} babble-noise files saved to {out_dir}")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()