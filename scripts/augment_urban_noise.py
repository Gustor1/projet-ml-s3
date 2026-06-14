#!/usr/bin/env python3
"""
scripts/augment_urban_noise.py
Augmente l'audio LibriSpeech avec de vrais bruits urbains (trafic, café, rue) 
à différents niveaux de SNR (Signal-to-Noise Ratio).
"""
import argparse
import json
import logging
import numpy as np
import soundfile as sf
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_urban_noise(noise_dir, target_length, seed=42):
    """
    Charge un fichier de bruit urbain aléatoire et l'adapte à la longueur cible.
    """
    np.random.seed(seed)
    noise_files = list(Path(noise_dir).glob("*.wav"))
    
    if not noise_files:
        raise FileNotFoundError(f"Aucun fichier .wav trouvé dans {noise_dir}")
    
    # Choisir un fichier de bruit aléatoire
    noise_file = np.random.choice(noise_files)
    noise, sr = sf.read(str(noise_file), dtype="float32")
    
    # Convertir stéréo -> mono si nécessaire
    if len(noise.shape) > 1:
        noise = noise.mean(axis=1)
    
    # Répéter le bruit s'il est plus court que l'audio cible
    if len(noise) < target_length:
        repeats = target_length // len(noise) + 1
        noise = np.tile(noise, repeats)
    
    # Tronquer à la longueur exacte
    noise = noise[:target_length]
    return noise, sr

def add_urban_noise(signal, noise, snr_db):
    """
    Ajoute le bruit au signal avec le SNR (en dB) spécifié.
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    # Éviter la division par zéro si le signal est silencieux
    if signal_power == 0:
        return signal
        
    # Calculer la puissance cible du bruit pour atteindre le SNR désiré
    target_noise_power = signal_power / (10 ** (snr_db / 10.0))
    
    # Facteur d'échelle pour ajuster le bruit
    scale_factor = np.sqrt(target_noise_power / noise_power)
    noise_scaled = noise * scale_factor
    
    # Mélanger et clipper pour éviter la saturation
    noisy_signal = signal + noise_scaled
    return np.clip(noisy_signal, -1.0, 1.0).astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="Augment audio with urban noise")
    parser.add_argument("--input_metadata", type=str, default="data/librispeech_metadata.json")
    parser.add_argument("--noise_dir", type=str, default="data/urban_noise_16k")
    parser.add_argument("--output_dir", type=str, default="data/augmented_urban")
    parser.add_argument("--snr_values", type=str, default="20,10,5")
    args = parser.parse_args()

    snr_list = [float(x) for x in args.snr_values.split(",")]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Charger les métadonnées de l'audio propre
    with open(args.input_metadata, "r", encoding="utf-8") as f:
        clean_meta = json.load(f)

    aug_meta = []
    total_tasks = len(clean_meta) * len(snr_list)
    task_idx = 0

    logger.info(f"Starting augmentation: {len(clean_meta)} files x {len(snr_list)} SNR levels = {total_tasks} tasks")

    for entry in clean_meta:
        src_path = Path(entry["file_path"])
        if not src_path.exists():
            logger.warning(f"Missing file: {src_path}")
            continue

        # Lire l'audio propre
        signal, sr = sf.read(str(src_path), dtype="float32")
        if len(signal.shape) > 1:
            signal = signal.mean(axis=1)

        stem = src_path.stem
        
        for snr in snr_list:
            task_idx += 1
            
            # Charger le bruit urbain (longueur adaptée)
            noise, _ = load_urban_noise(args.noise_dir, len(signal), seed=42 + task_idx)
            
            # Ajouter le bruit
            noisy_signal = add_urban_noise(signal, noise, snr)
            
            # Nommer et sauvegarder le fichier
            out_name = f"{stem}_urban_snr{int(snr)}dB.wav"
            out_path = out_dir / out_name
            sf.write(str(out_path), noisy_signal, sr)

            # Ajouter aux métadonnées
            aug_meta.append({
                "file_path": str(out_path),
                "original_file": str(src_path),
                "snr_db": snr,
                "noise_type": "urban_real",
                "duration": entry.get("duration", len(signal)/sr),
                "sample_rate": sr,
                "transcription": entry["transcription"],
                "language": "en"
            })
            
            if task_idx % 10 == 0:
                logger.info(f"[{task_idx}/{total_tasks}] Processed {out_name}")

    # Sauvegarder le nouveau fichier de métadonnées
    meta_path = Path(args.input_metadata).parent / "augmented_urban_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(aug_meta, f, indent=2, ensure_ascii=False)

    logger.info(f"{'='*60}")
    logger.info(f"✅ Done. {len(aug_meta)} urban-noise files saved to {out_dir}")
    logger.info(f"📄 Metadata saved to {meta_path}")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()