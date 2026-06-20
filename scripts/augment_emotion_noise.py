#!/usr/bin/env python3
"""
scripts/augment_emotion_noise.py
Adds white noise and urban noise to emotional speech samples at 20dB and 5dB SNR.
Generates data/emotion_augmented_metadata.json.
"""

import json
import logging
import numpy as np
import soundfile as sf
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_white_noise(signal, snr_db, seed=42):
    np.random.seed(seed)
    signal_power = np.mean(signal ** 2)
    if signal_power == 0:
        return signal
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    noisy_signal = signal + noise
    return np.clip(noisy_signal, -1.0, 1.0).astype(np.float32)

def load_urban_noise(noise_dir, target_length, seed=42):
    np.random.seed(seed)
    noise_files = list(Path(noise_dir).glob("*.wav"))
    if not noise_files:
        raise FileNotFoundError(f"No .wav files found in {noise_dir}")
    noise_file = np.random.choice(noise_files)
    noise, sr = sf.read(str(noise_file), dtype="float32")
    if len(noise.shape) > 1:
        noise = noise.mean(axis=1)
    if len(noise) < target_length:
        repeats = target_length // len(noise) + 1
        noise = np.tile(noise, repeats)
    return noise[:target_length], sr

def add_urban_noise(signal, noise, snr_db):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    if signal_power == 0 or noise_power == 0:
        return signal
    target_noise_power = signal_power / (10 ** (snr_db / 10.0))
    scale_factor = np.sqrt(target_noise_power / noise_power)
    noise_scaled = noise * scale_factor
    noisy_signal = signal + noise_scaled
    return np.clip(noisy_signal, -1.0, 1.0).astype(np.float32)

def main():
    input_metadata = Path("data/emotion_metadata.json")
    noise_dir = Path("data/urban_noise_16k")
    out_dir = Path("data/emotion_augmented")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_metadata.exists():
        raise FileNotFoundError(f"Metadata file not found: {input_metadata}")
        
    with open(input_metadata, "r", encoding="utf-8") as f:
        clean_meta = json.load(f)
        
    snr_levels = [20.0, 5.0]
    augmented_meta = []
    task_idx = 0
    
    # Total tasks = 24 files * 2 noise types * 2 SNR levels = 96 tasks
    total_tasks = len(clean_meta) * 2 * len(snr_levels)
    logger.info(f"Starting emotional audio augmentation: {total_tasks} total tasks.")
    
    for entry in clean_meta:
        src_path = Path(entry["file_path"])
        if not src_path.exists():
            logger.warning(f"File not found: {src_path}")
            continue
            
        signal, sr = sf.read(str(src_path), dtype="float32")
        if len(signal.shape) > 1:
            signal = signal.mean(axis=1)
            
        stem = src_path.stem
        
        # 1. White Gaussian Noise
        for snr in snr_levels:
            task_idx += 1
            noisy_signal = add_white_noise(signal, snr, seed=42 + task_idx)
            out_name = f"{stem}_white_snr{int(snr)}dB.wav"
            out_path = out_dir / out_name
            sf.write(str(out_path), noisy_signal, sr)
            
            augmented_meta.append({
                "file_path": str(out_path).replace("\\", "/"),
                "original_file": str(src_path).replace("\\", "/"),
                "snr_db": snr,
                "noise_type": "white_gaussian",
                "emotion_id": entry["emotion_id"],
                "emotion": entry["emotion"],
                "transcription": entry["transcription"],
                "duration": entry["duration"]
            })
            
        # 2. Urban Noise
        for snr in snr_levels:
            task_idx += 1
            try:
                noise, _ = load_urban_noise(noise_dir, len(signal), seed=42 + task_idx)
                noisy_signal = add_urban_noise(signal, noise, snr)
                out_name = f"{stem}_urban_snr{int(snr)}dB.wav"
                out_path = out_dir / out_name
                sf.write(str(out_path), noisy_signal, sr)
                
                augmented_meta.append({
                    "file_path": str(out_path).replace("\\", "/"),
                    "original_file": str(src_path).replace("\\", "/"),
                    "snr_db": snr,
                    "noise_type": "urban_real",
                    "emotion_id": entry["emotion_id"],
                    "emotion": entry["emotion"],
                    "transcription": entry["transcription"],
                    "duration": entry["duration"]
                })
            except Exception as e:
                logger.error(f"Failed to add urban noise to {stem}: {e}")
                
        if task_idx % 16 == 0 or task_idx == total_tasks:
            logger.info(f"Progress: [{task_idx}/{total_tasks}] tasks completed.")
            
    # Save metadata
    meta_path = Path("data/emotion_augmented_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(augmented_meta, f, indent=4)
        
    logger.info(f"Successfully generated {len(augmented_meta)} noisy emotional files.")
    logger.info(f"Augmented metadata saved to {meta_path}.")

if __name__ == "__main__":
    main()
