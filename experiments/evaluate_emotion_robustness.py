#!/usr/bin/env python3
"""
experiments/evaluate_emotion_robustness.py
Evaluates Speech Emotion Recognition (SER) accuracy using superb/wav2vec2-base-superb-er
across different noise types (white, urban), SNR levels (20dB, 5dB), and preprocessing methods (None, Wiener, SpecSub).
"""

import csv
import json
import logging
import time
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.signal import wiener
import torch
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Preprocessing methods
def preprocess_none(audio): 
    return audio

def preprocess_wiener(audio):
    return wiener(audio).astype(np.float32)

def preprocess_spectral_subtraction(audio, sr=16000, alpha=2.0, beta=0.01):
    n = len(audio)
    nfft = 2048
    hop = nfft // 2
    noise_len = int(0.5 * sr)
    noise_frame = audio[:noise_len]
    noise_spec = np.abs(np.fft.rfft(noise_frame, n=nfft))
    noise_pow = np.mean(noise_spec ** 2)

    result = np.zeros(n)
    window = np.hanning(nfft)
    
    for i in range(0, n, hop):
        frame = audio[i:i+nfft]
        if len(frame) < nfft:
            frame = np.pad(frame, (0, nfft - len(frame)))
        
        spec = np.fft.rfft(frame * window)
        power = np.abs(spec) ** 2
        clean_pow = np.maximum(power - alpha * noise_pow, beta * noise_pow)
        clean_spec = np.sqrt(clean_pow) * np.exp(1j * np.angle(spec))
        clean_frame = np.fft.irfft(clean_spec) * window
        
        chunk_len = min(nfft, n - i)
        result[i:i+chunk_len] += clean_frame[:chunk_len]

    max_val = np.max(np.abs(result))
    if max_val > 0:
        result = result / max_val
    return result[:n].astype(np.float32)

# Map RAVDESS emotion strings to Superb classes
EMOTION_MAP = {
    "neutral": "neu",
    "happy": "hap",
    "sad": "sad",
    "angry": "ang"
}

def main():
    clean_metadata_path = Path("data/emotion_metadata.json")
    noisy_metadata_path = Path("data/emotion_augmented_metadata.json")
    output_csv = Path("results/emotion_robustness.csv")
    
    if not clean_metadata_path.exists() or not noisy_metadata_path.exists():
        raise FileNotFoundError("Emotional dataset metadata not found. Run download and augment scripts first.")
        
    logger.info("Initializing SER classifier (superb/wav2vec2-base-superb-er)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ser_pipeline = pipeline(
        "audio-classification",
        model="superb/wav2vec2-base-superb-er",
        device=device
    )
    
    # 1. Run Baseline (Clean audio)
    logger.info("Running baseline evaluation on clean speech...")
    with open(clean_metadata_path, "r", encoding="utf-8") as f:
        clean_meta = json.load(f)
        
    clean_audios = []
    valid_clean_meta = []
    
    for entry in clean_meta:
        src_path = Path(entry["file_path"])
        if not src_path.exists():
            continue
            
        audio, sr = sf.read(str(src_path), dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
            
        clean_audios.append({"raw": audio, "sampling_rate": 16000})
        valid_clean_meta.append(entry)
        
    logger.info(f"Classifying {len(clean_audios)} clean speech files using batching...")
    # Run batch inference (batch_size=32 is efficient on CPU)
    preds_list = ser_pipeline(clean_audios, batch_size=32)
    
    clean_correct = 0
    clean_results = []
    
    for entry, preds in zip(valid_clean_meta, preds_list):
        pred_label = preds[0]["label"]
        true_label = EMOTION_MAP[entry["emotion"]]
        
        is_correct = 1 if pred_label == true_label else 0
        clean_correct += is_correct
        clean_results.append({
            "file_name": Path(entry["file_path"]).name,
            "true_emotion": true_label,
            "pred_emotion": pred_label,
            "correct": is_correct
        })
        
    clean_acc = clean_correct / len(valid_clean_meta) if valid_clean_meta else 0.0
    logger.info(f"Clean speech accuracy (Baseline): {clean_acc:.2%}")
    
    # 2. Run Noisy & Preprocessed Evaluation
    logger.info("Running robust evaluation on noisy & preprocessed speech...")
    with open(noisy_metadata_path, "r", encoding="utf-8") as f:
        noisy_meta = json.load(f)
        
    all_noisy_signals = []
    valid_noisy_meta = []
    
    logger.info("Generating preprocessed waveforms...")
    for entry in noisy_meta:
        src_path = Path(entry["file_path"])
        if not src_path.exists():
            continue
            
        audio, sr = sf.read(str(src_path), dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
            
        valid_noisy_meta.append(entry)
        
        # Method A: None (Noisy raw)
        audio_none = preprocess_none(audio)
        all_noisy_signals.append({"raw": audio_none, "sampling_rate": 16000})
        
        # Method B: Wiener
        audio_wiener = preprocess_wiener(audio)
        all_noisy_signals.append({"raw": audio_wiener, "sampling_rate": 16000})
        
        # Method C: Spectral Subtraction
        audio_spec_sub = preprocess_spectral_subtraction(audio, sr)
        all_noisy_signals.append({"raw": audio_spec_sub, "sampling_rate": 16000})
        
    logger.info(f"Classifying {len(all_noisy_signals)} noisy & preprocessed signals using batching...")
    # Run batch inference for all processed files at once
    noisy_preds_list = ser_pipeline(all_noisy_signals, batch_size=32)
    
    results = []
    for idx, entry in enumerate(valid_noisy_meta):
        true_label = EMOTION_MAP[entry["emotion"]]
        
        pred_none = noisy_preds_list[3 * idx][0]["label"]
        pred_wiener = noisy_preds_list[3 * idx + 1][0]["label"]
        pred_spec_sub = noisy_preds_list[3 * idx + 2][0]["label"]
        
        correct_none = 1 if pred_none == true_label else 0
        correct_wiener = 1 if pred_wiener == true_label else 0
        correct_spec_sub = 1 if pred_spec_sub == true_label else 0
        
        results.append({
            "file_name": Path(entry["file_path"]).name,
            "noise_type": entry["noise_type"],
            "snr_db": entry["snr_db"],
            "true_emotion": true_label,
            "pred_none": pred_none,
            "pred_wiener": pred_wiener,
            "pred_spec_sub": pred_spec_sub,
            "correct_none": correct_none,
            "correct_wiener": correct_wiener,
            "correct_spec_sub": correct_spec_sub
        })
            
    # Save CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file_name", "noise_type", "snr_db", "true_emotion", 
            "pred_none", "pred_wiener", "pred_spec_sub", 
            "correct_none", "correct_wiener", "correct_spec_sub"
        ])
        writer.writeheader()
        writer.writerows(results)
        
    logger.info(f"Results saved to {output_csv}")
    
    # Save Clean Results CSV
    clean_csv = output_csv.parent / "emotion_clean_results.csv"
    with open(clean_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "true_emotion", "pred_emotion", "correct"])
        writer.writeheader()
        writer.writerows(clean_results)
        
    logger.info(f"Clean speech results saved to {clean_csv}")
    
    # 3. Print Summarized Accuracy report
    print("\n" + "="*50)
    print("=== SPEECH EMOTION RECOGNITION ACCURACY SUMMARY ===")
    print("="*50)
    print(f"Clean Audio Baseline accuracy: {clean_acc:.2%}")
    print("-"*50)
    
    for noise_type in ["white_gaussian", "urban_real"]:
        for snr in [20.0, 5.0]:
            subset = [r for r in results if r["noise_type"] == noise_type and float(r["snr_db"]) == snr]
            if not subset:
                continue
                
            acc_none = sum(r["correct_none"] for r in subset) / len(subset)
            acc_wiener = sum(r["correct_wiener"] for r in subset) / len(subset)
            acc_spec_sub = sum(r["correct_spec_sub"] for r in subset) / len(subset)
            
            print(f"Noise: {noise_type} | SNR: {int(snr)}dB (N={len(subset)})")
            print(f"  - Raw Noisy (None): {acc_none:.2%}")
            print(f"  - Wiener Filter  : {acc_wiener:.2%}")
            print(f"  - Spec Subtraction: {acc_spec_sub:.2%}")
            print("-" * 50)
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
