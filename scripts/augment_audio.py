#!/usr/bin/env python3
"""
scripts/augment_audio.py
Augments clean audio with controlled Gaussian noise at specified SNR levels.
Outputs augmented WAVs and a new metadata JSON.
"""
import argparse
import json
import logging
import sys
import numpy as np
import soundfile as sf
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_noise(signal, snr_db):
    """Adds white Gaussian noise to achieve target SNR."""
    np.random.seed(42)  # Reproducibility
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    noisy_signal = signal + noise
    return np.clip(noisy_signal, -1.0, 1.0).astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="Augment audio with controlled noise")
    parser.add_argument("--input_metadata", type=str, default="data/librispeech_metadata.json")
    parser.add_argument("--output_dir", type=str, default="data/augmented")
    parser.add_argument("--snr_values", type=str, default="20,10,5", help="Comma-separated SNR values in dB")
    args = parser.parse_args()

    snr_list = [float(x) for x in args.snr_values.split(",")]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input_metadata, "r", encoding="utf-8") as f:
        clean_meta = json.load(f)

    aug_meta = []
    total_tasks = len(clean_meta) * len(snr_list)
    task_idx = 0

    for entry in clean_meta:
        src_path = Path(entry["file_path"])
        if not src_path.exists():
            logger.warning(f"Missing: {src_path}")
            continue

        signal, sr = sf.read(str(src_path), dtype="float32")
        if len(signal.shape) > 1:
            signal = signal.mean(axis=1)

        stem = src_path.stem
        for snr in snr_list:
            task_idx += 1
            noisy_signal = add_noise(signal, snr)
            out_name = f"{stem}_snr{int(snr)}dB.wav"
            out_path = out_dir / out_name

            sf.write(str(out_path), noisy_signal, sr)

            aug_meta.append({
                "file_path": str(out_path),
                "original_file": str(src_path),
                "snr_db": snr,
                "noise_type": "white_gaussian",
                "duration": entry["duration"],
                "sample_rate": sr,
                "transcription": entry["transcription"],
                "language": "en"
            })
            logger.info(f"[{task_idx}/{total_tasks}] Created {out_name} (SNR={snr}dB)")

    meta_path = Path(args.input_metadata).parent / "augmented_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(aug_meta, f, indent=2, ensure_ascii=False)

    logger.info(f"Done. {len(aug_meta)} augmented files saved to {out_dir}")
    logger.info(f"Metadata saved to {meta_path}")

if __name__ == "__main__":
    main()