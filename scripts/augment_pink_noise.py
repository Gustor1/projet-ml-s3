#!/usr/bin/env python3
"""
scripts/augment_pink_noise.py
Augments clean audio with pink noise (1/f spectrum) at specified SNR levels.
Pink noise is more realistic than white noise for real-world scenarios.
"""
import argparse
import json
import logging
import numpy as np
import soundfile as sf
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def pink_noise(length, seed=42):
    """
    Generate pink noise using Voss-McCartney algorithm.
    Produces 1/f spectrum (more energy in low frequencies).
    """
    np.random.seed(seed)
    num_rows = 16
    cols = length // num_rows + 1
    array = np.random.randn(num_rows, cols)
    pink = np.sum(array, axis=0)
    pink = np.repeat(pink, num_rows)[:length]
    # Normalize to unit variance
    pink = pink / np.std(pink)
    return pink

def add_pink_noise(signal, snr_db, seed=42):
    """Adds pink noise to achieve target SNR."""
    noise = pink_noise(len(signal), seed=seed)
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noisy_signal = signal + np.sqrt(noise_power) * noise
    return np.clip(noisy_signal, -1.0, 1.0).astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="Augment audio with pink noise")
    parser.add_argument("--input_metadata", type=str, default="data/librispeech_metadata.json")
    parser.add_argument("--output_dir", type=str, default="data/augmented_pink")
    parser.add_argument("--snr_values", type=str, default="20,10,5")
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
            noisy_signal = add_pink_noise(signal, snr)
            out_name = f"{stem}_pink_snr{int(snr)}dB.wav"
            out_path = out_dir / out_name

            sf.write(str(out_path), noisy_signal, sr)

            aug_meta.append({
                "file_path": str(out_path),
                "original_file": str(src_path),
                "snr_db": snr,
                "noise_type": "pink_1f",
                "duration": entry["duration"],
                "sample_rate": sr,
                "transcription": entry["transcription"],
                "language": "en"
            })
            logger.info(f"[{task_idx}/{total_tasks}] Created {out_name} (SNR={snr}dB)")

    meta_path = Path(args.input_metadata).parent / "augmented_pink_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(aug_meta, f, indent=2, ensure_ascii=False)

    logger.info(f"Done. {len(aug_meta)} pink-noise files saved to {out_dir}")
    logger.info(f"Metadata saved to {meta_path}")

if __name__ == "__main__":
    main()