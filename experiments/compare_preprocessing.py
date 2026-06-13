#!/usr/bin/env python3
"""
experiments/compare_preprocessing.py
Compares ASR performance on noisy audio with different preprocessing methods.
Methods: none, wiener, spectral_subtraction
"""
import argparse
import csv
import json
import logging
import time
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.signal import wiener
from transformers import pipeline
import jiwer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_none(audio): return audio

def preprocess_wiener(audio):
    return wiener(audio).astype(np.float32)

def preprocess_spectral_subtraction(audio, sr, alpha=2.0, beta=0.01):
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
        result[i:i+nfft] += clean_frame

    max_val = np.max(np.abs(result))
    if max_val > 0:
        result = result / max_val
    return result[:n].astype(np.float32)

def run_comparison(metadata_path, output_csv, model_size="tiny"):
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    logger.info(f"Loading Whisper {model_size}...")
    asr = pipeline("automatic-speech-recognition", model=f"openai/whisper-{model_size}", device="cpu", ignore_warning=True)

    methods = {
        "none": preprocess_none,
        "wiener": preprocess_wiener,
        "spectral_subtraction": lambda x: preprocess_spectral_subtraction(x, 16000)
    }

    results = []
    total = len(meta) * len(methods)
    idx = 0

    for entry in meta:
        wav_path = Path(entry["file_path"])
        if not wav_path.exists(): continue
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if len(audio.shape) > 1: audio = audio.mean(axis=1)

        ref = entry["transcription"].strip().lower()
        snr = entry["snr_db"]

        for method_name, func in methods.items():
            idx += 1
            logger.info(f"[{idx}/{total}] {wav_path.name} | SNR={snr}dB | Method={method_name}")
            try:
                proc_audio = func(audio)
                start = time.time()
                out = asr(proc_audio)
                lat = (time.time() - start) * 1000
                pred = out["text"].strip().lower()
                wer = jiwer.wer(ref, pred) if ref else 1.0
                results.append({"file_name": wav_path.name, "snr_db": snr, "method": method_name, "wer": round(wer, 4), "latency_ms": round(lat, 2)})
                logger.info(f"  -> WER: {wer:.2%} | Latency: {lat:.0f}ms")
            except Exception as e:
                logger.error(f"Error: {e}")
                results.append({"file_name": wav_path.name, "snr_db": snr, "method": method_name, "wer": -1, "latency_ms": -1})

    out_csv = Path(output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "snr_db", "method", "wer", "latency_ms"])
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Results saved to {out_csv}")
    valid = [r for r in results if r["wer"] >= 0]
    for m in methods.keys():
        m_data = [r["wer"] for r in valid if r["method"] == m]
        if m_data:
            logger.info(f"Average WER ({m}): {sum(m_data)/len(m_data):.2%}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default="data/augmented_metadata.json")
    parser.add_argument("--output", type=str, default="results/preprocessing_comparison.csv")
    parser.add_argument("--model", type=str, default="tiny")
    args = parser.parse_args()
    run_comparison(args.metadata, args.output, args.model)

if __name__ == "__main__":
    main()