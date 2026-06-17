#!/usr/bin/env python3
"""
experiments/compare_preprocessing.py
Compares ASR performance on noisy audio with different preprocessing methods.
Methods: none, wiener, spectral_subtraction
NEW: Extracts decoder perplexity for mechanistic analysis of hallucinations.
"""
import argparse
import csv
import json
import logging
import random
import time
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
from scipy.signal import wiener
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import jiwer

np.random.seed(42)
random.seed(42)

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
        
        chunk_len = min(nfft, n - i)
        result[i:i+chunk_len] += clean_frame[:chunk_len]

    max_val = np.max(np.abs(result))
    if max_val > 0:
        result = result / max_val
    return result[:n].astype(np.float32)

def extract_perplexity(model, processor, audio, reference_text, sr=16000):
    """
    Extract decoder perplexity (exp(cross-entropy loss)) for a given audio.
    Uses the REFERENCE transcription as labels to measure how 'surprised' 
    the model is by the correct text given the noisy audio.
    
    High perplexity = model is very uncertain about the correct transcription.
    """
    try:
        # Prepare input features
        input_features = processor(
            audio, 
            sampling_rate=sr, 
            return_tensors="pt"
        ).input_features.to(model.device)
        
        # Tokenize reference text as labels
        labels = processor(
            text=reference_text,
            return_tensors="pt"
        ).input_ids.to(model.device)
        
        # Get model outputs with loss (cross-entropy against reference)
        with torch.no_grad():
            outputs = model(input_features, labels=labels)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        
        return round(perplexity, 2)
    except Exception as e:
        logger.warning(f"Perplexity extraction failed: {e}")
        return -1.0

def run_comparison(metadata_path, output_csv, model_size="tiny"):
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    logger.info(f"Loading Whisper {model_size}...")
    # Load model and processor directly for perplexity access
    model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{model_size}").to("cpu")
    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{model_size}")
    
    # Create pipeline for transcription (convenience)
    from transformers import pipeline
    asr = pipeline(
        "automatic-speech-recognition", 
        model=f"openai/whisper-{model_size}", 
        device="cpu"
    )

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
        if not wav_path.exists(): 
            continue
            
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if len(audio.shape) > 1: 
            audio = audio.mean(axis=1)

        ref = entry["transcription"].strip().lower()
        snr = entry["snr_db"]

        for method_name, func in methods.items():
            idx += 1
            logger.info(f"[{idx}/{total}] {wav_path.name} | SNR={snr}dB | Method={method_name}")
            
            try:
                proc_audio = func(audio)
                
                # Transcription + latency
                start = time.time()
                out = asr(proc_audio)
                lat = (time.time() - start) * 1000
                
                # Perplexity extraction (mechanistic insight)
                # Uses REFERENCE text as labels to measure decoder uncertainty
                perplexity = extract_perplexity(model, processor, proc_audio, ref, sr)
                
                pred = out["text"].strip().lower()
                wer = jiwer.wer(ref, pred) if ref else 1.0
                cer = jiwer.cer(ref, pred) if ref else 1.0
                
                results.append({
                    "file_name": wav_path.name, 
                    "snr_db": snr, 
                    "method": method_name, 
                    "wer": round(wer, 4), 
                    "cer": round(cer, 4), 
                    "latency_ms": round(lat, 2),
                    "perplexity": perplexity
                })
                
                logger.info(f"  -> WER: {wer:.2%} | CER: {cer:.2%} | Latency: {lat:.0f}ms | Perplexity: {perplexity:.1f}")
                
            except Exception as e:
                logger.error(f"Error: {e}")
                results.append({
                    "file_name": wav_path.name, 
                    "snr_db": snr, 
                    "method": method_name, 
                    "wer": -1, 
                    "cer": -1, 
                    "latency_ms": -1,
                    "perplexity": -1
                })

    out_csv = Path(output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file_name", "snr_db", "method", "wer", "cer", "latency_ms", "perplexity"
        ])
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Results saved to {out_csv}")
    valid = [r for r in results if r["wer"] >= 0]
    for m in methods.keys():
        m_data = [r["wer"] for r in valid if r["method"] == m]
        if m_data:
            logger.info(f"Average WER ({m}): {sum(m_data)/len(m_data):.2%}")
        
        p_data = [r["perplexity"] for r in valid if r["method"] == m and r["perplexity"] > 0]
        if p_data:
            logger.info(f"Average Perplexity ({m}): {sum(p_data)/len(p_data):.1f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default="data/augmented_metadata.json")
    parser.add_argument("--output", type=str, default="results/preprocessing_comparison.csv")
    parser.add_argument("--model", type=str, default="tiny")
    args = parser.parse_args()
    run_comparison(args.metadata, args.output, args.model)

if __name__ == "__main__":
    main()