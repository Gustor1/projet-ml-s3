#!/usr/bin/env python3
"""
Convertit les fichiers de bruit urbain en WAV 16kHz mono pour compatibilité.
"""
import soundfile as sf
from pathlib import Path
import numpy as np

def convert_to_16k_mono(input_path, output_path):
    audio, sr = sf.read(str(input_path))
    
    # Convertir stéréo → mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Resampler si nécessaire
    if sr != 16000:
        from scipy.signal import resample
        target_samples = int(len(audio) * 16000 / sr)
        audio = resample(audio, target_samples)
        sr = 16000
    
    # Sauvegarder
    sf.write(str(output_path), audio, 16000, subtype='FLOAT')
    print(f"✅ Converti: {input_path.name} → {output_path.name} (16kHz mono)")

def main():
    input_dir = Path("data/urban_noise")
    output_dir = Path("data/urban_noise_16k")
    output_dir.mkdir(exist_ok=True)
    
    for wav_file in input_dir.glob("*.wav"):
        output_path = output_dir / wav_file.name
        convert_to_16k_mono(wav_file, output_path)
    
    print(f"\n✅ Tous les fichiers convertis dans {output_dir}")

if __name__ == "__main__":
    main()