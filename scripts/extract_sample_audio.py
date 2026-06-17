#!/usr/bin/env python3
"""
scripts/extract_sample_audio.py
Extrait un échantillon audio et applique le filtre de Wiener pour les spectrogrammes.
"""
import numpy as np
from scipy.signal import wiener
from scipy.io import wavfile
import soundfile as sf
from pathlib import Path
import json

def main():
    # Charger les métadonnées pour trouver un fichier pink noise
    metadata_path = Path("data/augmented_pink_metadata.json")
    if not metadata_path.exists():
        print("❌ Metadata file not found. Please run augment_pink_noise.py first.")
        return
    
    with open(metadata_path, "r") as f:
        meta = json.load(f)
    
    # Trouver un fichier à 5dB SNR
    sample_5db = [m for m in meta if m["snr_db"] == 5.0]
    if not sample_5db:
        print("❌ No 5dB SNR samples found.")
        return
    
    # Prendre le premier échantillon
    sample = sample_5db[0]
    file_path = Path(sample["file_path"])
    
    print(f"📁 Fichier source: {file_path}")
    
    # Charger l'audio
    audio, sr = sf.read(str(file_path), dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Appliquer le filtre de Wiener
    print("🔧 Application du filtre de Wiener...")
    wiener_audio = wiener(audio).astype(np.float32)
    
    # Créer le dossier de sortie
    output_dir = Path("data/audio_samples")
    output_dir.mkdir(exist_ok=True)
    
    # Sauvegarder les fichiers
    base_name = file_path.stem
    clean_name = base_name.replace("_pink_snr5dB", "_clean")
    
    # Sauvegarder l'audio bruité
    output_pink = output_dir / f"{base_name}.wav"
    wavfile.write(str(output_pink), sr, (audio * 32767).astype(np.int16))
    print(f"✅ Sauvegardé: {output_pink}")
    
    # Sauvegarder l'audio filtré
    output_wiener = output_dir / f"{base_name}_wiener.wav"
    wavfile.write(str(output_wiener), sr, (wiener_audio * 32767).astype(np.int16))
    print(f"✅ Sauvegardé: {output_wiener}")
    
    # Si tu as l'audio clean original, le sauvegarder aussi
    # (à adapter selon ta structure de données)
    print("\n💡 Pour obtenir l'audio clean, extrais-le depuis LibriSpeech test-clean")

if __name__ == "__main__":
    main()