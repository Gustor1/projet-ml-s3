#!/usr/bin/env python3
"""
scripts/extract_audio_samples.py
Extrait des échantillons audio pour les spectrogrammes.
"""
import json
import numpy as np
from scipy.io import wavfile
from scipy.signal import wiener
from pathlib import Path
import soundfile as sf

def main():
    # Charger les métadonnées pour trouver un fichier
    metadata_path = Path("data/augmented_pink_metadata.json")
    
    if not metadata_path.exists():
        print("❌ Metadata file not found. Trying babble metadata...")
        metadata_path = Path("data/augmented_babble_metadata.json")
    
    if not metadata_path.exists():
        print("❌ No metadata files found. Looking for any WAV files...")
        # Chercher n'importe quel fichier WAV dans results/ ou data/
        wav_files = list(Path(".").rglob("*.wav"))
        if wav_files:
            print(f"✅ Found {len(wav_files)} WAV files. Using the first one.")
            # Prendre le premier fichier trouvé
            sample_file = wav_files[0]
            print(f"Using: {sample_file}")
            
            # Charger l'audio
            audio, sr = sf.read(str(sample_file), dtype="float32")
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Créer le dossier de sortie
            output_dir = Path("data/audio_samples")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarder l'audio original
            output_clean = output_dir / "sample_clean.wav"
            wavfile.write(str(output_clean), sr, (audio * 32767).astype(np.int16))
            print(f"✅ Sauvegardé: {output_clean}")
            
            # Appliquer Wiener et sauvegarder
            wiener_audio = wiener(audio).astype(np.float32)
            output_wiener = output_dir / "sample_wiener.wav"
            wavfile.write(str(output_wiener), sr, (wiener_audio * 32767).astype(np.int16))
            print(f"✅ Sauvegardé: {output_wiener}")
            
            print("\n💡 Maintenant, lance: python scripts/generate_spectrograms.py")
            return
        else:
            print("❌ No WAV files found anywhere. You need to run your augmentation scripts first.")
            return
    
    # Charger les métadonnées
    with open(metadata_path, "r", encoding="utf-8") as f:
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
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        print("💡 You need to run your augmentation scripts first (augment_pink_noise.py or similar)")
        return
    
    # Charger l'audio
    audio, sr = sf.read(str(file_path), dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Créer le dossier de sortie
    output_dir = Path("data/audio_samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder l'audio bruité
    base_name = file_path.stem
    output_pink = output_dir / f"{base_name}.wav"
    wavfile.write(str(output_pink), sr, (audio * 32767).astype(np.int16))
    print(f"✅ Sauvegardé: {output_pink}")
    
    # Appliquer le filtre de Wiener
    print("🔧 Application du filtre de Wiener...")
    wiener_audio = wiener(audio).astype(np.float32)
    
    # Sauvegarder l'audio filtré
    output_wiener = output_dir / f"{base_name}_wiener.wav"
    wavfile.write(str(output_wiener), sr, (wiener_audio * 32767).astype(np.int16))
    print(f"✅ Sauvegardé: {output_wiener}")
    
    # Essayer de trouver l'audio clean original (optionnel)
    # Chercher dans les métadonnées un fichier correspondant sans bruit
    base_id = base_name.replace("_pink_snr5dB", "").replace("_babble_snr5dB", "")
    clean_candidates = [m for m in meta if base_id in m["file_path"]]
    
    print(f"\n💡 Pour obtenir l'audio clean, cherche dans ton dossier LibriSpeech original")
    print(f"   ou utilise un fichier de référence comme {base_id}")
    print(f"\n✅ Maintenant, modifie generate_spectrograms.py pour utiliser:")
    print(f"   clean_file = 'data/audio_samples/{base_name}.wav'")
    print(f"   pink_noisy_file = 'data/audio_samples/{base_name}.wav'")
    print(f"   wiener_filtered_file = 'data/audio_samples/{base_name}_wiener.wav'")

if __name__ == "__main__":
    main()