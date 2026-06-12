#!/usr/bin/env python3
"""
scripts/download_real_librispeech.py (v3 - FLAC support)
Télécharge de VRAIS fichiers audio LibriSpeech test-clean depuis OpenSLR.
Gère les fichiers .flac et les convertit en .wav 16kHz.
Compatible Windows PowerShell.
"""

import argparse
import json
import logging
import os
import sys
import tarfile
import tempfile
from pathlib import Path
import soundfile as sf

import librosa
import requests
from tqdm import tqdm

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# URL OpenSLR pour LibriSpeech test-clean
LIBRISPEECH_URL = "http://www.openslr.org/resources/12/test-clean.tar.gz"

def download_file(url, dest_path):
    """Télécharge un fichier avec barre de progression."""
    logger.info(f"Téléchargement depuis {url}...")
    
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f, tqdm(
            desc=dest_path.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        
        logger.info(f"✓ Téléchargement terminé: {dest_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur de téléchargement: {e}")
        return False

def extract_and_process(tar_path, output_dir, num_files):
    """Extrait l'archive FLAC et convertit en WAV 16kHz."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📦 Ouverture de l'archive: {tar_path}")
    
    metadata = []
    files_processed = 0
    
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            all_members = tar.getmembers()
            logger.info(f"🔍 {len(all_members)} membres totaux dans l'archive")
            
            # 🔧 CORRECTION: Chercher les fichiers .flac (pas .wav)
            flac_members = [m for m in all_members if m.isfile() and m.name.lower().endswith('.flac')]
            logger.info(f"🎵 {len(flac_members)} fichiers FLAC trouvés")
            
            if not flac_members:
                sample_names = [m.name for m in all_members[:10]]
                logger.warning(f"⚠️  Exemples: {sample_names}")
                return []
            
            selected_flacs = flac_members[:num_files]
            logger.info(f"📋 Sélection de {len(selected_flacs)} fichiers à convertir")
            
            # Pré-charger les transcriptions
            transcriptions = {}
            for member in all_members:
                if member.isfile() and member.name.endswith('.trans.txt'):
                    txt_obj = tar.extractfile(member)
                    if txt_obj:
                        content = txt_obj.read().decode('utf-8', errors='ignore')
                        for line in content.strip().split('\n'):
                            if ' ' in line:
                                key, text = line.split(' ', 1)
                                transcriptions[key] = text
            
            logger.info(f"📚 {len(transcriptions)} transcriptions chargées")
            
            for i, flac_member in enumerate(selected_flacs):
                try:
                    # Extraire le FLAC en mémoire
                    flac_obj = tar.extractfile(flac_member)
                    if not flac_obj:
                        continue
                    
                    # Nom du fichier de sortie (.wav)
                    flac_filename = Path(flac_member.name).name
                    wav_filename = flac_filename.replace('.flac', '.wav')
                    wav_filepath = output_path / wav_filename
                    
                    # Lire le FLAC en mémoire et convertir via librosa
                    flac_data = flac_obj.read()
                    
                    # Écrire temporairement pour librosa
                    import io
                    y, sr = librosa.load(io.BytesIO(flac_data), sr=16000)
                    
                    # Sauvegarder en WAV 16kHz
                    sf.write(str(wav_filepath), y, sr)
                    
                    # Extraire les IDs pour la transcription
                    # Format: .../speaker/chapter/speaker-chapter-id.flac
                    stem = Path(flac_member.name).stem  # sans extension
                    parts = stem.split('-')
                    if len(parts) >= 3:
                        file_key = '-'.join(parts[:3])  # speaker-chapter-id
                        transcription = transcriptions.get(file_key, "")
                        speaker_id = parts[0]
                        chapter_id = parts[1]
                    else:
                        transcription = ""
                        speaker_id = "unknown"
                        chapter_id = "unknown"
                    
                    duration = len(y) / sr
                    
                    metadata.append({
                        "file_path": str(wav_filepath),
                        "duration": round(duration, 2),
                        "sample_rate": 16000,
                        "language": "en",
                        "transcription": transcription,
                        "noise_type": "clean_speech",
                        "speaker_id": speaker_id,
                        "chapter_id": chapter_id,
                        "original_format": "flac"
                    })
                    
                    files_processed += 1
                    logger.info(f"  [{files_processed}/{len(selected_flacs)}] {wav_filename} - {duration:.1f}s")
                    
                except Exception as e:
                    logger.warning(f"⚠️  Erreur sur {flac_member.name}: {e}")
                    continue
        
        # Sauvegarder metadata
        metadata_path = Path(output_dir).parent / "librispeech_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ {files_processed} fichiers convertis dans {output_dir}")
        logger.info(f"📄 Métadonnées: {metadata_path}")
        
        return metadata
        
    except Exception as e:
        logger.error(f"❌ Erreur d'extraction: {e}", exc_info=True)
        return []

def main():
    parser = argparse.ArgumentParser(description="Download LibriSpeech test-clean (FLAC→WAV)")
    parser.add_argument("--num_files", type=int, default=20, help="Number of audio files")
    parser.add_argument("--output_dir", type=str, default="data/raw_librispeech", help="Output directory")
    parser.add_argument("--skip_download", action="store_true", help="Skip download if archive exists")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("🎙️  LIBRISPEECH DOWNLOADER (v3 - FLAC→WAV)")
    logger.info("=" * 60)
    logger.info(f"  • Output: {args.output_dir}")
    logger.info(f"  • Files: {args.num_files}")
    logger.info(f"  • Source: OpenSLR (FLAC format)")
    logger.info("-" * 60)
    
    temp_dir = Path(tempfile.gettempdir())
    tar_path = temp_dir / "test-clean.tar.gz"
    
    try:
        if not args.skip_download or not tar_path.exists():
            if not download_file(LIBRISPEECH_URL, tar_path):
                return 1
        else:
            logger.info(f"⏭️  Archive en cache: {tar_path}")
        
        metadata = extract_and_process(tar_path, args.output_dir, args.num_files)
        
        if metadata:
            logger.info("=" * 60)
            logger.info("✅ TÉLÉCHARGEMENT TERMINÉ")
            logger.info("=" * 60)
            logger.info(f"📁 Fichiers WAV créés: {len(metadata)}")
            logger.info(f"📊 Prochaine étape: experiments/baseline_wer.py")
            return 0
        else:
            logger.error("❌ Aucun fichier traité")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interruption utilisateur")
        return 1
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())