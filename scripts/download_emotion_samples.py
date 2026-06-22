#!/usr/bin/env python3
"""
scripts/download_emotion_samples.py
Downloads Actor 01 from RAVDESS Zenodo, extracts it, and prepares 24 samples 
for speech emotion recognition and sarcasm detection.
"""

import json
import logging
import os
import zipfile
import urllib.request
from pathlib import Path
import soundfile as sf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
EMOTIONS = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry"
}
STATEMENTS = {
    "01": "Kids are talking by the door.",
    "02": "Dogs are sitting by the door."
}

def download_and_extract(actors_list=["01", "02", "03", "04", "05", "06"]):
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = data_dir / "Audio_Speech_Actors_01-24.zip"
    raw_dir = data_dir / "raw_emotion"
    samples_dir = data_dir / "emotion_samples"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    if not zip_path.exists():
        logger.info(f"Downloading Audio-only Speech dataset from Zenodo: {URL}")
        urllib.request.urlretrieve(URL, zip_path)
        logger.info("Download completed.")
    else:
        logger.info("Audio_Speech_Actors_01-24.zip already exists.")
        
    # Ensure all actor strings are zero-padded to two digits (e.g., "1" -> "01")
    actors_formatted = []
    for actor in actors_list:
        try:
            actors_formatted.append(f"{int(actor):02d}")
        except ValueError:
            # Fallback if it's not a simple integer string
            actors_formatted.append(actor)
            
    actors_to_extract = [f"Actor_{actor}" for actor in actors_formatted]
    logger.info(f"Extracting {', '.join(actors_to_extract)} files from zip file...")
    
    # Optional: clean out old files in raw_dir and samples_dir to avoid mixing
    for f in raw_dir.glob("*.wav"):
        try:
            f.unlink()
        except Exception:
            pass
    for f in samples_dir.glob("*.wav"):
        try:
            f.unlink()
        except Exception:
            pass

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            # Avoid substring match (e.g. Actor_1 matching Actor_10) by checking the path parts
            member_parts = Path(member).parts
            if len(member_parts) >= 2 and member_parts[0] in actors_to_extract and member.endswith(".wav"):
                # Extract file flatly into raw_dir
                filename = os.path.basename(member)
                source = zip_ref.open(member)
                target = open(raw_dir / filename, "wb")
                with source, target:
                    target.write(source.read())
                    
    logger.info("Extraction completed.")
    
    # Process files
    metadata = []
    # Find all wav files
    wav_files = list(raw_dir.glob("*.wav"))
    logger.info(f"Found {len(wav_files)} raw files in extracted folder.")
    
    selected_files = []
    
    # Sort files to be deterministic
    wav_files.sort()
    
    for f in wav_files:
        parts = f.stem.split("-")
        if len(parts) < 7:
            continue
            
        emotion_idx = parts[2]
        statement_idx = parts[4]
        
        if emotion_idx in EMOTIONS:
            # Destination path
            dest_path = samples_dir / f.name
            
            # Copy/write to destination (ensure 16kHz mono)
            data, sr = sf.read(str(f))
            if len(data.shape) > 1:
                data = data.mean(axis=1)
                
            # If rate is different, resample. RAVDESS is 48kHz by default, we need 16kHz for ASR/SER
            if sr != 16000:
                import librosa
                data = librosa.resample(data, orig_sr=sr, target_sr=16000)
                sr = 16000
                
            sf.write(str(dest_path), data, sr)
            
            metadata.append({
                "file_path": str(dest_path).replace("\\", "/"),
                "file_name": f.name,
                "emotion_id": int(emotion_idx),
                "emotion": EMOTIONS[emotion_idx],
                "transcription": STATEMENTS[statement_idx],
                "duration": round(len(data) / sr, 2)
            })
            selected_files.append(f.name)
            
    # Save metadata JSON
    meta_path = data_dir / "emotion_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        
    logger.info(f"Successfully selected {len(metadata)} emotional samples and saved metadata to {meta_path}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download and extract RAVDESS emotional speech samples.")
    parser.file_name = "download_emotion_samples.py"
    parser.add_argument(
        "--actors", 
        type=str, 
        default="01,02,03,04,05,06",
        help="Comma-separated list of actors to extract (e.g., '01,02,03' or 'all')"
    )
    args = parser.parse_args()
    
    if args.actors.lower() == "all":
        actors_list = [f"{i:02d}" for i in range(1, 25)]
    else:
        actors_list = [a.strip() for a in args.actors.split(",") if a.strip()]
        
    download_and_extract(actors_list)

