#!/usr/bin/env python3
"""
experiments/sarcasm_detector.py
Runs a pipeline combining:
1. Whisper ASR (Speech-to-Text)
2. Wav2Vec2 SER (Speech Emotion Recognition)
3. DistilBERT Sentiment Analysis (Text Sentiment Classifier)
To diagnose mismatching emotional states (sarcasm or passive-aggressiveness).
"""

import argparse
import logging
from pathlib import Path
import soundfile as sf
import torch
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Map superb labels to display names
SER_LABELS = {
    "neu": "neutral",
    "hap": "happy",
    "ang": "angry",
    "sad": "sad"
}

def analyze_sarcasm(audio_path, model_size="tiny"):
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading pipelines on device: {device}...")
    
    # Load ASR pipeline
    asr_pipe = pipeline("automatic-speech-recognition", model=f"openai/whisper-{model_size}", device=device)
    
    # Load SER pipeline
    ser_pipe = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", device=device)
    
    # Load Sentiment pipeline
    sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)
    
    logger.info("Processing audio...")
    # Load audio to array
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
        
    # Step 1: Speech-to-Text
    logger.info("Step 1: Running Whisper ASR...")
    # Whisper expects 16kHz audio
    asr_res = asr_pipe(audio, generate_kwargs={"language": "en"})
    transcription = asr_res["text"].strip()
    
    # Step 2: Speech Emotion Recognition
    logger.info("Step 2: Running Wav2Vec2 SER...")
    ser_res = ser_pipe(audio)
    raw_emotion = ser_res[0]["label"]
    voice_emotion = SER_LABELS.get(raw_emotion, raw_emotion)
    ser_confidence = ser_res[0]["score"]
    
    # Step 3: Text Sentiment Analysis
    logger.info("Step 3: Running Sentiment Analysis on transcription...")
    if transcription:
        sent_res = sentiment_pipe(transcription)[0]
        text_sentiment = sent_res["label"].lower() # 'positive' or 'negative'
        sent_confidence = sent_res["score"]
    else:
        text_sentiment = "neutral"
        sent_confidence = 1.0
        
    # Step 4: Sarcasm / Mismatch detection logic
    is_sarcastic = False
    reason = "Normal vocal alignment."
    
    # A positive statement spoken with angry or sad voice
    if text_sentiment == "positive" and voice_emotion in ["angry", "sad"]:
        is_sarcastic = True
        reason = f"Positive words spoken with a negative voice ({voice_emotion})."
    # A negative statement spoken with happy voice
    elif text_sentiment == "negative" and voice_emotion == "happy":
        is_sarcastic = True
        reason = "Negative words spoken with a happy/excited voice."
    # A neutral statement said with high emotional charge
    elif text_sentiment == "neutral" and voice_emotion in ["angry", "happy", "sad"]:
        # In RAVDESS statements are neutral ("Kids are talking by the door.")
        # but actors express high emotion. This is a form of acted/passive-aggressive stress.
        is_sarcastic = True
        reason = f"Neutral statement spoken with an emotional voice ({voice_emotion})."

    # Display results
    print("\n" + "="*60)
    print("=== PASSIVE-AGGRESSIVE / SARCASM DETECTION REPORT ===")
    print("="*60)
    print(f"File Name      : {audio_path.name}")
    print(f"Transcription  : \"{transcription}\"")
    print(f"Text Sentiment : {text_sentiment.upper()} (conf: {sent_confidence:.1%})")
    print(f"Vocal Emotion  : {voice_emotion.upper()} (conf: {ser_confidence:.1%})")
    print("-"*60)
    if is_sarcastic:
        print(f"RESULT         : SARCASM DETECTED! [!] ")
        print(f"Reason         : {reason}")
    else:
        print(f"RESULT         : Normal Speech. [-]")
        print(f"Reason         : {reason}")
    print("="*60 + "\n")
    
    return {
        "file_name": audio_path.name,
        "transcription": transcription,
        "text_sentiment": text_sentiment,
        "voice_emotion": voice_emotion,
        "is_sarcastic": is_sarcastic,
        "reason": reason
    }

def main():
    parser = argparse.ArgumentParser(description="ASR + SER Sarcasm Detector")
    parser.add_argument(
        "--audio", 
        type=str, 
        default="data/emotion_samples/03-01-05-02-01-01-01.wav",
        help="Path to WAV audio file (16kHz mono)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tiny",
        choices=["tiny", "base"],
        help="Whisper model size"
    )
    args = parser.parse_args()
    analyze_sarcasm(args.audio, args.model)

if __name__ == "__main__":
    main()
