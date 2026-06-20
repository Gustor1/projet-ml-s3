# 🎭 Integrating Emotion & Sarcasm Detection ("Fun" Extension)

This document describes how to integrate Speech Emotion Recognition (SER) and sarcasm analysis into our audio preprocessing and ASR project (Topic 3).

---

## 💡 The "Fun & Fancy" Concept

The goal is to enrich our baseline pipeline:
1. **Input**: Noisy user audio.
2. **Preprocessing**: Cleaned audio (Wiener, etc.).
3. **ASR**: Text transcription (Whisper/Wav2Vec2).
4. **SER (New)**: Emotion detection in the voice (Anger, Happiness, Sadness, etc.).
5. **Sentiment Analysis (New)**: Analysis of the transcribed text's sentiment.
6. **Sarcasm Detector (New)**: Alert if the voice emotion does not match the text sentiment (e.g., positive text spoken in a very angry tone).

---

## 📊 Data Engineer Task (Scientific Angle)

As a Data Engineer, the goal is to analyze the impact of our preprocessing filters on non-verbal audio (emotion).

### Steps to follow:
1. **Download test samples**: Use an emotion dataset like **RAVDESS** (for example, 10 angry, 10 happy, 10 sad files).
2. **Apply noise**: Use our existing scripts in `scripts/` to add noise (white, pink, urban, babble) to these files at different SNR levels (20dB, 10dB, 5dB).
3. **Apply preprocessing**: Clean these noisy files with the Wiener filter and spectral subtraction.
4. **Calculate SER accuracy**: Compare the correct emotion classification rate on:
   - Clean raw audio (baseline).
   - Noisy audio.
   - Cleaned audio using the Wiener filter.
   - Cleaned audio using spectral subtraction.
5. **Document the results**: Add conclusions and a comparative chart in `docs/insights.md`.

---

## 🛠️ Technical Implementation Guide

### Step 1: Required Dependencies
Verify that the following libraries are installed in the virtual environment:
```bash
pip install transformers torch librosa
```

### Step 2: Speech Emotion Recognition Code
Here is a simple implementation using a Wav2Vec2 model trained for emotion (`superb/wav2vec2-base-superb-er`):

```python
import torch
import librosa
from transformers import pipeline

# Initialize the audio emotion classification pipeline
# Detected emotions: neutral, happy, sad, angry
emotion_classifier = pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

def get_voice_emotion(audio_path: str) -> dict:
    """
    Analyzes an audio file to extract its main emotion.
    """
    predictions = emotion_classifier(audio_path)
    # predictions looks like: [{'label': 'angry', 'score': 0.85}, {'label': 'happy', 'score': 0.05}, ...]
    dominant_emotion = predictions[0]
    return {
        "emotion": dominant_emotion["label"],
        "confidence": round(dominant_emotion["score"], 4),
        "all_scores": predictions
    }

# Example usage:
# result = get_voice_emotion("path/to/my_audio.wav")
# print(f"Detected emotion: {result['emotion']} ({result['confidence']:.2%})")
```

### Step 3: Sarcasm Detector (Text-Sentiment vs Voice-Emotion Logic)
To compare text and voice, we can use a lightweight text sentiment classifier:

```python
# Initialize the text sentiment pipeline
text_sentiment_classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def detect_sarcasm(text: str, voice_emotion: str) -> dict:
    """
    Compares text sentiment and voice emotion to identify sarcasm.
    """
    if not text.strip():
        return {"sarcastic": False, "reason": "No speech detected"}

    text_res = text_sentiment_classifier(text)[0]
    text_sentiment = text_res["label"].lower()  # 'positive' or 'negative'
    
    is_sarcastic = False
    reason = "Normal speech alignment"
    
    # Simple alignment mismatch rules
    if text_sentiment == "positive" and voice_emotion in ["angry", "sad"]:
        is_sarcastic = True
        reason = f"Positive words spoken with a negative voice ({voice_emotion})"
    elif text_sentiment == "negative" and voice_emotion == "happy":
        is_sarcastic = True
        reason = "Negative words spoken with a happy voice"
        
    return {
        "is_sarcastic": is_sarcastic,
        "reason": reason,
        "text_sentiment": text_sentiment,
        "voice_emotion": voice_emotion
    }
```

---

## 🚀 Demo Integration Plan (Streamlit / Gradio)

In the demo script `demo/app.py` (to be implemented by the demo engineer):
1. Provide an audio recorder.
2. Run the audio preprocessing chosen by the user.
3. Display the Whisper transcription.
4. Display the detected voice emotion with large emojis:
   - 😡 `angry`
   - 😄 `happy`
   - 😢 `sad`
   - 😐 `neutral`
5. Display a flash banner if sarcasm is detected.
