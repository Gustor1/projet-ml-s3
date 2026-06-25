# 🎭 Developer Guide: Integrating Speech Emotion Recognition & Sarcasm Detection (Affective Extension)

This document provides a technical integration guide for incorporating Speech Emotion Recognition (SER) and multimodal sarcasm detection into the audio preprocessing and ASR pipeline (Topic 3).

---

## 💡 System Concept & Multimodal Mismatch
The goal of this extension is to enrich the baseline ASR system with affective computing capabilities, creating a joint verbal and non-verbal analysis pipeline:

1. **Acoustic Input**: Noisy user voice recordings.
2. **Parallel Signal Routing**: Denoised audio stream (Wiener) is routed to ASR, while the original audio (silence-trimmed and peak-normalized) is routed to SER.
3. **Verbal Transcription (ASR)**: `openai/whisper-tiny` extracts the verbal text.
4. **Speech Emotion Recognition (SER)**: `superb/wav2vec2-base-superb-er` classifies vocal emotion from acoustic features (prosody, intensity, harmonics).
5. **Text Sentiment Analysis (NLP)**: `distilbert-base-uncased-finetuned-sst-2-english` classifies the literal semantic sentiment of the transcribed text.
6. **Sarcasm Detection**: Identifies semantic mismatch between verbal sentiment and vocal tone (e.g., positive words spoken in an angry, aggressive tone).

---

## 🛠️ Software Architecture & API Implementation

### 1. Dependencies and Environment Setup
Verify that the virtual environment includes PyTorch and the Hugging Face Transformers library:
```bash
pip install transformers torch librosa scipy
```

### 2. Speech Emotion Classifier Wrapper
We utilize a Wav2Vec2 model pre-trained via self-supervised learning on raw speech and fine-tuned for emotion recognition on the IEMOCAP corpus. The classifier expects a $16\text{ kHz}$ mono float32 array:

```python
import torch
import librosa
import numpy as np
from transformers import pipeline

# Initialize the self-supervised audio emotion classification pipeline
# Classifier labels: neutral, happy, sad, angry
emotion_classifier = pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er",
    device=0 if torch.cuda.is_available() else -1
)

def extract_vocal_emotion(audio_array: np.ndarray, sample_rate: int = 16000) -> dict:
    """
    Extracts the dominant emotional state from a raw audio waveform.
    
    Parameters:
        audio_array (np.ndarray): Float32 audio signal amplitude array.
        sample_rate (int): Sampling frequency (default: 16000 Hz).
        
    Returns:
        dict: dominant label, confidence score, and raw probability distribution.
    """
    # Ensure signal is sampled at 16 kHz
    if sample_rate != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
    
    # Run inference directly on the in-memory NumPy array
    predictions = emotion_classifier(audio_array)
    dominant_emotion = predictions[0]
    
    return {
        "emotion": dominant_emotion["label"],
        "confidence": round(dominant_emotion["score"], 4),
        "scores": predictions
    }
```

### 3. Sarcasm Detection & Multimodal Fusion Calibration
The sarcasm engine compares the semantic sentiment and vocal emotion, applying YIN-based pitch calibration to resolve Joy/Anger class ambiguities caused by microphone proximity clipping:

```python
# Initialize the distilled semantic sentiment pipeline
sentiment_classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def estimate_pitch_yin(audio_array: np.ndarray, sample_rate: int = 16000) -> float:
    """
    Estimates the mean fundamental frequency (F0) using the YIN algorithm.
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio_array,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sample_rate
    )
    # Remove NaN values from unvoiced segments
    valid_f0 = f0[~np.isnan(f0)]
    return float(np.mean(valid_f0)) if len(valid_f0) > 0 else 0.0

def detect_sarcasm_multimodal(text: str, audio_array: np.ndarray, raw_ser_result: dict) -> dict:
    """
    Applies multimodal calibration and cross-modal analysis to detect sarcasm.
    """
    if not text.strip():
        return {"is_sarcastic": False, "reason": "No speech detected"}

    # 1. Semantic Sentiment Classification
    text_res = sentiment_classifier(text)[0]
    text_sentiment = text_res["label"].lower()  # 'positive' or 'negative'
    
    # 2. Extract Pitch Trajectory (F0)
    mean_f0 = estimate_pitch_yin(audio_array)
    
    # 3. Apply Multimodal Fusion Calibration
    calibrated_emotion = raw_ser_result["emotion"]
    
    # Calibration Rule: High pitch + positive text corrects false positive Anger to Happy
    if text_sentiment == "positive":
        if raw_ser_result["emotion"] == "angry" and mean_f0 > 180.0:
            calibrated_emotion = "happy"
            
    # 4. Cross-Modal Mismatch Rules
    is_sarcastic = False
    reason = "Acoustic and semantic alignment"
    
    if text_sentiment == "positive" and calibrated_emotion in ["angry", "sad"]:
        is_sarcastic = True
        reason = f"Positive words spoken with a negative voice ({calibrated_emotion})"
    elif text_sentiment == "negative" and calibrated_emotion == "happy":
        is_sarcastic = True
        reason = "Negative words spoken with a happy voice"
        
    return {
        "is_sarcastic": is_sarcastic,
        "reason": reason,
        "text_sentiment": text_sentiment,
        "raw_emotion": raw_ser_result["emotion"],
        "calibrated_emotion": calibrated_emotion,
        "mean_f0_hz": round(mean_f0, 2)
    }
```

---

## 🚀 Streamlit Demo Integration Plan
In the interactive dashboard `demo/app.py` (implemented by the Demo Engineer), the pipeline should be structured as follows:

1. **Acoustic Capture**: Streamlit microphone recorder component returning raw float32 NumPy arrays.
2. **DSP Preprocessing Live Switch**: Toggles between `none`, `wiener`, and `spectral_subtraction`.
3. **Parallel Routing Engine**: Routes Wiener-filtered audio to Whisper-tiny, and peak-normalized/silence-trimmed raw audio to Wav2Vec2 SER.
4. **Visualization Layer**:
   - Displays Whisper transcription.
   - Plots spectrogram with YIN pitch ($F_0$) trajectory overlays.
   - Renders calibrated vocal emotion with animated indicators (😡 `angry`, 😄 `happy`, 😢 `sad`, 😐 `neutral`).
   - Renders a warning banner when sarcasm is detected ($is\_sarcastic = True$).
