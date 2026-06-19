import streamlit as st
import torch
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os
import json
import io
import time
from pathlib import Path
from scipy.signal import wiener
from transformers import pipeline

# Configure Page
st.set_page_config(
    page_title="SentiVoice & Sarcasm Detector",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphism Styles
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0b0813 0%, #110c22 50%, #1a1233 100%);
        color: #f7fafc;
        font-family: 'Outfit', sans-serif;
    }
    
    h1 {
        font-family: 'Outfit', sans-serif;
        font-weight: 800 !important;
        background: linear-gradient(to right, #a78bfa, #f43f5e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 4px 12px rgba(167, 139, 250, 0.2);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
        margin-bottom: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(167, 139, 250, 0.3);
    }
    
    .sarcasm-banner {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.25) 0%, rgba(220, 38, 38, 0.1) 100%);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.4);
        margin: 20px 0;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 10px rgba(239, 68, 68, 0.3); }
        50% { box-shadow: 0 0 25px rgba(239, 68, 68, 0.6); }
        100% { box-shadow: 0 0 10px rgba(239, 68, 68, 0.3); }
    }
    
    .sarcasm-title {
        color: #fca5a5;
        font-size: 26px;
        font-weight: 800;
        margin-bottom: 5px;
    }
    
    .sarcasm-reason {
        color: #fee2e2;
        font-size: 17px;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

# Preprocessing Functions
def preprocess_none(audio):
    return audio

def preprocess_wiener(audio, mysize=3):
    # Wiener filter requires odd window size
    if mysize % 2 == 0:
        mysize += 1
    return wiener(audio, mysize=mysize).astype(np.float32)

def preprocess_spectral_subtraction(audio, sr, alpha=2.0, beta=0.01):
    n = len(audio)
    nfft = 2048
    hop = nfft // 2
    noise_len = int(min(0.5 * sr, n))
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

# Trimming Silent Margins to avoid skewing standardization
def trim_silence(y, top_db=30):
    try:
        intervals = librosa.effects.split(y, top_db=top_db)
        if len(intervals) > 0:
            start = intervals[0][0]
            end = intervals[-1][1]
            return y[start:end]
    except Exception:
        pass
    return y

# Peak Normalization to limit proximity distortion / clipping
def normalize_volume(y):
    max_val = np.max(np.abs(y))
    if max_val > 0:
        return y / max_val
    return y

# Multimodal Fusion Calibration Heuristic
def fuse_modalities(emotion_preds, text_sentiment, sent_score, mean_pitch):
    """
    Calibrates / adjusts Wav2Vec2 emotion prediction scores based on text sentiment and pitch.
    This corrects common SER errors (like mistaking loud happy speech for anger).
    """
    scores = {p["label"]: p["score"] for p in emotion_preds}
    
    # Ensure all four standard labels exist
    for k in ["neu", "hap", "ang", "sad"]:
        if k not in scores:
            scores[k] = 0.0
            
    # 1. Text Sentiment Calibration:
    if text_sentiment == "positive" and sent_score > 0.6:
        # If words are positive, boost Happy and Neutral, penalize Angry and Sad
        boost = 0.25 * sent_score
        scores["hap"] += boost
        scores["neu"] += boost * 0.5
        scores["ang"] -= boost * 0.7
        scores["sad"] -= boost * 0.7
        
    elif text_sentiment == "negative" and sent_score > 0.6:
        # If words are negative, boost Angry and Sad, penalize Happy
        boost = 0.20 * sent_score
        scores["ang"] += boost * 0.6
        scores["sad"] += boost * 0.6
        scores["hap"] -= boost * 0.8
        
    # 2. Pitch-based calibration:
    if mean_pitch > 180.0:
        # High pitch indicates high arousal (Happy or Angry). Sad is extremely unlikely.
        scores["sad"] -= 0.15
        # If it was neutral, high pitch pushes it towards happy/angry
        scores["neu"] -= 0.10
        # If text is positive, high pitch points strongly to happy, not angry
        if text_sentiment == "positive":
            scores["hap"] += 0.15
            scores["ang"] -= 0.10
            
    elif mean_pitch > 0.0 and mean_pitch < 130.0:
        # Low pitch indicates low arousal (Sad or Neutral)
        scores["hap"] -= 0.15
        scores["ang"] -= 0.15
        scores["sad"] += 0.10
        scores["neu"] += 0.10
        
    # Bound scores to >= 0
    for k in scores:
        scores[k] = max(scores[k], 0.0)
        
    # Re-normalize to sum to 1.0
    total = sum(scores.values())
    if total > 0:
        for k in scores:
            scores[k] /= total
    else:
        scores = {"neu": 1.0, "hap": 0.0, "ang": 0.0, "sad": 0.0}
        
    # Reconstruct the sorted list of predictions
    calibrated_preds = [
        {"label": k, "score": scores[k]}
        for k in sorted(scores, key=scores.get, reverse=True)
    ]
    return calibrated_preds

# SNR Estimation Helper
def estimate_snr(audio, sr=16000):
    try:
        noise_len = int(min(0.5 * sr, len(audio)))
        if noise_len <= 0:
            return 0.0
        
        noise_power = np.mean(audio[:noise_len] ** 2)
        signal_power = np.mean(audio[noise_len:] ** 2) if len(audio) > noise_len else np.mean(audio ** 2)
        
        if noise_power < 1e-10:
            noise_power = 1e-10
        if signal_power < 1e-10:
            signal_power = 1e-10
            
        snr = 10 * np.log10(signal_power / noise_power)
        return float(snr)
    except Exception:
        return 0.0

# Pitch Estimation Helper using YIN
def estimate_pitch_contour(audio, sr=16000):
    try:
        # Avoid long processing times by capping pitch analysis to first 10 seconds
        y_pitch = audio
        if len(y_pitch) > 10 * sr:
            y_pitch = y_pitch[:10*sr]
            
        # Human fundamental frequency (F0) is typically 75Hz - 400Hz
        f0 = librosa.yin(y_pitch, fmin=75, fmax=400, sr=sr)
        
        # Filter out NaN/Inf and out of bounds
        valid_f0 = f0[(f0 >= 75) & (f0 <= 400) & (~np.isnan(f0)) & (~np.isinf(f0))]
        
        if len(valid_f0) > 0:
            return float(np.mean(valid_f0)), float(np.std(valid_f0)), f0
        else:
            return 0.0, 0.0, f0
    except Exception:
        return 0.0, 0.0, np.array([])

# Load Models with Caching
@st.cache_resource
def load_models():
    device = 0 if torch.cuda.is_available() else -1
    with st.spinner("🚀 Loading neural models into memory... This may take a minute on first load."):
        asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=device)
        sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)
        emotion = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", device=device)
    return asr, sentiment, emotion

# Helper to generate WAV bytes for browser player
def get_wav_bytes(y, sr):
    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format='WAV')
    return buffer.getvalue()

# Mapping for Emotions
EMOJI_MAP = {
    "neu": ("😐", "Neutral", "#a0aec0"),
    "hap": ("😄", "Happy", "#48bb78"),
    "sad": ("😢", "Sad", "#4299e1"),
    "ang": ("😡", "Angry", "#f56565")
}

# Main Application
def main():
    inject_custom_css()
    
    st.title("🎭 SentiVoice & Sarcasm Detector")
    st.write("An interactive multi-modal dashboard evaluating the impact of local audio preprocessing on verbal and non-verbal speech metrics.")
    
    # Load models
    asr, sentiment, emotion = load_models()
    
    # Absolute paths
    base_dir = Path(__file__).resolve().parent.parent
    metadata_path = base_dir / "data" / "emotion_metadata.json"
    
    # Sidebar config
    st.sidebar.header("🎛️ Audio Source Configuration")
    source_type = st.sidebar.radio(
        "Select Audio Source:",
        ["RAVDESS Dataset Sample", "Record Your Voice (Live)", "Upload WAV File"]
    )
    
    # Load RAVDESS list
    metadata = []
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
    audio_path = None
    y_raw, sr = None, 16000
    
    if source_type == "RAVDESS Dataset Sample":
        if not metadata:
            st.sidebar.error("RAVDESS metadata not found. Please run scripts/download_emotion_samples.py.")
        else:
            sample_options = {
                f"{item['emotion'].upper()} | Statement: '{item['transcription']}' ({item['file_name']})": item
                for item in metadata
            }
            selected_label = st.sidebar.selectbox("Choose a sample:", list(sample_options.keys()))
            selected_item = sample_options[selected_label]
            audio_path = base_dir / "data" / "emotion_samples" / selected_item["file_name"]
            if audio_path.exists():
                y_raw, sr = librosa.load(str(audio_path), sr=16000)
                st.sidebar.success(f"Loaded: {selected_item['file_name']}")
            else:
                st.sidebar.error("Audio sample file not found on disk.")
                
    elif source_type == "Record Your Voice (Live)":
        st.sidebar.info("🗣️ Speak in English for transcription and sentiment detection.")
        recorded_file = st.sidebar.audio_input("Record audio here:")
        if recorded_file is not None:
            y_raw, sr = librosa.load(recorded_file, sr=16000)
            st.sidebar.success("Voice successfully recorded!")
            
    elif source_type == "Upload WAV File":
        uploaded_file = st.sidebar.file_uploader("Upload WAV audio file:", type=["wav"])
        if uploaded_file is not None:
            y_raw, sr = librosa.load(uploaded_file, sr=16000)
            st.sidebar.success("File uploaded successfully!")
            
    # Preprocessing selection
    st.sidebar.header("🧹 Preprocessing Filter")
    preprocess_method = st.sidebar.selectbox(
        "Choose Preprocessing Filter:",
        ["None (Raw Audio)", "Wiener Filter", "Spectral Subtraction"]
    )
    
    # Preprocessing filter hyperparameters
    wiener_size = 3
    ss_alpha = 2.0
    ss_beta = 0.01
    
    if preprocess_method == "Wiener Filter":
        st.sidebar.subheader("Wiener Filter Settings")
        wiener_size = st.sidebar.slider(
            "Noise Filter Size (odd size):",
            min_value=3,
            max_value=15,
            value=3,
            step=2
        )
    elif preprocess_method == "Spectral Subtraction":
        st.sidebar.subheader("Spectral Subtraction Settings")
        ss_alpha = st.sidebar.slider(
            "Oversubtraction Factor (alpha):",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.1
        )
        ss_beta = st.sidebar.slider(
            "Spectral Floor (beta):",
            min_value=0.001,
            max_value=0.1,
            value=0.01,
            step=0.001,
            format="%.3f"
        )
    
    # Primary action button
    analyze_btn = st.sidebar.button("🔍 Run Full Pipeline", use_container_width=True)
    
    if y_raw is not None:
        # Preprocess
        if preprocess_method == "None (Raw Audio)":
            y_proc = preprocess_none(y_raw)
        elif preprocess_method == "Wiener Filter":
            y_proc = preprocess_wiener(y_raw, mysize=wiener_size)
        else:
            y_proc = preprocess_spectral_subtraction(y_raw, sr, alpha=ss_alpha, beta=ss_beta)
            
        # Audio Players Row
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔊 Raw Audio")
            st.audio(get_wav_bytes(y_raw, sr), format="audio/wav")
        with col2:
            st.subheader(f"🔊 Preprocessed Audio ({preprocess_method})")
            st.audio(get_wav_bytes(y_proc, sr), format="audio/wav")
            
        # Pitch tracking estimation
        with st.spinner("⏳ Extracting acoustic pitch contours..."):
            pitch_mean_raw, pitch_std_raw, f0_raw = estimate_pitch_contour(y_raw, sr)
            pitch_mean_proc, pitch_std_proc, f0_proc = estimate_pitch_contour(y_proc, sr)
            
        # Waveform & Spectrogram comparison
        st.subheader("📊 Signal Analysis (Waveform & Spectrogram with Pitch Overlay)")
        fig, axes = plt.subplots(2, 2, figsize=(14, 7))
        
        # Plot raw waveform
        t_raw = np.linspace(0, len(y_raw)/sr, len(y_raw))
        axes[0, 0].plot(t_raw, y_raw, color='#6366f1', alpha=0.8)
        axes[0, 0].set_title("Raw Waveform", color='#cbd5e1')
        axes[0, 0].set_ylabel("Amplitude", color='#cbd5e1')
        axes[0, 0].grid(True, alpha=0.1)
        
        # Plot preprocessed waveform
        t_proc = np.linspace(0, len(y_proc)/sr, len(y_proc))
        axes[0, 1].plot(t_proc, y_proc, color='#10b981', alpha=0.8)
        axes[0, 1].set_title(f"Preprocessed Waveform ({preprocess_method})", color='#cbd5e1')
        axes[0, 1].grid(True, alpha=0.1)
        
        # Plot raw spectrogram with pitch overlay
        D_raw = librosa.amplitude_to_db(np.abs(librosa.stft(y_raw)), ref=np.max)
        librosa.display.specshow(D_raw, sr=sr, x_axis='time', y_axis='log', ax=axes[1, 0])
        axes[1, 0].set_title("Raw Spectrogram & Pitch Tracking", color='#cbd5e1')
        axes[1, 0].set_ylabel("Frequency (Log Hz)", color='#cbd5e1')
        axes[1, 0].set_xlabel("Time (s)", color='#cbd5e1')
        if len(f0_raw) > 0:
            times_raw = librosa.times_like(f0_raw, sr=sr, hop_length=512)
            pitch_plot_raw = f0_raw.copy()
            pitch_plot_raw[(pitch_plot_raw < 75) | (pitch_plot_raw > 400)] = np.nan
            axes[1, 0].plot(times_raw, pitch_plot_raw, color='#f43f5e', linewidth=2.5, label='Pitch (F0)')
            axes[1, 0].legend(loc='upper right')
        
        # Plot preprocessed spectrogram with pitch overlay
        D_proc = librosa.amplitude_to_db(np.abs(librosa.stft(y_proc)), ref=np.max)
        librosa.display.specshow(D_proc, sr=sr, x_axis='time', y_axis='log', ax=axes[1, 1])
        axes[1, 1].set_title("Preprocessed Spectrogram & Pitch Tracking", color='#cbd5e1')
        axes[1, 1].set_xlabel("Time (s)", color='#cbd5e1')
        if len(f0_proc) > 0:
            times_proc = librosa.times_like(f0_proc, sr=sr, hop_length=512)
            pitch_plot_proc = f0_proc.copy()
            pitch_plot_proc[(pitch_plot_proc < 75) | (pitch_plot_proc > 400)] = np.nan
            axes[1, 1].plot(times_proc, pitch_plot_proc, color='#f43f5e', linewidth=2.5, label='Pitch (F0)')
            axes[1, 1].legend(loc='upper right')
        
        # Style plots
        for ax in axes.flat:
            ax.tick_params(colors='#cbd5e1')
            ax.set_facecolor('#0f0c1b')
        fig.patch.set_facecolor('#0f0c1b')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Analysis phase
        if analyze_btn or st.session_state.get('run_init', False):
            st.session_state['run_init'] = True
            
            with st.spinner("⚡ Running calibrated multimodal inference (ASR + Sentiment + Vocal Emotion)..."):
                start_time = time.time()
                
                # --- RAW SPEECH INFERENCE ---
                asr_raw_res = asr(y_raw)
                text_raw = asr_raw_res["text"].strip()
                
                if text_raw:
                    sent_raw_res = sentiment(text_raw)[0]
                    text_sentiment_raw = sent_raw_res["label"].lower()
                    sent_score_raw = sent_raw_res["score"]
                else:
                    text_sentiment_raw = "neutral"
                    sent_score_raw = 0.0
                
                # Normalize & Trim silence from RAW audio for classification
                y_raw_ser = trim_silence(y_raw)
                y_raw_ser = normalize_volume(y_raw_ser)
                
                emotion_raw_preds = emotion(y_raw_ser)
                # Apply multimodal fusion calibration
                emotion_raw_preds = fuse_modalities(
                    emotion_raw_preds, 
                    text_sentiment_raw, 
                    sent_score_raw, 
                    pitch_mean_raw
                )
                dominant_emotion_raw = emotion_raw_preds[0]["label"]
                
                # --- PREPROCESSED SPEECH INFERENCE ---
                asr_proc_res = asr(y_proc)
                text_proc = asr_proc_res["text"].strip()
                
                if text_proc:
                    sent_proc_res = sentiment(text_proc)[0]
                    text_sentiment_proc = sent_proc_res["label"].lower()
                    sent_score_proc = sent_proc_res["score"]
                else:
                    text_sentiment_proc = "neutral"
                    sent_score_proc = 0.0
                
                # Normalize & Trim silence from PREPROCESSED audio for classification
                y_proc_ser = trim_silence(y_proc)
                y_proc_ser = normalize_volume(y_proc_ser)
                
                emotion_proc_preds = emotion(y_proc_ser)
                # Apply multimodal fusion calibration
                emotion_proc_preds = fuse_modalities(
                    emotion_proc_preds, 
                    text_sentiment_proc, 
                    sent_score_proc, 
                    pitch_mean_proc
                )
                dominant_emotion_proc = emotion_proc_preds[0]["label"]
                
                latency = (time.time() - start_time) * 1000
                
            # Sarcasm Check (Raw)
            is_sarcastic_raw = False
            sarcasm_reason_raw = "Normal speech alignment."
            if text_sentiment_raw == "positive" and dominant_emotion_raw in ["ang", "sad"]:
                is_sarcastic_raw = True
                sarcasm_reason_raw = f"Literal words are POSITIVE, but raw vocal tone is {dominant_emotion_raw.upper()}."
            elif text_sentiment_raw == "negative" and dominant_emotion_raw == "hap":
                is_sarcastic_raw = True
                sarcasm_reason_raw = "Literal words are NEGATIVE, but raw vocal tone is HAPPY."
            elif text_sentiment_raw == "neutral" and dominant_emotion_raw in ["ang", "hap", "sad"]:
                is_sarcastic_raw = True
                sarcasm_reason_raw = f"Neutral statement spoken with emotional raw voice ({dominant_emotion_raw.upper()})."

            # Sarcasm Check (Preprocessed)
            is_sarcastic_proc = False
            sarcasm_reason_proc = "Normal speech alignment."
            if text_sentiment_proc == "positive" and dominant_emotion_proc in ["ang", "sad"]:
                is_sarcastic_proc = True
                sarcasm_reason_proc = f"Literal words are POSITIVE, but preprocessed vocal tone is {dominant_emotion_proc.upper()}."
            elif text_sentiment_proc == "negative" and dominant_emotion_proc == "hap":
                is_sarcastic_proc = True
                sarcasm_reason_proc = "Literal words are NEGATIVE, but preprocessed vocal tone is HAPPY."
            elif text_sentiment_proc == "neutral" and dominant_emotion_proc in ["ang", "hap", "sad"]:
                is_sarcastic_proc = True
                sarcasm_reason_proc = f"Neutral statement spoken with emotional preprocessed voice ({dominant_emotion_proc.upper()})."
                
            # Sarcasm Alert Display
            sarc1, sarc2 = st.columns(2)
            with sarc1:
                if is_sarcastic_raw:
                    st.markdown(f"""
                    <div class="sarcasm-banner">
                        <div class="sarcasm-title">⚠️ SARCASM DETECTED (RAW)</div>
                        <div class="sarcasm-reason">{sarcasm_reason_raw}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success("✅ Raw Speech Alignment: Normal")
            with sarc2:
                if is_sarcastic_proc:
                    st.markdown(f"""
                    <div class="sarcasm-banner">
                        <div class="sarcasm-title">⚠️ SARCASM DETECTED (PREPROCESSED)</div>
                        <div class="sarcasm-reason">{sarcasm_reason_proc}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success("✅ Preprocessed Speech Alignment: Normal")
                
            # Main metrics display (Side-by-Side raw vs preprocessed)
            col_raw_dashboard, col_proc_dashboard = st.columns(2)
            
            with col_raw_dashboard:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.subheader("📝 Raw Verbal Content (ASR & NLP)")
                st.write("**Whisper Transcription:**")
                st.info(f"\"{text_raw}\"" if text_raw else "No speech detected.")
                
                # Sentiment Display
                if text_raw:
                    badge_color = "#10b981" if text_sentiment_raw == "positive" else "#ef4444"
                    st.markdown(f"""
                    **Text Sentiment Analysis:**
                    <span style="background-color: {badge_color}; padding: 6px 12px; border-radius: 8px; font-weight: bold; color: white;">
                        {text_sentiment_raw.upper()} ({sent_score_raw:.2%})
                    </span>
                    """, unsafe_allow_html=True)
                    st.progress(sent_score_raw)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.subheader("🎭 Raw Vocal Emotion (Calibrated Wav2Vec2 SER)")
                
                # Show dominant emotion
                emoji_r, name_r, color_r = EMOJI_MAP.get(dominant_emotion_raw, ("😐", "Unknown", "#a0aec0"))
                st.markdown(f"""
                **Dominant Emotion Detected:**
                <div style="font-size: 54px; margin: 10px 0;">{emoji_r} <span style="color: {color_r}; font-weight: 800; font-size: 38px;">{name_r.upper()}</span></div>
                """, unsafe_allow_html=True)
                
                # Plot probabilities
                st.write("**Confidence Breakdown:**")
                labels_r = []
                scores_r = []
                colors_r = []
                for p in emotion_raw_preds:
                    lbl = p["label"]
                    score = p["score"]
                    _, d_name, d_color = EMOJI_MAP.get(lbl, ("😐", lbl, "#a0aec0"))
                    labels_r.append(d_name)
                    scores_r.append(score)
                    colors_r.append(d_color)
                
                fig_bar_r, ax_bar_r = plt.subplots(figsize=(6, 2.5))
                bars_r = ax_bar_r.barh(labels_r, scores_r, color=colors_r, height=0.6)
                ax_bar_r.set_xlim(0, 1.0)
                ax_bar_r.set_facecolor('#0f0c1b')
                fig_bar_r.patch.set_facecolor('#0f0c1b')
                ax_bar_r.tick_params(colors='#cbd5e1')
                ax_bar_r.spines['top'].set_visible(False)
                ax_bar_r.spines['right'].set_visible(False)
                ax_bar_r.spines['bottom'].set_color('#cbd5e1')
                ax_bar_r.spines['left'].set_color('#cbd5e1')
                
                for bar in bars_r:
                    width = bar.get_width()
                    ax_bar_r.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{width:.1%}', 
                                va='center', ha='left', color='#cbd5e1', fontweight='bold', fontsize=9)
                                
                st.pyplot(fig_bar_r)
                plt.close(fig_bar_r)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.subheader("🔊 Raw Acoustic Metrics")
                snr_val = estimate_snr(y_raw, sr)
                wpm_val = (len(text_raw.split()) / (len(y_raw)/sr)) * 60 if len(y_raw) > 0 and text_raw else 0
                st.metric("Estimated Signal-to-Noise Ratio (SNR)", f"{snr_val:.2f} dB")
                st.metric("Pitch (F0) Mean / Std Dev", f"{pitch_mean_raw:.1f} Hz (±{pitch_std_raw:.1f} Hz)")
                st.metric("Speech Rate", f"{wpm_val:.0f} WPM")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col_proc_dashboard:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.subheader(f"📝 Preprocessed Verbal ({preprocess_method})")
                st.write("**Whisper Transcription:**")
                st.info(f"\"{text_proc}\"" if text_proc else "No speech detected.")
                
                # Sentiment Display
                if text_proc:
                    badge_color = "#10b981" if text_sentiment_proc == "positive" else "#ef4444"
                    st.markdown(f"""
                    **Text Sentiment Analysis:**
                    <span style="background-color: {badge_color}; padding: 6px 12px; border-radius: 8px; font-weight: bold; color: white;">
                        {text_sentiment_proc.upper()} ({sent_score_proc:.2%})
                    </span>
                    """, unsafe_allow_html=True)
                    st.progress(sent_score_proc)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.subheader("🎭 Preprocessed Vocal Emotion (Calibrated)")
                
                # Show dominant emotion
                emoji_p, name_p, color_p = EMOJI_MAP.get(dominant_emotion_proc, ("😐", "Unknown", "#a0aec0"))
                st.markdown(f"""
                **Dominant Emotion Detected:**
                <div style="font-size: 54px; margin: 10px 0;">{emoji_p} <span style="color: {color_p}; font-weight: 800; font-size: 38px;">{name_p.upper()}</span></div>
                """, unsafe_allow_html=True)
                
                # Plot probabilities
                st.write("**Confidence Breakdown:**")
                labels_p = []
                scores_p = []
                colors_p = []
                for p in emotion_proc_preds:
                    lbl = p["label"]
                    score = p["score"]
                    _, d_name, d_color = EMOJI_MAP.get(lbl, ("😐", lbl, "#a0aec0"))
                    labels_p.append(d_name)
                    scores_p.append(score)
                    colors_p.append(d_color)
                
                fig_bar_p, ax_bar_p = plt.subplots(figsize=(6, 2.5))
                bars_p = ax_bar_p.barh(labels_p, scores_p, color=colors_p, height=0.6)
                ax_bar_p.set_xlim(0, 1.0)
                ax_bar_p.set_facecolor('#0f0c1b')
                fig_bar_p.patch.set_facecolor('#0f0c1b')
                ax_bar_p.tick_params(colors='#cbd5e1')
                ax_bar_p.spines['top'].set_visible(False)
                ax_bar_p.spines['right'].set_visible(False)
                ax_bar_p.spines['bottom'].set_color('#cbd5e1')
                ax_bar_p.spines['left'].set_color('#cbd5e1')
                
                for bar in bars_p:
                    width = bar.get_width()
                    ax_bar_p.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{width:.1%}', 
                                va='center', ha='left', color='#cbd5e1', fontweight='bold', fontsize=9)
                                
                st.pyplot(fig_bar_p)
                plt.close(fig_bar_p)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.subheader("🔊 Denoised Acoustic Metrics")
                snr_val_proc = estimate_snr(y_proc, sr)
                wpm_val_proc = (len(text_proc.split()) / (len(y_proc)/sr)) * 60 if len(y_proc) > 0 and text_proc else 0
                st.metric("Estimated Signal-to-Noise Ratio (SNR)", f"{snr_val_proc:.2f} dB", delta=f"{snr_val_proc - snr_val:+.2f} dB")
                st.metric("Pitch (F0) Mean / Std Dev", f"{pitch_mean_proc:.1f} Hz (±{pitch_std_proc:.1f} Hz)")
                st.metric("Speech Rate", f"{wpm_val_proc:.0f} WPM")
                st.markdown('</div>', unsafe_allow_html=True)
                
            # Pipeline performance
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h3>⚡ Multimodal Pipeline Latency</h3>
                <h2 style="color: #a78bfa; font-weight: 800; font-size: 36px; margin: 5px 0;">{latency:.0f} ms</h2>
                <p style="color: #cbd5e1; margin-bottom: 0;">ASR (Whisper) + Sentiment (DistilBERT) + SER (Wav2Vec2) ran successfully locally.</p>
            </div>
            """, unsafe_allow_html=True)

    else:
        # Default placeholder screen
        st.info("👈 Please select or record an audio file in the sidebar to launch the analysis.")
        
        st.markdown("""
        ### 🧪 How does it work?
        1. **Choose an Audio Source**: Select one of the prepared emotional audio files from the RAVDESS dataset, record your own voice in real-time, or upload a custom WAV file.
        2. **Configure a Preprocessing Filter**: Select a DSP algorithm (Wiener Filter or Spectral Subtraction) or leave it raw. Adjust hyperparameters in the sidebar to customize filter strengths.
        3. **Multimodal Inference Pipeline**:
           - **ASR**: Whisper Tiny transcribes the verbal speech content.
           - **NLP**: DistilBERT analyzes text sentiment alignment.
           - **SER**: Wav2Vec2 detects vocal emotions directly from raw acoustic wave inputs. 
           - **Calibration**: Trims silence and peak normalizes speech volume to handle close-mic clipping. It then fuses text sentiment and voice pitch to correct classification errors (such as mistaking loud happy talking for anger).
        4. **Comparative Analysis**: Compare raw speech and preprocessed speech side-by-side! Observe the direct influence of noise subtraction on model prediction scores, transcription accuracy, SNR gain, and pitch contour tracking.
        """)

if __name__ == "__main__":
    main()
