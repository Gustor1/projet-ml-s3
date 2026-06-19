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

def preprocess_wiener(audio):
    return wiener(audio).astype(np.float32)

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
    
    # Primary action button
    analyze_btn = st.sidebar.button("🔍 Run Full Pipeline", use_container_width=True)
    
    if y_raw is not None:
        # Preprocess
        if preprocess_method == "None (Raw Audio)":
            y_proc = preprocess_none(y_raw)
        elif preprocess_method == "Wiener Filter":
            y_proc = preprocess_wiener(y_raw)
        else:
            y_proc = preprocess_spectral_subtraction(y_raw, sr)
            
        # Audio Players Row
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔊 Raw Audio")
            st.audio(get_wav_bytes(y_raw, sr), format="audio/wav")
        with col2:
            st.subheader(f"🔊 Preprocessed Audio ({preprocess_method})")
            st.audio(get_wav_bytes(y_proc, sr), format="audio/wav")
            
        # Waveform & Spectrogram comparison
        st.subheader("📊 Signal Analysis (Waveform & Spectrogram)")
        fig, axes = plt.subplots(2, 2, figsize=(14, 6))
        
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
        
        # Plot raw spectrogram
        D_raw = librosa.amplitude_to_db(np.abs(librosa.stft(y_raw)), ref=np.max)
        librosa.display.specshow(D_raw, sr=sr, x_axis='time', y_axis='linear', ax=axes[1, 0])
        axes[1, 0].set_title("Raw Spectrogram", color='#cbd5e1')
        axes[1, 0].set_ylabel("Hz", color='#cbd5e1')
        axes[1, 0].set_xlabel("Time (s)", color='#cbd5e1')
        
        # Plot preprocessed spectrogram
        D_proc = librosa.amplitude_to_db(np.abs(librosa.stft(y_proc)), ref=np.max)
        librosa.display.specshow(D_proc, sr=sr, x_axis='time', y_axis='linear', ax=axes[1, 1])
        axes[1, 1].set_title("Preprocessed Spectrogram", color='#cbd5e1')
        axes[1, 1].set_xlabel("Time (s)", color='#cbd5e1')
        
        # Style plots
        for ax in axes.flat:
            ax.tick_params(colors='#cbd5e1')
            ax.set_facecolor('#0f0c1b')
        fig.patch.set_facecolor('#0f0c1b')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Analysis phase
        if analyze_btn or st.session_state.get('run_init', False):
            st.session_state['run_init'] = True
            
            with st.spinner("⚡ Running multimodal inference (ASR + Sentiment + Vocal Emotion)..."):
                start_time = time.time()
                
                # 1. ASR Whisper
                asr_res = asr(y_proc)
                text = asr_res["text"].strip()
                
                # 2. Text Sentiment
                if text:
                    sent_res = sentiment(text)[0]
                    text_sentiment = sent_res["label"].lower() # 'positive' or 'negative'
                    sent_score = sent_res["score"]
                else:
                    text_sentiment = "neutral"
                    sent_score = 0.0
                    
                # 3. Speech Emotion Recognition (SER)
                # Wav2Vec2 requires file path or numpy array
                # Save processed to temp buffer to feed pipeline
                temp_wav = io.BytesIO()
                sf.write(temp_wav, y_proc, sr, format='WAV')
                temp_wav.seek(0)
                
                emotion_preds = emotion(temp_wav.read())
                
                latency = (time.time() - start_time) * 1000
                
            # Sarcasm Check
            dominant_emotion = emotion_preds[0]["label"] # e.g. 'ang', 'hap', 'sad', 'neu'
            
            is_sarcastic = False
            sarcasm_reason = ""
            
            if text_sentiment == "positive" and dominant_emotion in ["ang", "sad"]:
                is_sarcastic = True
                sarcasm_reason = f"Literal words are POSITIVE, but the vocal tone is {dominant_emotion.upper()}."
            elif text_sentiment == "negative" and dominant_emotion == "hap":
                is_sarcastic = True
                sarcasm_reason = "Literal words are NEGATIVE, but the vocal tone is HAPPY."
                
            # Sarcasm Alert Display
            if is_sarcastic:
                st.markdown(f"""
                <div class="sarcasm-banner">
                    <div class="sarcasm-title">⚠️ SARCASM / PASSIVE-AGGRESSIVE DETECTED!</div>
                    <div class="sarcasm-reason">{sarcasm_reason}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("✅ Speech Alignment Normal: The vocal emotion matches the verbal message sentiment.")
                
            # Main metrics display
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.subheader("📝 Verbal Content (ASR & NLP)")
                st.write("**Whisper Transcription:**")
                st.info(f"\"{text}\"" if text else "No speech detected.")
                
                # Sentiment Display
                if text:
                    badge_color = "#10b981" if text_sentiment == "positive" else "#ef4444"
                    st.markdown(f"""
                    **Text Sentiment Analysis:**
                    <span style="background-color: {badge_color}; padding: 6px 12px; border-radius: 8px; font-weight: bold; color: white;">
                        {text_sentiment.upper()} ({sent_score:.2%})
                    </span>
                    """, unsafe_allow_html=True)
                    st.progress(sent_score)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Latency Card
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.subheader("⚡ Pipeline Performance")
                st.metric("Total Latency", f"{latency:.0f} ms", delta="Local GPU/CPU execution")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col_right:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.subheader("🎭 Vocal Emotion (Wav2Vec2 SER)")
                
                # Show dominant emotion
                emoji, name, color = EMOJI_MAP.get(dominant_emotion, ("😐", "Unknown", "#a0aec0"))
                st.markdown(f"""
                **Dominant Emotion Detected:**
                <div style="font-size: 54px; margin: 10px 0;">{emoji} <span style="color: {color}; font-weight: 800; font-size: 38px;">{name.upper()}</span></div>
                """, unsafe_allow_html=True)
                
                # Plot probabilities
                st.write("**Confidence Breakdown:**")
                labels = []
                scores = []
                colors = []
                for p in emotion_preds:
                    lbl = p["label"]
                    score = p["score"]
                    _, d_name, d_color = EMOJI_MAP.get(lbl, ("😐", lbl, "#a0aec0"))
                    labels.append(d_name)
                    scores.append(score)
                    colors.append(d_color)
                
                fig_bar, ax_bar = plt.subplots(figsize=(6, 3))
                bars = ax_bar.barh(labels, scores, color=colors, height=0.6)
                ax_bar.set_xlim(0, 1.0)
                ax_bar.set_facecolor('#0f0c1b')
                fig_bar.patch.set_facecolor('#0f0c1b')
                ax_bar.tick_params(colors='#cbd5e1')
                ax_bar.spines['top'].set_visible(False)
                ax_bar.spines['right'].set_visible(False)
                ax_bar.spines['bottom'].set_color('#cbd5e1')
                ax_bar.spines['left'].set_color('#cbd5e1')
                
                # Annotate percentages
                for bar in bars:
                    width = bar.get_width()
                    ax_bar.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{width:.1%}', 
                                va='center', ha='left', color='#cbd5e1', fontweight='bold', fontsize=9)
                                
                st.pyplot(fig_bar)
                st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Default placeholder screen
        st.info("👈 Please select or record an audio file in the sidebar to launch the analysis.")
        
        st.markdown("""
        ### 🧪 How does it work?
        1. **Choose an Audio Sample**: You can select one of the prepared emotional audio files from the RAVDESS dataset, record your own voice in real-time, or upload a custom WAV file.
        2. **Configure a Preprocessing Filter**: Select a DSP algorithm (Wiener Filter or Spectral Subtraction) or leave it raw.
        3. **Inference Pipeline**:
           - **ASR**: Whisper Tiny transcribes the verbal speech.
           - **NLP**: DistilBERT analyzes the text sentiment.
           - **SER**: Wav2Vec2 detects the vocal emotion from pitch and acoustics.
        4. **Sarcasm Engine**: The dashboard cross-checks both modalities to flag mismatched tone-vs-text alerts!
        """)

if __name__ == "__main__":
    main()
