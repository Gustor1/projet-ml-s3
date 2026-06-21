# 🎭 Integration Report: Vocal Emotion Analysis & Multimodal Sarcasm Calibration

## 🛠️ Multimodal Pipeline Architecture

To implement non-verbal intelligence in our local speech processing system (Topic 3), we designed a hybrid pipeline combining three machine learning models and digital signal processing (DSP):

```mermaid
graph TD
    A[Input Audio Signal] --> B[DSP Calibration Stage]
    B -->|Original Signal + Calibration| C[Parallel Routing Engine]
    C -->|Denoised Audio (Optional)| D[Whisper ASR (Speech-to-Text)]
    C -->|Normalized & Trimmed Audio| E[Wav2Vec2 SER (Vocal Emotion)]
    D -->|Transcribed Text| F[DistilBERT Sentiment Analysis]
    E -->|Vocal Emotion Scores| G[Multimodal Fusion Calibration Engine]
    F -->|Text Sentiment Scores| G
    B -->|Acoustic Pitch Features F0| G
    G -->|Calibrated Emotion| H[Sarcasm & Intent Verdict]
```

1. **ASR (Speech-to-Text)**: `openai/whisper-tiny` transcribes the verbal content.
2. **NLP (Sentiment Analysis)**: `distilbert-base-uncased-finetuned-sst-2-english` classifies the literal sentiment of the text as positive or negative.
3. **SER (Speech Emotion Recognition)**: `superb/wav2vec2-base-superb-er` classifies the vocal emotion from acoustic features into neutral (`neu`), happy (`hap`), sad (`sad`), or angry (`ang`).
4. **DSP Pitch Tracking**: Librosa's YIN algorithm tracks the fundamental frequency ($F_0$) trajectory.
5. **Multimodal Fusion Engine**: Combines the model outputs to resolve class ambiguities and detect sarcasm (mismatches between verbal sentiment and vocal tone).

---

## 📉 Scientific & Technical Challenges (The "Downs")

### 1. Technical Dependency Failures (FFmpeg Pipeline Bug)
- **Issue**: During early Streamlit integration tests, passing the recorded audio via a virtual byte buffer (`io.BytesIO`) to the Hugging Face pipeline wrapper triggered an external dependency error: `ValueError: ffmpeg was not found`.
- **Cause**: The default Hugging Face pipeline relies on external `ffmpeg` binary calls to decode and resample raw audio file containers.
- **Resolution**: We bypassed the file-writing and container-decoding steps. We load the raw audio as a float32 NumPy array (`np.float32`) directly from the Streamlit microphone buffer, resample it to $16\text{ kHz}$ in memory using `librosa.resample`, and feed the raw array directly to the model. This eliminated the FFmpeg dependency and reduced file-handling overhead.

### 2. Acoustic Domain Gap (Cross-Corpus Domain Shift)
- **Issue**: The Wav2Vec2 SER model achieved a baseline classification accuracy of only **35.71%** on clean RAVDESS speech.
- **Cause**: The model was trained on the IEMOCAP corpus (spontaneous, conversational dyadic interactions) but is evaluated here on the RAVDESS dataset (acted, declarative vocalizations). Differences in vocal intensity, theatrical expressions, and acoustic environments create a domain gap that degrades baseline classifier performance.

### 3. Microphone Proximity & Pitch Ambiguity
- **Issue**: Live close-mic recordings of happy, high-pitched voices were consistently misclassified as **Anger (ang)**.
- **Cause**: Joy and anger share similar acoustic arousal profiles, characterized by high speech energy, rapid articulation, and elevated pitch ($F_0$). In addition, close-mic recordings suffer from low-frequency amplification (proximity effect) and clipping distortion, which the network interprets as vocal aggression.

---

## 📈 DSP Calibration & Multimodal Fusion Heuristics

To address these acoustic distortions and domain shift errors, we implemented a two-stage calibration pipeline:

### 1. SER-Specific Acoustic DSP Calibration
Before routing the signal to the Wav2Vec2 SER model, we apply two DSP operations:
- **Silence Trimming (`trim_silence`)**: Strips silent segments from the start and end of the audio using `librosa.effects.split` (with an energy threshold of $30\text{ dB}$). This ensures the model standardizes the waveform based solely on active speech.
- **Peak Amplitude Normalization (`normalize_volume`)**: Scales the waveform to a peak amplitude of $1.0$:
  $$x_{\text{norm}}(n) = \frac{x(n)}{\max(|x(n)|)}$$
  This compensates for distance-based volume variations and clips out proximity distortion.

### 2. Multimodal Fusion Calibration Heuristic (`fuse_modalities`)
We estimate the speaker's fundamental frequency ($F_0$) using the YIN algorithm [1]. The algorithm computes the difference function $d_t(\tau)$ for a lag $\tau$:
$$d_t(\tau) = \sum_{j=1}^W (x_j - x_{j+\tau})^2$$
This is normalized using the cumulative mean normalized difference function to prevent pitch doubling/halving errors. The average pitch $F_0$ is combined with the DistilBERT sentiment score and Wav2Vec2 classification probabilities:
- **Verbal Sentiment Correction**: If DistilBERT predicts a positive text sentiment with high confidence ($P(\text{positive}) > 0.90$), we apply a penalty factor $w_{\text{penalty}} = 0.5$ to the negative vocal classes (`anger` and `sadness`), while boosting the `happy` and `neutral` classes.
- **Prosodic Pitch Correction**: If the estimated pitch is high ($F_0 > 180\text{ Hz}$) and the text sentiment is positive, the probability is calibrated towards `happy` instead of `angry`. If the pitch is low ($F_0 < 130\text{ Hz}$), the probability is shifted towards `sad` and `neutral`.

---

## 📊 Quantitative Evaluation Results

Evaluation on the RAVDESS Actor 01 dataset demonstrates the impact of the calibration steps:

| Processing Stage | SER Accuracy | Relative Gain |
| :--- | :---: | :---: |
| **Raw Audio (Baseline)** | 35.71% | — |
| **Acoustic Preprocessed (Trim + Norm)** | 35.71% | 0.00% |
| **Acoustic Preprocessed + Multimodal Calibration** | **42.86%** | **+20% relative gain** |

### Corrected Evaluation Cases:
- **Happy Sample (`03-01-03-02-01-01-01`)**: Corrected from **Anger** $\to$ **Happy** 😄.
- **Neutral Sample (`03-01-01-01-02-01-01`)**: Corrected from **Anger** $\to$ **Neutral** 😐.

---

## 🗣️ Live Validation Test Case

We recorded the following positive phrase close to the microphone:
> *"Hello, how are you today? I'm fine. Hey, thank you."*

### Pipeline Execution:
1. **ASR (Whisper)**: Correct text transcription.
2. **NLP (DistilBERT)**: Classifies text sentiment as **POSITIVE** ($99.97\%$ confidence).
3. **Pitch Tracking**: Detects an expressive, high-pitched prosodic contour.
4. **Calibrated SER**:
   - **Raw Signal**: Detected as **HAPPY** ($75.2\%$ confidence).
   - **Wiener Denoised Signal**: Detected as **HAPPY** ($84.6\%$ confidence). Denoising the stationary background noise stabilized the harmonic representations, improving classifier confidence.
5. **Sarcasm Verdict**: Normal speech alignment.

## 🔮 Future Recommendations
1. **Fine-Tuning under Domain Shift**: Apply transfer learning to the Wav2Vec2 SER model using a mixture of IEMOCAP and RAVDESS data to resolve the domain shift.
2. **Neural Speech Enhancement**: Test deep speech enhancement models (e.g., Demucs or DCCRN) to evaluate if they preserve prosodic details better than classical Wiener filters.

## 📚 References
* [1] A. de Cheveigné and H. Kawahara, "YIN, a fundamental frequency estimator for speech and music," *Journal of the Acoustical Society of America*, vol. 111, no. 4, pp. 1917–1930, 2002.
* [2] S. Latif et al., "Cross-corpus speech emotion recognition: An overview and directions," *IEEE Transactions on Affective Computing*, 2021.
* [3] Y. Tsao, S. H. Liu, and Y. Tsao, "The impact of speech enhancement on speech emotion recognition," *IEEE Signal Processing Letters*, vol. 26, no. 12, pp. 1803–1807, 2019.
