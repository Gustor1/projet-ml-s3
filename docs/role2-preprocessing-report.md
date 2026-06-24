# Audio Preprocessing & DSP Architecture Report (Role 2)

## 1. Introduction and Objectives
The core objective of the preprocessing pipeline in this project is to enhance the robustness of the downstream ASR (Whisper) and SER (Wav2Vec2) models when dealing with noisy, real-world audio. Given the severe degradation of neural networks under acoustic domain shifts, robust front-end signal processing is essential. 

However, as highlighted by our empirical results, ASR and SER have fundamentally conflicting requirements. While ASR benefits from static noise suppression, SER relies heavily on prosodic micro-features (pitch, jitter, shimmer) which are often destroyed by traditional denoising. This report details the theoretical justification and engineering trade-offs behind our preprocessing architecture, directly addressing the limitations of existing approaches.

## 2. Denoising Algorithms: Choices and Justification

### 2.1 The Wiener Filter
**Choice Justification:** The Wiener filter was selected as our primary stationary noise reduction baseline. It is a well-established optimal linear estimator that minimizes the mean-square error (MMSE) between the estimated and true signal spectra.
**Theoretical Basis:** It assumes that the noise is stationary with a flat power spectral density. We implemented this via `scipy.signal.wiener`.
**Limitations Encountered:** As demonstrated in our experiments (Role 4), while Wiener filtering improved WER by -2.75% under 5 dB white noise, it severely degraded ASR performance on non-stationary colored noise (e.g., urban or pink noise). This happens because the Wiener filter cannot adapt rapidly enough to transient acoustic events, resulting in spectral distortion.

### 2.2 Spectral Subtraction
**Choice Justification:** Spectral subtraction (using overlap-add FFT) was chosen to provide a dynamic noise-floor estimation approach. By estimating the noise profile from an initial silence segment, it subtracts the noise magnitude spectrum from the noisy signal.
**Theoretical Limits and Observations:** Literature (e.g., *Evans et al., 2005*) points out fundamental limitations of spectral subtraction, specifically the generation of "musical noise" (isolated spectral peaks) and spectral holes. Our findings confirm this: Whisper's self-attention mechanism interprets these musical noise artifacts as phonemic structures, leading to significant word insertion errors (WER increased by up to +27.0%). 
**Conclusion:** We maintain the spectral subtraction module in our API (`preprocessing/denoise.py`) as a documented negative result. It serves as a scientific baseline demonstrating why magnitude-only subtraction fails for modern transformer-based ASR models.

## 3. The Parallel Stream Routing Architecture

To resolve the conflict between ASR and SER requirements, we designed a **Parallel Stream Routing Architecture** (`preprocessing/pipeline.py`).

*   **The Problem:** Traditional pipelines apply a single denoising pass before feeding the audio to downstream models. However, our experiments proved that Wiener filtering acts as an "emotional eraser," dropping SER accuracy from 45.8% to 24.4% by smoothing out vocal prosody.
*   **The Solution:** We decoupled the audio streams. 
    1.  **ASR Stream:** Optionally passes through the Wiener filter (if SNR < 10 dB).
    2.  **SER Stream:** Bypasses denoising completely. Instead, it only undergoes silence trimming and peak normalization.

This architectural decision ensures that the Wav2Vec2 model receives the untouched high-frequency prosodic cues, while Whisper benefits from static noise reduction when necessary.

## 4. Voice Activity Detection (VAD) and Normalization

Before SER inference, the audio must be conditioned. In `preprocessing/vad.py`, we implemented:
1.  **Energy-based Silence Trimming:** Removes silent margins. Wav2Vec2 attention mechanisms can be distracted by long periods of silence, especially if those periods contain room impulse responses.
2.  **Peak Normalization:** Neural SER models are highly sensitive to input amplitude. By scaling the amplitude to `[-1.0, 1.0]`, we ensure that the model evaluates the *relative* energy dynamics (emotion) rather than the *absolute* recording gain (microphone distance). This simple step yielded a +20% relative improvement in SER accuracy.

## 5. Conclusion

Our audio preprocessing approach goes beyond applying black-box filters. By theoretically analyzing the impact of classical DSP on transformer architectures, we identified critical flaws in naïve denoising pipelines. Our **Parallel Routing Architecture** successfully addresses these limitations, providing an engineered solution that optimizes both transcription and emotion recognition simultaneously.

## References
1. Evans, N. W., et al. (2005). "On the Fundamental Limitations of Spectral Subtraction," *Proc. EUSIPCO*.
2. Oppenheim, A., & Lim, J. (1981). "The importance of phase in signals," *Proceedings of the IEEE*.
3. Gong, Y., et al. (2023). "Whisper-AT: Noise-Robust Automatic Speech Recognizers are Also Strong Audio Event Taggers," *Interspeech*.
