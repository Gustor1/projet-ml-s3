# ⚖️ Engineering Trade-offs in Multi-Task Speech Pipelines

This document analyzes the core engineering and design trade-offs identified during the evaluation of local audio preprocessing for Automatic Speech Recognition (ASR) and Speech Emotion Recognition (SER) models.

---

## 1. Upstream Preprocessing vs. Noise Character (The Lab-to-Real-World Gap)
- **Context**: ASR pipelines often implement an always-on noise filter or trigger denoising based on an estimated Signal-to-Noise Ratio (SNR).
- **Trade-off**: While Wiener filtering improves word accuracy under stationary white Gaussian noise at low SNR (**5 dB**, $-2.75\%$ absolute $WER$ gain), it degrades performance under realistic noise profiles: pink noise ($+11.13\%$), non-stationary urban street noise ($+9.31\%$), and babble noise ($+8.12\%$).
- **Analysis**: Classical filters assume stationary noise. On colored or non-stationary noise, these assumptions fail, causing **spectral tilt distortion** (formant attenuation) and **tracking lag** (temporal smearing). These distortions corrupt the acoustic cues that Whisper's transformer encoder relies on.
- **Engineering Decision**: **Discard always-on classical preprocessing.** SNR-based activation is insufficient because a simple energy-based SNR meter cannot distinguish between flat white noise and realistic colored or non-stationary noise. The default recommendation for edge-ASR is to pass raw audio directly to the model, allowing Whisper's encoder to handle noise natively, or to deploy deep learning-based source separation (e.g., DCCRN) that models speech patterns.

---

## 2. DSP Compute Overhead vs. ASR Decoding Optimization
- **Context**: Upstream audio preprocessing consumes local CPU cycles but can improve signal quality, potentially reducing downstream model computation.
- **Trade-off**: Upstream CPU latency overhead vs. downsteam autoregressive decoding speed.
- **Analysis**: Applying the Wiener filter adds $50\text{–}100\text{ ms}$ of DSP computation overhead on edge CPUs. Under severe noise, this clean-up increases the decoder's token-level confidence, which reduces the number of beam search branches and decoding iterations, saving $\sim 100\text{ ms}$ of downstream GPU/CPU time. However, under mild noise, the DSP filter provides no decoding speedup, making the upstream compute overhead a net latency penalty.
- **Engineering Decision**: The DSP compute overhead is **unjustified** for classical filters. Since these filters degrade accuracy in realistic environments, the processing cost yields negative value. The optimal configuration for edge-ASR is raw signal inference, saving local CPU cycles and memory.

---

## 3. Model Parameters vs. Intrinsic Noise Robustness
- **Context**: Selecting the model size for edge deployment under RAM and thermal constraints.
- **Trade-off**: Computational footprint (RAM, disk space, battery consumption) vs. model capacity and noise robustness.
- **Analysis**: `openai/whisper-tiny` (39M parameters, ~150MB footprint) is suitable for edge memory limits but is sensitive to noise, exhibiting an $18.60\%$ baseline $WER$ on clean speech and $27.47\%$ $WER$ at 5 dB white noise. Larger models (e.g., `whisper-base` at 74M or `whisper-small` at 244M) possess higher capacity, allowing them to build noise-robust representations natively.
- **Engineering Decision**: We utilized Whisper-tiny to stress-test the preprocessing filters. However, for production systems, upgrading the model size to `whisper-base` or `whisper-small` is a more robust way to gain noise tolerance than adding classical DSP filters to a smaller model. As shown in the cross-modal ablation study, upgrading the model size also reduces sentiment flip cascades in downstream NLP tasks.

---

## 4. Spectral Subtraction vs. Signal Integrity
- **Context**: Using magnitude-subtraction algorithms to remove background noise.
- **Trade-off**: Noise attenuation depth vs. the introduction of spectral holes and musical noise.
- **Analysis**: Spectral subtraction removes background noise but introduces **spectral holes** (zeroed bins due to over-subtraction rectification) and **musical noise** (random spectral peaks). These artifacts distort the spectral envelope. While tolerable to human hearing, this distortion destroys the acoustic cues needed for phoneme classification, causing Whisper's autoregressive decoder to collapse and generate hallucinations ($WER \ge 100\%$).
- **Engineering Decision**: **Spectral subtraction is discarded.** The phase and magnitude distortions it introduces are more damaging to transformer ASR models than the original noise.

---

## 5. Verbal Enhancement vs. Prosodic Preservation (ASR vs. SER Routing)
- **Context**: Preprocessing requirements for joint speech-to-text (ASR) and emotion recognition (SER) tasks.
- **Trade-off**: Cleaning the spectral envelope for ASR vs. preserving the temporal and pitch dynamics for SER.
- **Analysis**: Speech enhancement filters (Wiener) smooth amplitude and frequency micro-variations to improve verbal intelligibility. However, these micro-variations (jitter, shimmer, pitch contours) contain the vocal prosody that SER models rely on. Wiener filtering drops Wav2Vec2 SER accuracy under white noise by **21.43% absolute** (from $45.83\%$ to $24.40\%$), acting as an "emotional eraser" that reduces expressive voice features to neutrality or sadness.
- **Engineering Decision**: **Do not apply classical preprocessing globally in multi-task pipelines.** For systems requiring both ASR and SER (e.g., emotion-aware conversational interfaces), implement a **parallel routing architecture**:
  - Denoise the audio stream (if necessary) and route it to the ASR model.
  - Pass the original noisy audio (applying only silence trimming and peak amplitude normalization) directly to the SER model.
  - Calibrate the SER predictions in post-processing using a **multimodal fusion engine** that combines the text sentiment (DistilBERT) and estimated pitch ($F_0$ from Librosa's YIN tracker).