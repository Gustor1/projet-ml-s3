# ⚖️ Engineering Trade-offs

This document summarizes the key engineering trade-offs identified during the project. For detailed scientific analysis, refer to `docs/insights.md` and the individual experiment reports.

## 1. Preprocessing Activation vs. Noise Type (The "Lab-to-Real-World" Gap)
- **Observation**: The Wiener filter only provides a marginal benefit on synthetic white Gaussian noise at low SNR (5dB, -2.75% WER). However, it actively **degrades** performance on realistic noise profiles: pink noise (+11.13%), real urban noise (+9.31%), and babble/crowd noise (+8.12%, plus triggering ASR hallucinations).
- **Trade-off**: Activating classical preprocessing "always-on" or based solely on an SNR threshold improves lab benchmarks but severely harms real-world user experience.
- **Final Decision**: **Do NOT implement preprocessing by default.** SNR-based activation is explicitly abandoned because a simple SNR meter cannot distinguish between white noise (where Wiener helps) and realistic noise (where it harms). If preprocessing is strictly required, explore Deep Learning-based source separation (e.g., VoiceFilter, Conv-TasNet) rather than classical spectral filtering.

## 2. Latency vs. Accuracy Gain
- **Observation**: Applying the Wiener filter adds computational overhead on the edge device (CPU), while providing no end-to-end latency benefit since the ASR inference time remains roughly equivalent.
- **Trade-off**: Algorithmic complexity and processing overhead vs. potential accuracy improvement.
- **Final Decision**: The computational cost is **unjustified**. Since the accuracy "gain" is actually a degradation in realistic environments, adding this processing step provides negative value. The optimal pipeline for edge devices is direct inference without classical DSP preprocessing.

## 3. Model Size vs. Baseline Performance
- **Observation**: Whisper tiny (39M parameters) exhibits a baseline WER of ~18.6% on clean audio in our setup, which is higher than the official LibriSpeech benchmark (~7.5%) due to our specific single-speaker subset and lack of text normalization.
- **Trade-off**: Lightweight model (easy edge deployment, low memory footprint) vs. Heavy model (better inherent noise robustness and accuracy).
- **Final Decision**: We intentionally used Whisper tiny to stress-test the preprocessing pipeline. The goal was to prove whether classical preprocessing could rescue a weak model. The conclusion is that it cannot reliably do so without introducing severe artifacts. For production, upgrading to Whisper `base` or `small` is a more effective way to gain noise robustness than adding fragile DSP preprocessing to `tiny`.

## 4. Spectral Subtraction vs. System Stability
- **Observation**: Spectral subtraction consistently degraded performance across all noise types (+6.79% to +26.99% WER) and initially caused pipeline crashes due to FFT boundary mismatches.
- **Trade-off**: Aggressive noise removal vs. introduction of "musical noise" artifacts and pipeline instability.
- **Final Decision**: Spectral subtraction is **completely discarded** from the recommended pipeline. The artifacts it introduces are more damaging to Whisper's autoregressive decoder than the original noise itself.

## 5. Speech Enhancement vs. Non-Verbal Feature Preservation (ASR vs. SER)
- **Observation**: While noise removal filters (Wiener) modestly improve ASR under specific noise conditions (stationary white noise), they severely degrade downstream Speech Emotion Recognition (SER) models (reducing classification accuracy from 39.29% to 17.86% under white noise at 5dB).
- **Trade-off**: Cleaning the audio signal for verbal content recognition (ASR) vs. preserving acoustic indicators of the speaker's emotional state (SER, pitch, energy dynamics).
- **Final Decision**: **Do NOT apply classical preprocessing globally in multi-task pipelines.** In systems requiring both ASR and SER (e.g., emotion-aware conversational interfaces), the raw audio must either be routed directly to the SER model (bypassing the noise filter), or the pipeline should utilize modern deep feature learning that handles noise natively.