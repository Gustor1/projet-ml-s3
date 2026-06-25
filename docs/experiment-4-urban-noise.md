# 🧪 Experiment 4: Real-World Urban Noise (Non-Stationary Environments)

## 📚 Theoretical Background & Non-Stationary Noise Dynamics

Synthetic stationary noise types (e.g., white Gaussian or Voss-McCartney pink noise) are useful for controlled laboratory benchmarks but do not model the complexity of real-world environments. Real-world mobile and edge-ASR deployments operate in environments characterized by **non-stationary noise** [1].

### 1. Non-Stationarity Defined
In a stationary acoustic process, the statistical properties (mean, variance, auto-correlation) are constant over time. For non-stationary noise (e.g., urban street sounds, traffic, cafe chatter), the power spectral density ($PSD$) is time-varying:
$$PSD_v(\omega, t) = \mathcal{F}\{R_{vv}(\tau, t)\}$$
where $R_{vv}(\tau, t) = E[v(t)v(t-\tau)]$ is the time-dependent auto-correlation function.

Non-stationary noise is characterized by:
- **Impulsive Events**: Sudden acoustic transients (e.g., car horns, door slams, footsteps) with short duration and high energy.
- **Modulated Backgrounds**: Slowly varying spectral shapes (e.g., passing vehicle engines, crowd noise envelopes).
- **Time-Varying Frequency Banding**: Noise components that shift across frequency bands over time.

### 2. Filter Tracking Lag & Temporal Smearing
Classical DSP speech enhancement algorithms (Wiener filtering and Spectral Subtraction) rely on the assumption of noise stationarity to estimate the noise spectrum $P_{vv}(\omega)$ during silent segments [2]. In non-stationary environments, this assumption is violated, leading to two processing failures:

1. **Tracking Lag**: The estimation algorithm updates the noise PSD using a recursive smoothing equation with a time constant $\alpha_{\text{smooth}}$:
   $$\hat{P}_{vv}(\omega, t) = \alpha_{\text{smooth}} \hat{P}_{vv}(\omega, t-1) + (1-\alpha_{\text{smooth}}) |Y(\omega, t)|^2$$
   When a sudden noise peak (e.g., a car horn) occurs, the filter fails to attenuate it because $\hat{P}_{vv}(\omega, t)$ lags behind the instantaneous noise power. After the transient peak passes, $\hat{P}_{vv}(\omega, t)$ remains artificially high. The filter then over-attenuates the subsequent speech frame, erasing speech segments.
2. **Temporal Smearing & Musical Noise**: The rapid fluctuation of the time-varying gain filter $H(\omega, t)$ introduces amplitude modulation and phase distortions. This causes **temporal smearing** (blurring of phonetic boundaries) and severe **musical noise** (isolated spectral peaks). While these artifacts are tolerable to human listeners, they disrupt the temporal self-attention maps in Whisper's encoder, leading to transcription errors.

---

## 📖 Context & Scientific Motivation
Experiments 2 and 3 established that preprocessing algorithms are highly sensitive to the noise spectrum, showing that Wiener filtering helps on flat white noise but degrades performance on colored pink noise. This experiment evaluates whether these algorithms generalize to real-world non-stationary urban noise (using recordings from the DEMAND database), addressing the **lab-to-real-world gap** in ASR evaluation.

## 🎯 Hypotheses
* **$H_0$ (Null Hypothesis)**: Preprocessing algorithms perform similarly on real-world non-stationary urban noise as on synthetic stationary noise (white/pink).
* **$H_1$ (Alternative Hypothesis)**: Real-world non-stationary noise violates the stationarity assumptions of classical DSP filters, causing tracking lag and spectral distortions that degrade ASR performance compared to the raw noisy baseline.

---

## 🔬 Experimental Protocol

### Dataset & Augmentation
- **Source**: The same 20 LibriSpeech `test-clean` files (Speaker 6930) used in the previous experiments.
- **Urban Noise Source**: Real recordings from the Diverse Environments Multichannel Acoustic Noise Database (DEMAND) [3], capturing traffic, cafe ambiance, and street scenes.
- **SNR Mixing**: Mixed at **20 dB**, **10 dB**, and **5 dB** SNR.
- **Total Inferences**: $60$ runs per method (180 total inferences).

### Processing Methods
1. `none`: No preprocessing (raw urban-noisy audio).
2. `wiener`: Wiener spectral denoising.
3. `spectral_subtraction`: Multi-band spectral subtraction ($\alpha=2.0, \beta=0.01$).

---

## 📊 Empirical Results

### Summary Averages (60 Samples per Method)

| Method | Avg $WER$ | Δ vs. Baseline (`none`) | Observation |
|--------|-----------|------------------------|-------------|
| `none` (Raw) | 22.17% | — | Baseline under urban noise |
| `wiener` | 25.58% | +3.41% ❌ | Degrades ASR accuracy |
| `spectral_subtraction` | 33.45% | +11.28% ❌ | Severely degrades ASR accuracy |

### Results Breakdown by SNR Level

| SNR Level | Method | Avg $WER$ | Avg $CER$ | Δ vs. none | Observation |
|-----------|--------|-----------|-----------|------------|-------------|
| **20 dB** (Low) | `none` | 18.24% | 4.11% | — | Baseline control |
| | `wiener` | 19.18% | 4.89% | +0.94% ❌ | Slight degradation |
| | `spectral_subtraction` | 25.04% | 8.94% | +6.80% ❌ | Degrades performance |
| **10 dB** (Mod) | `none` | 22.12% | 5.84% | — | Baseline control |
| | `wiener` | 22.07% | 6.12% | -0.05% | Neutral |
| | `spectral_subtraction` | 28.40% | 10.11% | +6.29% ❌ | Severe degradation |
| **5 dB** (Severe) | `none` | **26.17%** | **7.44%** | — | Baseline control |
| | `wiener` | **35.48%** | **11.23%** | **+9.31% ❌** | Massive degradation |
| | `spectral_subtraction` | 46.92% | 19.98% | **+20.75% ❌** | Catastrophic degradation |

---

## 🔍 Scientific Discussion & The Lab-to-Real-World Gap

### 1. The Lab-to-Real-World Gap
This experiment demonstrates the gap between laboratory benchmarks and real-world performance. In Experiment 2 (White Gaussian Noise), the Wiener filter improved ASR accuracy under severe noise (**5 dB SNR**), reducing $WER$ by $2.75\%$ absolute. However, under real-world urban noise at the same SNR (**5 dB SNR**), the Wiener filter degraded performance, increasing $WER$ by **9.31% absolute** ($35.48\%$ vs $26.17\%$).

This reversal is caused by the transition from stationary to non-stationary noise. White Gaussian noise is stationary and has a flat spectrum, which matches the assumptions of the Wiener filter. Urban noise contains transient events (e.g., vehicle horns, footsteps, sudden voices) that violate these assumptions. This confirms the alternative hypothesis ($H_1$): classical DSP algorithms optimized for stationary noise fail to generalize to real-world acoustic scenes.

### 2. Analysis of Tracking Lag and Distortion
The degradation under urban noise is caused by the filter's noise estimation tracking lag. When a transient vehicle horn occurs:
1. The noise PSD estimate $\hat{P}_{vv}(\omega, t)$ lags behind the instantaneous noise power. The filter fails to attenuate the horn, allowing it to corrupt the speech signal.
2. After the vehicle horn passes, the noise PSD estimate remains artificially high. The filter continues to apply high attenuation to the subsequent frames. If the speaker begins talking immediately after the horn, the filter attenuates the initial consonants and vowels of the speech.
3. This attenuation distorts the spectral envelope, smoothing out high-frequency phonetic transitions. Whisper's encoder, which relies on these temporal transitions to extract acoustic features, is unable to correctly identify the phonemes, leading to word deletions and substitutions.

### 3. Spectral Subtraction Failures
Spectral subtraction performed poorly across all SNR levels, with $WER$ reaching **46.92%** at 5 dB SNR. The non-stationarity of urban noise exacerbates the musical noise and spectral holes associated with spectral subtraction. The aggressive over-subtraction factor ($\alpha=2.0$) removes significant portions of the target speech during transient noise events, leaving a fragmented signal that confuses Whisper's autoregressive decoder.

## ⚖️ Engineering Recommendation
1. **Do Not Deploy Classical Preprocessing in Urban Environments**: Wiener filtering and spectral subtraction should not be used in mobile or edge-ASR systems operating in urban environments.
2. **Use Machine Learning-Based Enhancement**: Pipelines operating in non-stationary environments should utilize deep learning-based speech enhancement models (e.g., DCCRN or Conv-TasNet). These models are trained on real-world noise datasets and learn to model the temporal structure of speech, avoiding the tracking lag associated with classical DSP filters.
3. **Optimize End-to-End**: Rather than using a standalone denoising model, ASR systems should be trained or fine-tuned directly on noisy data, allowing the model's encoder to learn noise-robust representations natively.

## 📚 References
* [1] E. Vincent et al., "The CHiME speech separation and recognition challenges: An overview," *Computer Speech & Language*, vol. 46, pp. 287–308, 2017.
* [2] A. V. Oppenheim and J. S. Lim, "The importance of phase in signals," *Proceedings of the IEEE*, vol. 69, no. 5, pp. 529–541, 1981.
* [3] J. Thiemann, N. Ito, and E. Vincent, "The Diverse Environments Multichannel Acoustic Noise Database (DEMAND): A database of multichannel environmental noise recordings," *Proceedings of the Meetings on Acoustics*, 2013.
* [4] C. Evans, J. S. Mason, and W. M. Campbell, "On the Fundamental Limitations of Spectral Subtraction," *Proceedings of the European Signal Processing Conference (EUSIPCO)*, 2005.
