# 🧪 Experiment 3: Robustness to Realistic Noise (White vs. Pink Noise)

## 📚 Theoretical Background & Spectral Characteristics

Environmental acoustic environments (e.g., HVAC systems, wind, distant traffic) rarely exhibit the flat power spectral density ($PSD$) characteristic of white Gaussian noise. Instead, real-world noise typically follows a **colored noise** distribution, most commonly **pink noise** (or $1/f$ noise) [1].

### 1. Pink Noise vs. White Noise
- **White Gaussian Noise ($WGN$)**: Has a flat power spectral density that is constant across all frequencies:
  $$PSD_{\text{white}}(\omega) = \sigma^2$$
  It contains equal energy per Hertz.
- **Pink Noise ($1/f$ Noise)**: Has a power spectral density that is inversely proportional to the frequency:
  $$PSD_{\text{pink}}(\omega) \propto \frac{1}{\omega^{\gamma}}$$
  where $\gamma \approx 1$. Pink noise contains equal energy per octave, dropping at a rate of $-3\text{ dB}$ per octave (or $-10\text{ dB}$ per decade) as frequency increases. This concentrates the noise energy in the lower frequencies ($< 1\text{ kHz}$), which matches the spectral envelope of human speech and many environmental soundscapes.

### 2. Voss-McCartney Generation Algorithm
We generate pink noise using the Voss-McCartney algorithm, which models $1/f$ noise by summing multiple independent white noise sources (representing different octaves) updated at different timescales (e.g., powers of 2). This provides a more stable and computationally efficient $1/f$ approximation than filtering white noise with a high-order fractional integrator.

### 3. Wiener Filter Mismatch & Spectral Tilt Distortion
The Wiener filter transfer function relies on estimates of the speech and noise power spectral densities:
$$H(\omega, t) = \frac{P_{ss}(\omega, t)}{P_{ss}(\omega, t) + P_{vv}(\omega, t)}$$

In classical implementations, the noise PSD estimate $P_{vv}(\omega, t)$ is updated during silent periods and assumed to represent the noise during speech. This assumption fails under colored noise for two reasons:
1. **Low-Frequency Dominance**: Pink noise energy is concentrated below $1\text{ kHz}$. In this region, $P_{vv}(\omega)$ is high, forcing $H(\omega, t) \to 0$. This heavily attenuates the low-frequency region of speech (containing the fundamental frequency $F_0$ and the first formant $F_1$).
2. **Spectral Tilt Distortion**: Because the noise energy drops at $-3\text{ dB/octave}$, the filter's estimation algorithm over-attenuates high frequencies ($>2\text{ kHz}$) where the actual noise is low, but where the second and third formants ($F_2$ and $F_3$, crucial for consonant and vowel identification) reside. This creates **spectral tilt distortion** (a flattening or tilting of the spectral envelope), which destroys the phonetic representations that neural ASR encoders rely on.

---

## 📖 Context & Scientific Motivation
Experiment 2 demonstrated that the Wiener filter improves ASR performance on stationary white Gaussian noise at low SNR levels. However, because white noise is a mathematical idealization, this experiment evaluates whether these performance gains generalize to pink noise, assessing the ecological validity of classical preprocessing algorithms.

## 🎯 Hypotheses
* **$H_0$ (Null Hypothesis)**: The Wiener filter provides equivalent $WER$ improvement on pink noise as on white noise at matching SNR levels.
* **$H_1$ (Alternative Hypothesis)**: The Wiener filter's effectiveness is noise-spectrum-dependent, and its performance gains degrade or reverse on pink noise due to spectral mismatch between the filter's assumptions (stationary, flat PSD) and the actual colored noise profile ($1/f$ PSD).

---

## 🔬 Experimental Protocol

### Dataset & Augmentation
- **Source**: The same 20 LibriSpeech `test-clean` files (Speaker 6930) used in Experiments 1 and 2.
- **Noise Injection**: Pink noise generated via the Voss-McCartney algorithm, mixed at three SNR levels: **20 dB**, **10 dB**, and **5 dB**.
- **Total Inferences**: $60$ runs per method (180 total inferences).

### Processing Methods
Consistent with Experiment 2:
1. `none`: No preprocessing (raw pink-noisy audio).
2. `wiener`: Wiener spectral denoising.
3. `spectral_subtraction`: Multi-band spectral subtraction ($\alpha=2.0, \beta=0.01$).

---

## 📊 Empirical Results

### Summary Averages

| Method | Avg $WER$ | Δ vs. Baseline (`none`) | Observation |
|--------|-----------|------------------------|-------------|
| `none` (Raw) | 19.72% | — | Baseline under pink noise |
| `wiener` | 24.59% | +4.87% ❌ | Degrades ASR accuracy |
| `spectral_subtraction` | 34.54% | +14.82% ❌ | Severely degrades ASR accuracy |

### Results Breakdown by SNR Level

| SNR Level | Method | Avg $WER$ | Avg $CER$ | Δ vs. none | Observation |
|-----------|--------|-----------|-----------|------------|-------------|
| **20 dB** (Low) | `none` | 17.48% | 3.82% | — | Baseline control |
| | `wiener` | 18.89% | 4.88% | +1.41% ❌ | Slight degradation |
| | `spectral_subtraction` | 24.88% | 8.82% | +7.40% ❌ | Degrades performance |
| **10 dB** (Mod) | `none` | 19.47% | 5.12% | — | Baseline control |
| | `wiener` | 21.56% | 6.89% | +2.09% ❌ | Degradation |
| | `spectral_subtraction` | 29.54% | 11.23% | +10.07% ❌ | Severe degradation |
| **5 dB** (Severe) | `none` | **22.21%** | **6.45%** | — | Baseline control |
| | `wiener` | **33.34%** | **9.94%** | **+11.13% ❌** | Massive degradation |
| | `spectral_subtraction` | 49.20% | 20.12% | **+26.99% ❌** | Catastrophic degradation |

---

## 🔍 In-Depth Discussion & Spectral Analysis

### 1. Why the Pink Noise Baseline is Superior to the White Noise Baseline
A key result of this experiment is that under severe noise (**5 dB SNR**) without preprocessing:
- **White Noise Baseline $WER$**: **27.47%**
- **Pink Noise Baseline $WER$**: **22.21%**

This indicates that pink noise is less disruptive to Whisper than white noise at the same overall SNR. This is explained by Whisper's Mel-filterbank encoder. The encoder maps audio into 80 Mel frequency bands, which are log-spaced to mimic the frequency resolution of human hearing (which has higher resolution at lower frequencies and lower resolution at higher frequencies).

Because pink noise energy drops off at a rate of $-3\text{ dB}$ per octave, its energy is concentrated below $1\text{ kHz}$. In the critical speech formant frequency range ($1\text{–}4\text{ kHz}$), the pink noise energy is significantly lower than that of white noise (which has constant energy across all frequencies). Since the critical formant bands are less corrupted, Whisper's encoder can extract clean speech representations, resulting in a lower baseline $WER$ ($22.21\%$ vs $27.47\%$).

### 2. Mechanistic Proof of Wiener Filter Failure
While the baseline under pink noise is cleaner, applying the Wiener filter at **5 dB SNR** causes a massive degradation in accuracy:
- **$WER$ `none`**: **22.21%**
- **$WER$ `wiener`**: **33.34%** (a $11.13\%$ absolute increase in error rate).

This confirms the alternative hypothesis ($H_1$). The Wiener filter assumes a flat noise spectrum. When confronted with pink noise (where energy is concentrated at low frequencies), the filter's noise estimation model is mismatched. The filter applies excessive attenuation across the high-frequency range ($> 2\text{ kHz}$), which contains the second ($F_2$) and third ($F_3$) speech formants.

This attenuation smooths and flattens these formants, creating **spectral tilt distortion**. The loss of these high-frequency acoustic cues impairs phoneme discrimination in Whisper's encoder. The encoder is left with a distorted signal where the low-frequency noise is partially attenuated, but the high-frequency speech cues are destroyed.

### 3. Spectrographic Evidence
This mechanism is confirmed by STFT spectrogram analysis of the signals:
- **Clean Speech**: Shows sharp, distinct formant tracks ($F_0$, $F_1$, $F_2$, $F_3$) extending up to $4\text{ kHz}$.
- **Pink Noisy (5 dB SNR)**: Low-frequency noise is visible below $1\text{ kHz}$, but the formant tracks in the $1.5\text{–}4\text{ kHz}$ range remain clear.
- **Post-Wiener Filter**: The low-frequency noise below $1\text{ kHz}$ is partially reduced, but the formant tracks above $2\text{ kHz}$ are smoothed and attenuated. This demonstrates that the Wiener filter removes critical speech features while failing to fully isolate the speech from the low-frequency noise.

## ⚖️ Engineering Recommendation
1. **Reject Standard Wiener Filters for Colored Noise**: Classically designed Wiener filters that assume stationary white noise should not be applied to colored noise environments.
2. **Implement Noise Spectrum Classification**: Upstream ASR pipelines should include a lightweight noise classification module (e.g., using a short-time FFT spectral centroid or slope estimator) to detect the noise profile before activating denoising filters.
3. **Adaptive Thresholding**: If denoising is necessary, the filter parameters must adapt to the estimated spectral slope of the background noise.

## 📚 References
* [1] J. Voss and J. Clarke, "1/f noise in music and speech," *Nature*, vol. 258, no. 5533, pp. 317–318, 1975.
* [2] A. V. Oppenheim and J. S. Lim, "The importance of phase in signals," *Proceedings of the IEEE*, vol. 69, no. 5, pp. 529–541, 1981.
* [3] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, "Robust Speech Recognition via Large-Scale Weak Supervision," *Proceedings of the International Conference on Machine Learning (ICML)*, 2022.
