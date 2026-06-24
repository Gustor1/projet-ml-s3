# 🧪 Experiment 2: Comprehensive Preprocessing Evaluation (White Gaussian Noise)

## 📚 Theoretical Background & DSP Formulations

In local speech recognition systems, preprocessing is often introduced to improve robustness to environmental noise. This experiment evaluates two classical frequency-domain speech enhancement algorithms: Spectral Subtraction and Wiener Filtering.

### 1. Spectral Subtraction (Boll, 1979)
Spectral subtraction assumes that the noise is additive and stationary [1]. Let the noisy signal be $y(n) = s(n) + v(n)$, where $s(n)$ is the target speech and $v(n)$ is the noise. In the Short-Time Fourier Transform (STFT) domain:
$$Y(\omega, t) = S(\omega, t) + V(\omega, t)$$

The noise spectrum is estimated during silent (non-speech) segments, yielding an average noise magnitude spectrum $|\hat{V}(\omega)|$. The speech magnitude spectrum is estimated by subtracting the noise estimate:
$$|\hat{S}(\omega, t)|^b = |Y(\omega, t)|^b - \alpha |\hat{V}(\omega)|^b$$
where $b$ determines the subtraction domain ($b=1$ for magnitude subtraction, $b=2$ for power spectral subtraction), and $\alpha \ge 1$ is the over-subtraction factor used to compensate for noise variance.

To prevent negative spectral magnitudes, the half-wave rectified output is bounded by a spectral floor:
$$|\hat{S}(\omega, t)|^b = \max\left( |Y(\omega, t)|^b - \alpha |\hat{V}(\omega)|^b, \, \beta |Y(\omega, t)|^b \right)$$
where $\beta \in [0, 1]$ is the spectral floor parameter (set to $0.01$ in our implementation). The enhanced time-domain signal $\hat{s}(n)$ is reconstructed via the Inverse Short-Time Fourier Transform (ISTFT) using the phase of the original noisy signal $\angle Y(\omega, t)$.

### 2. Wiener Filtering (Lim & Oppenheim, 1978)
The Wiener filter is the optimal linear estimator under the Minimum Mean Square Error (MMSE) criterion [2]. In the frequency domain, the Wiener filter transfer function $H(\omega, t)$ is defined as:
$$H(\omega, t) = \frac{P_{ss}(\omega, t)}{P_{ss}(\omega, t) + P_{vv}(\omega, t)}$$
where $P_{ss}(\omega, t)$ and $P_{vv}(\omega, t)$ are the power spectral densities (PSD) of the clean speech and noise, respectively. Expressed in terms of the a priori SNR, $\xi(\omega, t) = P_{ss}(\omega, t) / P_{vv}(\omega, t)$:
$$H(\omega, t) = \frac{\xi(\omega, t)}{1 + \xi(\omega, t)}$$

In practice, $\xi(\omega, t)$ is estimated iteratively using the decision-directed approach (Ephraim & Malah, 1984), which averages the past enhanced signal frame and the current noisy SNR to ensure temporal smoothness.

## 📖 Context & Scientific Objective
The objective of this experiment is to measure the degradation curve of the Whisper-tiny model under additive White Gaussian Noise ($WGN$), and to evaluate whether classical DSP algorithms ($Wiener$ and $Spectral\ Subtraction$) improve ASR performance (reducing $WER$ and $CER$) or introduce harmful acoustic distortions.

## 🎯 Hypotheses

* **$H_0$ (Null Hypothesis)**: Classical speech enhancement filters (Wiener, Spectral Subtraction) improve ASR Word Error Rate ($WER$) under all SNR conditions when applied as an upstream preprocessing step.
* **$H_1$ (Alternative — Conditional Gain)**: The benefit of classical filters is SNR-conditional. Wiener filtering only improves $WER$ under *severe* noise levels (5 dB SNR), where noise suppression outweighs spectral distortion costs. Under mild to moderate noise (20–10 dB SNR), the spectral artifacts introduced by the filter degrade model accuracy.
* **$H_2$ (Spectral Subtraction)**: Spectral subtraction will degrade performance under all SNR conditions due to the introduction of spectral holes and musical noise artifacts that Whisper's self-attention layers misinterpret as phonemic structures.

## 🔬 Experimental Protocol

### Dataset & Noise Augmentation
- **Source**: The same 20 LibriSpeech `test-clean` files (Speaker 6930) established in Experiment 1.
- **Noise Injection**: Synthetic additive White Gaussian Noise ($WGN$), which has a flat power spectral density ($PSD_v(\omega) = \sigma^2$), generated at three Signal-to-Noise Ratio ($SNR$) levels:
  - **20 dB SNR**: Low noise (quiet room environment).
  - **10 dB SNR**: Moderate noise (standard office/cafe background).
  - **5 dB SNR**: Severe noise (extreme factory/industrial noise).
- **Total Inferences**: $180$ runs (20 files $\times$ 3 SNR levels $\times$ 3 processing methods).

### Processing Methods
1. `none`: No preprocessing (raw noisy audio fed directly to Whisper).
2. `wiener`: Wiener spectral denoising.
3. `spectral_subtraction`: Multi-band spectral subtraction ($\alpha=2.0, \beta=0.01$).

---

## 🛠️ FFT Boundary Bug Analysis (Phase 1 & 2)

During the initial execution, the `spectral_subtraction` pipeline crashed on 100% of the test samples with a `ValueError: operands could not be broadcast together`.

### Mathematical and Array-Level Root Cause
The bug occurred during the overlap-add (OLA) reconstruction stage of the custom spectral subtraction algorithm. The audio signal of total length $N$ was processed in short frames of length $L$ (the FFT window size, $L=2048$) with a hop size $R=512$.

For any frame starting at index $i$, the inverse FFT reconstructed a time-domain frame $x_{\text{rec}}$ of length $L$. The algorithm added this reconstructed frame back to the output buffer:
$$\text{output}[i : i + L] = \text{output}[i : i + L] + x_{\text{rec}}$$

At the boundary of the final frame, the remaining samples in the buffer were less than the FFT window size ($N - i < L$). The slice `output[i : i + L]` attempted to access indices beyond the array bounds, causing an array shape mismatch error:
$$\text{shape of } \text{output}[i : i + L] = N - i \quad \neq \quad \text{shape of } x_{\text{rec}} = L$$

### Code Modification & Resolution
We resolved this issue by implementing safe boundary checking, which truncates the reconstructed frame to match the remaining audio length at the boundary:
```python
# Before (crashed on last frame)
result[i : i + nfft] += clean_frame

# After (safe boundary handling)
chunk_len = min(nfft, n - i)
result[i : i + chunk_len] += clean_frame[:chunk_len]
```
The entire experiment was re-run from scratch after this fix to ensure consistency in CPU execution metrics.

---

## 📊 Phase 3 — Final Empirical Results

### Summary Averages

| Method | Avg $WER$ | Avg $CER$ | Avg Latency ($\tau$) | Observation |
|--------|-----------|-----------|---------------------|-------------|
| `none` (Raw) | 22.41% | 6.78% | 3,227 ms | Baseline under noise |
| `wiener` | 21.69% | 7.11% | 3,153 ms | Modest gain (~3.2% relative) |
| `spectral_subtraction` | 31.84% | 12.68% | 3,168 ms | ❌ Catastrophic degradation |

### Results Breakdown by SNR Level

| SNR Level | Method | Avg $WER$ | Avg $CER$ | Avg Latency ($\tau$) | Observation |
|-----------|--------|-----------|-----------|---------------------|-------------|
| **20 dB** (Low) | `none` | 18.94% | 4.35% | 3,385 ms | Baseline control |
| | `wiener` | 18.79% | 4.97% | 3,357 ms | Neutral |
| | `spectral_subtraction` | 25.73% | 9.10% | 3,365 ms | ❌ Degrades performance |
| **10 dB** (Mod) | `none` | 20.81% | 6.14% | 3,297 ms | Baseline control |
| | `wiener` | 21.57% | 7.17% | 3,196 ms | Slight degradation |
| | `spectral_subtraction` | 27.67% | 10.64% | 3,081 ms | ❌ Degrades performance |
| **5 dB** (Severe) | `none` | 27.47% | 9.86% | 2,998 ms | Baseline control |
| | `wiener` | **24.72%** | **9.20%** | **2,906 ms** | ✅ Best method (10% relative gain) |
| | `spectral_subtraction` | 42.11% | 18.29% | 3,058 ms | ❌ Worst method |

---

## 🔍 Scientific Discussion & Engineering Trade-offs

### 1. The "Goldilocks Zone" of Wiener Filtering
The results show that the Wiener filter is not a universal solution. It only provides an accuracy improvement under severe noise conditions (**5 dB SNR**, $WER$ drops from $27.47\%$ to $24.72\%$). At mild noise levels (**20 dB SNR**), the filter is neutral ($18.79\%$ vs $18.94\%$), and at **10 dB SNR**, it slightly increases $WER$.

This behavior is explained by the trade-off between **noise suppression** and **speech distortion**. In quiet conditions, the noise power $P_{vv}(\omega)$ is small, meaning the Wiener gain $H(\omega, t) \approx 1$. However, minor estimation errors in $P_{vv}(\omega)$ lead to phase shifts and amplitude smoothing. Whisper's transformer encoder, which relies on clean Mel-filterbank representations, is sensitive to these phase and amplitude alterations. Under mild noise, the damage from these spectral artifacts outweighs the benefit of noise reduction. At 5 dB SNR, the raw noise is so severe that removing it, even at the cost of minor phase distortion, provides a net benefit to the model.

### 2. Why Spectral Subtraction Degrades Performance
Spectral subtraction catastrophically degraded $WER$ across all SNR levels (rising to **42.11%** at 5 dB SNR, a $14.64\%$ absolute increase over baseline). This is a well-documented phenomenon in traditional speech processing, which Evans et al. (2005) categorized into three fundamental error types [3]:
1. **Magnitude Errors**: Subtracting the average noise spectrum $|\hat{V}(\omega)|$ from the varying instantaneous spectrum $|Y(\omega, t)|$ creates negative values. The half-wave rectification step ($\max$ function) sets these bins to the spectral floor. This creates **spectral holes** (zeroed frequency bins), which destroy speech formants.
2. **Phase Errors**: Spectral subtraction retains the noisy phase $\angle Y(\omega, t)$ to reconstruct the time-domain signal. Under low SNR, the phase is highly corrupted, causing severe perceptual distortion.
3. **Cross-Term Errors**: The assumption that speech and noise phase cross-terms are zero ($E[S(\omega)V^*(\omega)] = 0$) holds only over infinite time. In short analysis windows ($25\text{ ms}$), these cross-terms are non-zero, leading to random spectral peaks known as **musical noise** (isolated sinusoidal tones).

While the human auditory system can filter out musical noise, the autoregressive decoder in Whisper tiny is easily confused by these artificial spectral peaks. The model interprets musical noise as phonemes, leading to word insertions and substitutions.

### 3. Latency Dynamics
Denoising slightly reduces Whisper's latency under severe noise (e.g., at 5 dB, `wiener` latency is $2,906\text{ ms}$ vs $2,998\text{ ms}$ for `none`). This occurs because cleaner input signals increase the decoder's token confidence, which reduces the number of autoregressive beams and decoding iterations required to reach the end-of-sequence token. However, this minor latency reduction is offset by the DSP pipeline overhead.

## ⚖️ Engineering Recommendation

1. **Discard Spectral Subtraction**: It introduces severe spectral holes and musical noise that degrade transformer-based ASR performance.
2. **Conditional Denoising Only**: Classical filters should not be always-on. They should only be activated when estimated background noise exceeds a specific threshold (e.g., $SNR < 10\text{ dB}$).
3. **Explore Deep Learning Alternatives**: Modern deep source separation networks (such as DCCRN or Conv-TasNet) should be investigated as alternatives to classical DSP, as they learn to model speech structure and avoid the artifacts associated with spectral subtraction.

## 📚 References
* [1] S. Boll, "Suppression of acoustic noise in speech using spectral subtraction," *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 27, no. 2, pp. 113–120, 1979.
* [2] J. S. Lim and A. V. Oppenheim, "All-pole modeling of degraded speech," *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 26, no. 3, pp. 197–210, 1978.
* [3] C. Evans, J. S. Mason, and W. M. Campbell, "On the Fundamental Limitations of Spectral Subtraction," *Proceedings of the European Signal Processing Conference (EUSIPCO)*, 2005.
