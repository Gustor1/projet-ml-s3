# Experiment 2: Preprocessing Comparison (None vs Wiener Filter)

## Objective
Investigate whether a simple spectral denoising algorithm (Wiener Filter) improves ASR recognition accuracy in noisy environments compared to raw noisy input.

## Methodology
- **Dataset**: 20 LibriSpeech test-clean files augmented with white Gaussian noise at three levels:
  - **20dB SNR**: Low noise (clean environment)
  - **10dB SNR**: Moderate noise (busy street)
  - **5dB SNR**: High noise (noisy factory/crowd)
- **ASR Model**: Whisper tiny (CPU inference)
- **Methods Compared**:
  1. `none`: Direct ASR on noisy audio (Baseline)
  2. `wiener`: Audio pre-processed with SciPy's Wiener filter
- **Metrics**: Word Error Rate (WER) and Inference Latency (ms)

## Results Summary
Based on 60 valid comparisons (20 files × 3 SNR levels). The `spectral_subtraction` method was attempted but failed due to implementation errors (FFT shape mismatch) and is excluded from final analysis.

| SNR Level | Method | Average WER | Observation |
|-----------|--------|-------------|-------------|
| **20dB** (Low Noise) | none | ~19% | Baseline |
| 20dB | wiener | ~19% | No significant gain; slight degradation in some cases. |
| **10dB** (Moderate) | none | ~20% | Degradation begins |
| 10dB | wiener | ~21% | Filter introduces artifacts; no improvement. |
| **5dB** (High Noise) | none | ~27% | Significant performance drop |
| 5dB | wiener | ~24% | **Clear improvement** (~11% relative gain). |

## Key Insight
The Wiener filter is **not a silver bullet**. It only improves performance significantly at very low SNR (5dB). At higher SNR levels (10-20dB), the filter introduces slight spectral artifacts that confuse the ASR model, leading to neutral or worse performance.

## Engineering Trade-off
- **Benefit**: In extreme noise (5dB), we recover ~3-4% absolute WER, which is significant for intelligibility.
- **Cost**: The filter adds computational overhead (~100-200ms) and risks degrading clean speech (at 20dB).
- **Conclusion**: A "always-on" preprocessing pipeline is inefficient. An adaptive approach (activating the filter only when estimated noise > 10dB) would be the optimal engineering solution.

## Visual Evidence
See `visuals/preprocessing_comparison.png`. The chart clearly shows the divergence at 5dB where `wiener` outperforms `none`, while bars merge at 20dB.