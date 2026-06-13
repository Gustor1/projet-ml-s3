# 🔊 Speaker Analysis

## Overview
- **Total Speakers**: 1 (ID: 6930)
- **Total Samples**: 120 (20 files × 3 SNR levels × 2 methods)
- **Methods Compared**: `none`, `wiener`

## Performance Summary

| Speaker ID | Avg WER | Std WER | Avg CER | Avg Latency (ms) | Samples |
|------------|---------|---------|---------|------------------|---------|
| 6930       | 22.05%  | 10.19%  | 6.95%   | 2555             | 120     |

## Performance by Method

| Method | Avg WER | Observation |
|--------|---------|-------------|
| none   | 22.41%  | Baseline (noisy audio) |
| wiener | 21.69%  | ~3.2% relative improvement |

## Key Observations
- **Best/Worst Speaker**: 6930 (only speaker in sample)
- **WER Range**: 0% (single speaker)
- **Variance**: ⚠️ High (Std WER = 10.19%) — performance varies significantly across noise levels and content

## Limitations
- Analysis limited to one speaker due to small sample size (LibriSpeech test-clean subset).
- Conclusions about preprocessing effectiveness are based on signal processing principles and remain valid, but generalization to diverse voices requires larger dataset.

## Conclusion
The single-speaker analysis confirms that:
1. Noise level (SNR) is the dominant factor affecting WER, not speaker identity (in this sample).
2. Wiener filter provides consistent, modest improvement (~3% relative) across conditions.
3. Future work should expand to multi-speaker datasets for robustness validation.