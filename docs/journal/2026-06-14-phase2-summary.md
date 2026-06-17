# 2026-06-14 — Phase 2: Debug & Final Results

## What Was Done
- Identified & fixed FFT boundary bug in `preprocess_spectral_subtraction()`
- Re-ran full experiment (180 inferences) with all 3 methods functional
- Added CER metric to all results
- Updated documentation with final, synchronized numbers

## Key Finding
Wiener filter helps at 5dB SNR (~10% relative WER gain) but spectral subtraction degrades performance everywhere (+9.4% absolute WER). This proves: *method selection matters more than "any preprocessing is better"*.

## Reproducibility
- All seeds fixed: `np.random.seed(42)`
- Commands documented in each script header
- LibriSpeech test-clean (20 files) publicly available
- Full CSV: `results/preprocessing_comparison.csv`

## Next
- Phase 3: Final packaging + handoff to video team