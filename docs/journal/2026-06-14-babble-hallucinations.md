# 📅 2026-06-14 — Discovery of ASR Hallucinations (Experiment 5)

## 🎯 Objective
Evaluate the impact of preprocessing on "babble" noise (crowd), the most realistic and challenging scenario (Cocktail Party Problem).

## 🚨 Anomaly Detected
During the raw analysis of `results/babble_noise_comparison.csv`, 6 samples (all at 5dB SNR) exhibited **WER > 1.0 (100%)**, reaching up to 59.33 (i.e., 5933%).

## 🧠 Analysis & Decision
- **Hypothesis**: This is not a simple recognition error, but **ASR hallucinations**. Faced with an ambiguous acoustic signal (target speech drowned in other speech), the Whisper tiny decoder "invents" fluent sentences that are completely disconnected from the reference.
- **Corrective Action**: Rather than biasing the averages, we developed `scripts/analyze_babble_robust.py` to exclude these outliers (WER ≥ 1.0) and compute robust statistics on the 174/180 valid samples.

## ✅ Result
- Confirmation that classical filtering (Wiener/Spectral) catastrophically worsens performance on babble (+8.12% and +18.49% degradation at 5dB).
- Addition of Insight 10 in `docs/insights.md` on this critical phenomenon for real-world deployment.