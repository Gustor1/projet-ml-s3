# 💡 Curated Insights & Engineering Trade-offs

##  Insight 1: The "Noise Floor" Effect
**Observation**: Whisper tiny struggles significantly as noise increases.
- At 20dB SNR: WER ~19%
- At 5dB SNR: WER ~27%
**Takeaway**: Without preprocessing, the model's performance degrades linearly with noise. This confirms the need for an upstream cleaning stage in real-world mobile scenarios.

##  Insight 2: Preprocessing is Context-Dependent (The "Goldilocks" Zone)
**Observation**: The Wiener filter helps at 5dB (WER drops to ~24%) but is useless or harmful at 20dB.
**Trade-off**: Applying a filter to already decent audio can distort the signal (phase shifts, musical noise), confusing the ASR.
**Decision**: We should not apply preprocessing blindly. A Voice Activity Detector (VAD) or Noise Estimator should trigger the filter only when noise is severe.

## 🔍 Insight 3: Latency vs. Accuracy Balance
**Observation**: Adding the Wiener filter increases processing time by ~150-200ms per file.
**Impact**: In a "real-time" mobile app, every millisecond counts.
**Recommendation**: The accuracy gain at 5dB (recovering intelligibility) justifies the latency cost. However, at 20dB, the latency cost is wasted since accuracy doesn't improve.

##  Video Script Snippets (English)
- *"We found that preprocessing isn't magic. In fact, applying a noise filter to clean audio can actually make things worse by introducing artifacts."*
- *"The data shows a clear threshold: our Wiener filter only shines when the noise is loud (5dB). Below that, it's better to let the raw audio pass through."*
- *"This suggests that the future of local ASR isn't just 'better filters', but 'smarter switching'—activating processing only when the environment demands it."*