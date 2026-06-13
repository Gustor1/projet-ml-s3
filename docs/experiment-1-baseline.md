# Experiment 1: Baseline ASR (Whisper tiny on clean audio)

## Objective
Establish a WER reference on LibriSpeech test-clean without any preprocessing.

## Methodology
- **Model**: Whisper tiny (39M parameters, CPU)
- **Dataset**: 20 files LibriSpeech test-clean (clean English speech)
- **Metric**: Word Error Rate (WER) via jiwer
- **Measurement**: Inference latency per file

## Results
| Metric | Value |
|--------|-------|
| Average WER | 18.60% |
| Average Latency | 1894 ms/file |
| Files processed | 20/20 |

## Main Insight
Whisper tiny, although fast to load (~150MB), produces a relatively high WER (18.60%) even on clean audio. This result serves as a **baseline reference**: any preprocessing that reduces this WER will be considered beneficial.

## Trade-off Identified
- **Advantage**: Very lightweight model, easy deployment
- **Cost**: High latency (1.9s) for "real-time" and WER could be better
- **Hypothesis**: A larger model (base/small) would reduce WER but increase latency -> to test

## Next Step
Add artificial noise (various SNRs) then test the impact of preprocessing methods (Wiener, noisereduce) on WER.