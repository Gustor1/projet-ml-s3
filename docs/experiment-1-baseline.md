# 🧪 Experiment 1: Baseline ASR Performance (Clean Audio)

## 📚 Related Work & Theoretical Background

### Neural Automatic Speech Recognition (ASR)
Radford et al. (2022) introduced Whisper, a sequence-to-sequence transformer model trained on 680,000 hours of weakly supervised multilingual web audio [1]. Traditional ASR models (e.g., Kaldi or DeepSpeech) are heavily optimized on clean academic speech datasets like LibriSpeech [2], resulting in high sensitivity to out-of-domain acoustics and noise. Whisper's massive weakly supervised pre-training regime enables robust generalization across accents and environmental conditions without specialized domain adaptation.

### Whisper Architecture & Model Selection Rationale
The core architecture consists of an encoder-decoder transformer. The audio is resampled to 16 kHz and converted into an 80-channel log-magnitude Mel-spectrogram via a 25-millisecond window and 10-millisecond hop size. This feature map is processed by two convolutional layers and a standard transformer encoder. The transformer decoder autoregressively generates text tokens using cross-attention over the encoder's hidden representations.

For this research project, **`openai/whisper-tiny` (39M parameters, 4 layers, 6 attention heads)** was selected as the target model. This choice is scientifically and engineering-justified as follows:

1. **Edge-Compute Emulation**: Edge and mobile devices are highly constrained in RAM, storage, and thermal budgets. The `tiny` model (~150 MB footprint) runs efficiently on mobile CPUs, serving as a realistic proxy for local, offline deployments.
2. **Diagnostic Accessibility (Gradients & Attention States)**: By using the standard Hugging Face/PyTorch implementation of Whisper, we maintain direct programmatic access to the internal cross-attention weight matrices, hidden representations, and token-level log-probabilities. This is essential for calculating decoder perplexity ($PPL$) and tracking the mechanics of model hallucinations (see Experiment 5).
3. **Exclusion of Alternatives**:
   * **`Faster-Whisper`**: While `Faster-Whisper` (using CTranslate2) is highly optimized for deployment speed and memory consumption, it acts as a compiled C++ black box. Extracting raw PyTorch tensor states, token-level gradients, or custom cross-attention scores is either impossible or requires modifying the underlying native library, which limits diagnostic research.
   * **`WhisperX`**: WhisperX incorporates a Voice Activity Detection (VAD) model (PyAnnote) and a forced phoneme alignment model (Wav2Vec2). Although superior for caption synchronization, it is a complex multi-model pipeline. Using it would make it impossible to isolate the *intrinsic* response of Whisper's autoregressive decoder to raw, noisy, or classically preprocessed speech signals.

## 📖 Context & Scientific Objective
Before evaluating the impact of audio preprocessing in noisy environments, it is critical to establish a **ground-truth baseline**. The scientific objective of this experiment is to measure the baseline performance limits of the Whisper-tiny model on clean, uncorrupted speech. This serves as the control group ($G_0$) against which all noisy and preprocessed conditions (Experiments 2–5) will be compared.

## 🎯 Hypotheses

* **$H_1$ (Accuracy Baseline)**: Under clean conditions, the Whisper-tiny model will achieve a Word Error Rate ($WER$) $< 20\%$ for Speaker 6930, accounting for speaker-specific complexity and our strict, unnormalized text comparison (retaining punctuation and casing). This is expected to be higher than the official LibriSpeech benchmark (~7.6% $WER$) due to the lack of aggressive text normalization.
* **$H_2$ (Latency Bottleneck)**: Despite its lightweight parameter count (39M), local CPU inference latency will exceed $1.5\text{ seconds}$ per audio sample (averaging $\sim 3\text{ seconds}$ of speech) due to the sequential token generation loop of the autoregressive transformer decoder.

## 🔬 Experimental Protocol

### Dataset & Audio Specifications
- **Corpus**: LibriSpeech `test-clean` subset [2].
- **Subset Selection**: 20 files from Speaker ID: 6930. A single speaker was chosen to eliminate speaker-dependent acoustic variances (pitch, accent, speaking rate) when isolating the impact of SNR and DSP filters.
- **Audio Format**: Native 16 kHz mono WAV (PCM 16-bit).

### Metrics & Formulations
We evaluate the system using three primary metrics:

1. **Word Error Rate ($WER$)**:
   $$WER = \frac{S + D + I}{N} \times 100\%$$
   where $S$ is the number of substitutions, $D$ is deletions, $I$ is insertions, and $N$ is the total number of words in the ground-truth reference transcript.
2. **Character Error Rate ($CER$)**:
   $$CER = \frac{S_c + D_c + I_c}{N_c} \times 100\%$$
   computed at the individual character level (including spacing and punctuation) to assess structural orthographic precision.
3. **Inference Latency ($\tau$)**:
   $$\tau = T_{\text{end}} - T_{\text{start}} \quad (\text{measured in milliseconds})$$
   representing the wall-clock time required to load the audio tensor, run the encoder forward pass, and execute the autoregressive decoding loop on CPU.

### Execution
The baseline execution is run via:
```bash
python experiments/baseline_wer.py
```
Results are saved to `results/baseline.csv`.

## 📊 Results

| Metric | Value | Statistical Observation |
|--------|-------|-------------|
| **Average $WER$** | **18.60%** | Baseline error rate under clean conditions |
| **Average $CER$** | **4.45%** | Fine-grained character deviation ($CER \approx 0.24 \times WER$) |
| **Average Latency ($\tau$)** | **1,894 ms** | Mean CPU execution time per file |
| **Success Rate** | **100% (20/20)** | No computational failures or out-of-memory states |

## 🔍 In-Depth Analysis & Variance

### Deviation from Official Benchmarks
Our empirical $WER$ of 18.60% deviates significantly from the official Whisper-tiny benchmark of 7.6% on the LibriSpeech test-clean dataset. This deviation is not an anomaly but is explained by two design choices:
1. **Unnormalized Text Comparison**: Official evaluations apply an aggressive text normalizer that converts words to lowercase, expands numbers (e.g., "19" $\rightarrow$ "nineteen"), and strips all punctuation. Our pipeline preserves punctuation and casing, which penalizes capitalization mismatch and minor punctuation discrepancies.
2. **Speaker Specificity**: Speaker 6930 exhibits unique acoustic and syntactic traits (e.g., rapid pacing and homophonic phrases) that increase local phonetic uncertainty. This single-speaker constraint is necessary to control variables across subsequent experiments but introduces a speaker bias that raises the absolute $WER$ baseline.

### Error Analysis & Speaker Variance
We observed a significant variance in $WER$ across files, ranging from $8.5\%$ to $45.2\%$ (specifically on file `0002.wav`).
- **Acoustic Coarticulation**: Whisper's decoder occasionally misinterprets adjacent words under soft speaking volumes (e.g., transcribing "poured in upon" as "report in upon").
- **Orthographic Penalties**: Preserving casing and contraction marks (e.g., "don't" vs. "do not") accounts for approximately $30\%$ of the error rate, showing that the model's core phonetic transcription is more accurate than the raw $WER$ suggests (supported by the low 4.45% $CER$).

## ⚖️ Engineering Trade-offs Identified

| Dimension | Observation | Engineering Implication |
|-----------|-------------|-------------------------|
| **Model Footprint** | 39M parameters (~150MB). | Highly viable for local mobile deployment without memory exhaustion. |
| **Accuracy Floor** | 18.60% $WER$ on clean audio. | Set as the strict threshold: any preprocessing algorithm that raises this $WER$ under clean/noisy states is rejected. |
| **Latency Overhead** | $\sim 1.9\text{ seconds}$ per utterance. | The autoregressive search is the primary bottleneck. Upstream preprocessing filters must keep latency $< 100\text{ ms}$ to avoid worsening this latency. |

## 🎯 Conclusion & Next Steps
The baseline experiment confirms that `openai/whisper-tiny` provides a functional, lightweight local ASR engine, but exhibits an orthographic error rate of 18.60% on clean speech under strict scoring conditions.

**Transition to Experiment 2**: We will introduce synthetic additive White Gaussian Noise at 20dB, 10dB, and 5dB SNR to evaluate the degradation curve of the raw model, and test whether classical digital signal processing (DSP) algorithms (Wiener filtering and Spectral Subtraction) can stabilize the ASR accuracy or if they introduce harmful acoustic artifacts.

## 📚 References
* [1] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, "Robust Speech Recognition via Large-Scale Weak Supervision," *Proceedings of the International Conference on Machine Learning (ICML)*, 2022.
* [2] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, "Librispeech: An ASR corpus based on public domain audio books," *Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pp. 5206–5210, 2015.