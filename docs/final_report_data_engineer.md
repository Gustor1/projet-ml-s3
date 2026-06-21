# A Research-Grade Evaluation of Classical Speech Enhancement in Deep Learning-Based ASR and SER Multimodal Pipelines

**Role: Data & Experimentation Engineer (Eliott)**  
**Topic 3: Local Audio Preprocessing for Better ASR Performance**

---

## Abstract
This report presents a scientific evaluation of classical frequency-domain speech enhancement algorithms — Wiener filtering and Spectral Subtraction — when integrated with modern sequence-to-sequence Automatic Speech Recognition (ASR) and Speech Emotion Recognition (SER) models. Using a local edge deployment based on `openai/whisper-tiny` (39M parameters) and `superb/wav2vec2-base-superb-er`, we evaluate performance across six experiments covering synthetic stationary noise (White Gaussian), colored noise ($1/f$ pink noise), real-world non-stationary noise (DEMAND database), and overlapping speech interference (babble noise). 

Our findings demonstrate a **lab-to-real-world gap**: while Wiener filtering provides a modest gain under severe white noise at 5 dB SNR (a $2.75\%$ absolute $WER$ improvement), it degrades performance across all realistic noise profiles, including pink noise ($+11.13\%$), non-stationary urban street noise ($+9.31\%$), and babble noise ($+8.12\%$). Under babble noise, we document a hallucination trigger in Whisper's autoregressive decoder, showing that decoder perplexity ($PPL > 10,000$) serves as a predictor for model collapse. We also analyze the **enhancement-distortion trade-off**, demonstrating that Wiener filtering drops Wav2Vec2 SER accuracy by $21.43\%$ absolute under white noise by over-smoothing vocal prosody. We resolve this by designing a **parallel routing architecture** and a **multimodal fusion calibration engine** combining text sentiment (DistilBERT) and estimated pitch ($F_0$ from Librosa's YIN tracker), yielding a **+20% relative gain** in emotion classification accuracy.

---

## 1. Introduction & Scientific Background
Automatic Speech Recognition (ASR) has transitioned from traditional hidden Markov models (HMM-GMM) to end-to-end deep neural network (DNN) sequence-to-sequence transformers, such as Whisper [1]. While neural ASR models trained on large-scale datasets show high robustness, their performance degrades under low Signal-to-Noise Ratio (SNR) regimes.

In mobile and edge deployments, preprocessing is often introduced upstream to clean the audio signal. However, this classical signal processing paradigm assumes noise stationarity and does not model the acoustic representations utilized by neural networks. In multi-task pipelines that perform both ASR and Speech Emotion Recognition (SER), denoising filters can introduce spectral and temporal distortions that erase vocal prosody.

This report evaluates this **enhancement-distortion trade-off** and the **lab-to-real-world gap**. We evaluate:
1. Whether classical filters improve neural ASR.
2. How different noise spectra (white, pink, urban, babble) affect this behavior.
3. The downstream impact of denoising on vocal emotion features.
4. Multimodal calibration heuristics to resolve classification errors.

---

## 2. Model Architecture & Selection Rationale

For local edge execution under RAM, storage, and battery constraints, we selected a lightweight, three-model multimodal stack:

```mermaid
graph TD
    A[Input Audio Signal] --> B[Parallel Routing Engine]
    B -->|Denoised Stream (ASR)| C[Whisper-Tiny ASR]
    B -->|Normalized Stream (SER)| D[Wav2Vec2 SER]
    C -->|ASR Transcription| E[DistilBERT Sentiment]
    D -->|Vocal Emotion Scores| F[Multimodal Fusion Engine]
    E -->|Text Sentiment Scores| F
    F -->|Calibrated Emotion| G[Sarcasm Verdict & Intent Diagnosis]
```

### 2.1 Whisper-Tiny (`openai/whisper-tiny`)
Whisper-tiny (39M parameters, 4 encoder layers, 4 decoder layers, 6 attention heads) was selected over commercial or heavier alternatives:
- **Edge-Compute Emulation**: Serves as a proxy for edge and mobile CPUs with strict thermal and memory limits.
- **Diagnostic Accessibility**: Using the standard Hugging Face/PyTorch implementation provides programmatic access to internal cross-attention weight matrices, hidden representations, and token-level log-probabilities. This is essential for calculating decoder perplexity and analyzing the mechanics of hallucinations.
- **Exclusion of `Faster-Whisper`**: While `Faster-Whisper` (using CTranslate2) is optimized for deployment speed, it acts as a compiled C++ wrapper. Accessing internal model states or token-level gradients is highly constrained, making it unsuitable for diagnostic research.
- **Exclusion of `WhisperX`**: WhisperX incorporates forced phoneme alignment (Wav2Vec2) and VAD preprocessing (PyAnnote). Using it would make it impossible to isolate the intrinsic response of Whisper's autoregressive decoder to raw, noisy, or classically preprocessed speech signals.
- **Local Compute Constraints & Hardware Overload Avoidance**: Larger variants such as `whisper-base` (74M), `whisper-small` (244M), or `whisper-large` (1.55B parameters) introduce substantial VRAM footprints and memory bandwidth demands. Running a multi-model pipeline (ASR, SER, and NLP) concurrently on a standard consumer laptop or edge CPU/GPU would quickly trigger Out-Of-Memory (OOM) exceptions, severe thermal throttling, and overall system instability or crash (computational collapse). Using `whisper-tiny` ensures reliable local execution without overwhelming the consumer hardware.

### 2.2 Wav2Vec2 SER (`superb/wav2vec2-base-superb-er`)
We select a Wav2Vec2 model pre-trained on 960 hours of unlabeled speech (Baevski et al., 2020) [4] and fine-tuned on the IEMOCAP corpus [5] for the SUPERB benchmark. It captures temporal and spectral prosody features superior to traditional engineered MFCC features. Testing this IEMOCAP-trained model on the RAVDESS dataset allows us to study the **cross-corpus domain shift** in affective computing.

### 2.3 DistilBERT Sentiment (`distilbert-base-uncased-finetuned-sst-2-english`)
DistilBERT (Sanh et al., 2019) is a distilled version of BERT, retaining 97% of its language understanding while being 40% smaller and 60% faster [6]. It provides rapid verbal sentiment classification for edge multimodal fusion, matching the CPU resource constraints of Whisper-tiny and Wav2Vec2.

---

## 3. Mathematical Framework & DSP Algorithms

### 3.1 Spectral Subtraction (Boll, 1979)
Spectral subtraction assumes additive stationary noise [2]. Let the noisy signal be $y(n) = s(n) + v(n)$. In the Short-Time Fourier Transform (STFT) domain, the speech magnitude spectrum is estimated as:
$$|\hat{S}(\omega, t)|^b = \max\left( |Y(\omega, t)|^b - \alpha |\hat{V}(\omega)|^b, \, \beta |Y(\omega, t)|^b \right)$$
where $b=1$ for magnitude subtraction, $b=2$ for power spectral subtraction, $\alpha \ge 1$ is the over-subtraction factor, and $\beta \in [0, 1]$ is the spectral floor parameter to reduce musical noise. Time reconstruction utilizes the original noisy phase:
$$\hat{s}(n) = \text{ISTFT}\left( |\hat{S}(\omega, t)| e^{j \angle Y(\omega, t)} \right)$$

### 3.2 Wiener Filtering (Lim & Oppenheim, 1978)
The Wiener filter transfer function in the frequency domain is defined as:
$$H(\omega, t) = \frac{P_{ss}(\omega, t)}{P_{ss}(\omega, t) + P_{vv}(\omega, t)} = \frac{\xi(\omega, t)}{1 + \xi(\omega, t)}$$
where $P_{ss}$ and $P_{vv}$ are the speech and noise power spectral densities, respectively, and $\xi$ is the a priori SNR estimated using the decision-directed approach.

### 3.3 YIN Pitch Estimation (de Cheveigné & Kawahara, 2002)
To extract the fundamental frequency ($F_0$) trajectory, we compute the difference function $d_t(\tau)$ for lag $\tau$ over window $W$:
$$d_t(\tau) = \sum_{j=1}^W (x_j - x_{j+\tau})^2$$
This is normalized using the cumulative mean normalized difference function to prevent pitch doubling or halving errors:
$$d'_t(\tau) = \begin{cases} 1 & \text{if } \tau = 0 \\ \frac{d_t(\tau)}{\frac{1}{\tau} \sum_{j=1}^\tau d_t(j)} & \text{otherwise} \end{cases}$$
The average pitch $F_0$ is estimated by finding the first local minimum of $d'_t(\tau)$ below a specific threshold.

### 3.4 Decoder Perplexity ($PPL$)
ASR token generation uncertainty is quantified via the decoder's perplexity ($PPL$):
$$PPL = \exp\left( -\frac{1}{T} \sum_{t=1}^T \log P(y^*_t \mid y^*_{<t}, X) \right)$$
where $y^*$ is the ground-truth token sequence and $X$ is the acoustic feature map.

---

## 4. Experimental Results Synthesis

We consolidate the results of Experiments 1–6 across all noise types, SNR levels, and processing methods in the tables below.

### 4.1 ASR Metric Summary (WER / CER)
All metrics are evaluated on the 20 LibriSpeech files (Speaker 6930). Babble noise results reflect robust statistics (hallucinations excluded, N=174).

| Noise Type | SNR Level | Metric | Raw Noisy (`none`) | Wiener Filter | Spectral Subtraction |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Clean Baseline** | Clean | WER | 18.60% | — | — |
| | | CER | 4.45% | — | — |
| **White Gaussian** | 20 dB | WER | 18.94% | 18.79% | 25.73% ❌ |
| | | CER | 4.35% | 4.97% | 9.10% |
| | 10 dB | WER | 20.81% | 21.57% | 27.67% ❌ |
| | | CER | 6.14% | 7.17% | 10.64% |
| | 5 dB | WER | 27.47% | **24.72%** ✅ | 42.11% ❌ |
| | | CER | 9.86% | **9.20%** | 18.29% |
| **Pink 1/f** | 20 dB | WER | 17.48% | 18.89% | 24.88% ❌ |
| | 10 dB | WER | 19.47% | 21.56% | 29.54% ❌ |
| | 5 dB | WER | **22.21%** | 33.34% ❌ | 49.20% ❌ |
| **Urban Real** | 20 dB | WER | 18.24% | 19.18% | 25.04% ❌ |
| | 10 dB | WER | 22.12% | 22.07% | 28.40% ❌ |
| | 5 dB | WER | **26.17%** | 35.48% ❌ | 46.92% ❌ |
| **Babble Crowd** | 20 dB | WER | 19.11% | 18.97% | 20.98% ❌ |
| (Robust) | 10 dB | WER | 20.78% | 26.73% ❌ | 31.67% ❌ |
| | 5 dB | WER | **37.00%** | 45.12% ❌ | 55.49% ❌ |

---

### 4.2 Speech Emotion Recognition Accuracy Summary
Evaluated on the balanced dataset of 6 actors (168 clean files, 672 noisy files).

| Environmental Noise | SNR Level | Raw Noisy (`none`) | Wiener Filter | Spectral Subtraction |
|:---|:---:|:---:|:---:|:---:|
| **Clean Baseline** | Clean | **37.50%** | — | — |
| **White Gaussian** | 20 dB | **49.40%** | 33.33% ❌ | 44.05% ❌ |
| | 5 dB | **45.83%** | 24.40% ❌ | 31.55% ❌ |
| **Urban Real** | 20 dB | 44.64% | **45.83%** ✅ | 41.07% ❌ |
| | 5 dB | 35.12% | **35.71%** ✅ | 32.14% ❌ |

---

## 5. Key Contributions & Discussion

### 5.1 The Lab-to-Real-World Gap
Our results demonstrate a gap between laboratory benchmarks and real-world performance. Under White Gaussian Noise (5 dB SNR), the Wiener filter improved ASR accuracy, reducing $WER$ by $2.75\%$ absolute. However, under real-world noise at the same SNR (5 dB):
- **Pink Noise**: Wiener filter degraded $WER$ by **+11.13% absolute**.
- **Urban Noise**: Wiener filter degraded $WER$ by **+9.31% absolute**.
- **Babble Noise**: Wiener filter degraded $WER$ by **+8.12% absolute**.

This degradation is explained by the violation of the filter's noise stationarity assumptions.
- **Pink Noise**: Has a $1/f$ power spectrum. The Wiener filter over-attenuates high frequencies ($>2\text{ kHz}$) where speech formants $F_2$ and $F_3$ reside, causing **spectral tilt distortion**.
- **Urban Noise**: Real-world urban noise contains transient events (e.g., horns, sirens) that violate stationarity assumptions. The filter's noise estimate suffers from **tracking lag**, failing to adapt to transient peaks and over-attenuating the subsequent speech signal, leading to temporal smearing.

This is supported by Whisper-AT findings (Gong et al., 2023) [3]. Whisper's encoder encodes background noise, and the decoder transcribes speech conditioned on that noise type. By removing or distorting this background noise profile, classical filters disrupt the encoder's representations, degrading downstream ASR accuracy.

---

### 5.2 Mechanistic Proof of Babble-Noise Hallucinations
Under severe babble noise (**5 dB SNR**), the background noise consists of overlapping human speech, resulting in complete spectral overlap with the target voice. Classical filters cannot separate the two signals and introduce phase distortions that destroy the acoustic cues needed for correct phoneme identification.

This corruption causes **ASR hallucinations** in Whisper's autoregressive decoder. When the encoder representations are ambiguous, the cross-attention weights become uniform, and the decoder's language model prior takes over, generating fluent, grammatically correct sentences that are unrelated to the audio.

To monitor this behavior, we analyze the decoder's perplexity ($PPL$):
- **Hallucinated runs** ($WER \ge 100\%$): Average $PPL = \mathbf{34,881}$.
- **Non-hallucinated runs**: Average $PPL = \mathbf{898}$.

All hallucinated runs had $PPL > 20,000$, and no non-hallucinated run exceeded $17,318$. This confirms that high perplexity is a predictor of model collapse. In production monitoring, setting a threshold of $PPL > 10,000$ detects hallucinations with **100% recall** and **42.9% precision**, enabling proactive rejection of low-confidence transcriptions.

---

### 5.3 Parallel Routing Architecture for Joint ASR/SER
In multi-task systems that perform both speech transcription and emotion recognition, classical denoising filters (Wiener) smooth amplitude and frequency micro-variations (jitter, shimmer, pitch contours) to improve intelligibility. While this helps ASR transcription under certain conditions, it erases the vocal prosody that Wav2Vec2 SER relies on, dropping emotion classification accuracy by **21.43% absolute** (from $45.83\%$ to $24.40\%$).

To address this, we designed a **parallel routing architecture**:

```
                           +----------------------------+
                           |     Raw Noisy Audio        |
                           +--------------+-------------+
                                          |
                   +----------------------+----------------------+
                   |                                             |
     +-------------v-------------+                 +-------------v-------------+
     |   Wiener Denoising DSP    |                 |   Acoustic Calibration    |
     |  (Optimal for ASR Input)  |                 |    (Trim & Normalize)     |
     +-------------+-------------+                 +-------------+-------------+
                   |                                             |
     +-------------v-------------+                 +-------------v-------------+
     |        Whisper ASR        |                 |        Wav2Vec2 SER       |
     +-------------+-------------+                 +-------------+-------------+
                   |                                             |
     +-------------v-------------+                               |
     |   DistilBERT Sentiment    |                               |
     +-------------+-------------+                               |
                   |                                             |
                   +----------------------+----------------------+
                                          |
                           +--------------v-------------+
                           |  Multimodal Fusion Engine  |
                           |  (YIN Pitch + ASR + SER)   |
                           +--------------+-------------+
                                          |
                           +--------------v-------------+
                           |  Calibrated Affect Output  |
                           +----------------------------+
```

We pass the denoised audio to the ASR model, while passing a peak-normalized, silence-trimmed version of the original noisy audio directly to the SER model. The SER classification scores are calibrated in post-processing using our multimodal fusion engine (cross-referencing DistilBERT's text sentiment and the voice's pitch $F_0$), boosting classification accuracy on RAVDESS Actor 01 from **35.71% to 42.86%** (+20% relative gain).

---

## 6. Future Directions & Academic Recommendations
1. **Neural Source Separation**: Replace classical filters with deep neural networks optimized for source separation (e.g., Conv-TasNet or Demucs). These models can learn to isolate the target speaker's voice while preserving temporal and prosodic structures, avoiding the spectral distortions associated with classical DSP filters.
2. **Domain Adaptation**: Fine-tune the Wav2Vec2 SER model using a mixture of IEMOCAP and RAVDESS data to resolve the cross-corpus domain gap, improving baseline emotion classification accuracy.
3. **Edge Optimization**: Export the three-model pipeline (Whisper-tiny, Wav2Vec2, DistilBERT) to ONNX Runtime and apply Dynamic INT8 Quantization. This will reduce CPU latency and memory usage, enabling real-time execution on edge devices.

---

## 7. Bibliography
* [1] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, "Robust Speech Recognition via Large-Scale Weak Supervision," *Proceedings of the International Conference on Machine Learning (ICML)*, 2022.
* [2] S. Boll, "Suppression of acoustic noise in speech using spectral subtraction," *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 27, no. 2, pp. 113–120, 1979.
* [3] Y. Gong, H. Luo, and J. Glass, "Whisper-AT: Noise-Robust Automatic Speech Recognizers are Also Strong General Audio Event Taggers," *Proceedings of Interspeech*, pp. 2798–2802, 2023.
* [4] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, "wav2vec 2.0: A framework for self-supervised learning of speech representations," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, pp. 12449–12460, 2020.
* [5] C. Busso, M. Bulut, C. C. Lee, A. Kazemzadeh, E. Mower, S. Kim, J. N. Chang, S. Lee, and S. S. Narayanan, "IEMOCAP: Interactive emotional dyadic motion capture database," *Language Resources and Evaluation*, vol. 42, no. 4, pp. 335–359, 2008.
* [6] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter," *arXiv preprint arXiv:1910.01108*, 2019.
* [7] C. Evans, J. S. Mason, and W. M. Campbell, "On the Fundamental Limitations of Spectral Subtraction," *Proceedings of the European Signal Processing Conference (EUSIPCO)*, 2005.
* [8] A. de Cheveigné and H. Kawahara, "YIN, a fundamental frequency estimator for speech and music," *Journal of the Acoustical Society of America*, vol. 111, no. 4, pp. 1917–1930, 2002.
* [9] Y. Tsao, S. H. Liu, and Y. Tsao, "The impact of speech enhancement on speech emotion recognition," *IEEE Signal Processing Letters*, vol. 26, no. 12, pp. 1803–1807, 2019.
* [10] S. Latif et al., "Cross-corpus speech emotion recognition: An overview and directions," *IEEE Transactions on Affective Computing*, 2021.
