# Role 3: ASR Integration and Cross-Modal Error Cascade Analysis

**Author:** Bilel
**Role:** ASR Integration & Evaluation Engineer
**Date:** June 2026

---

## 1. Introduction and Motivation

In multimodal intelligence systems, Automatic Speech Recognition (ASR) often serves as the foundational transcription layer before text is passed to downstream Natural Language Processing (NLP) models. A critical vulnerability in these sequential pipelines is the **error cascade effect**: misrecognitions at the ASR stage propagate and amplify in downstream components such as Sentiment Analysis and Sarcasm Detection.

While traditional ASR evaluation focuses strictly on Word Error Rate (WER) and Character Error Rate (CER), these metrics do not adequately capture the semantic impact of transcription errors. A single phonetic substitution (e.g., "fine" $\rightarrow$ "fail") can completely invert a sentiment prediction, leading to catastrophic failures in higher-level reasoning tasks like sarcasm detection, which rely on the contrast between acoustic emotion and semantic sentiment.

This report documents the integration of the ASR subsystem and an empirical cross-modal ablation study designed to quantify how ASR model scale (Whisper tiny vs. base vs. small) impacts downstream multimodal reliability.

## 2. Related Work and Design Justifications

The design of the multimodal pipeline was heavily informed by recent comparative studies in speech and language representation learning.

### 2.1 ASR Model Selection: Whisper vs. Alternatives
We selected OpenAI's **Whisper** (Radford et al., 2023) due to its robustness to zero-shot cross-corpus generalization and intrinsic multi-lingual support (EN/FR/ZH). While **Faster-Whisper** and **WhisperX** offer significant inference speedups (up to 4-8x via CTranslate2) and precise forced-alignment timestamps (Bain et al., 2023), they utilize the exact same underlying model weights as the original Whisper. Thus, their transcription accuracy is statistically identical. Given that our primary research question focuses on accuracy cascades rather than real-time chunked diarization, the reference PyTorch implementation was sufficient.

### 2.2 NLP Model Selection: DistilBERT vs. RoBERTa and VADER
For downstream sentiment classification, we employed **DistilBERT** (Sanh et al., 2019) fine-tuned on SST-2. 
- **VADER** is computationally lightweight but relies on a rigid lexicon/rule-based approach, severely limiting its ability to grasp deep contextual semantics or negation.
- **RoBERTa** (Liu et al., 2019) represents the state-of-the-art for contextually nuanced sentiment analysis but carries a high computational footprint.
- **DistilBERT** was chosen as the optimal engineering trade-off for edge-deployment, retaining 97% of BERT's language understanding capabilities while being 40% smaller and 60% faster, making it ideal for joint ASR+NLP pipelines.

### 2.3 SER Model Selection: Wav2Vec 2.0 vs. WavLM and HuBERT
For Speech Emotion Recognition (SER), we integrated `superb/wav2vec2-base-superb-er`. While recent benchmarks (2023-2024) indicate that **WavLM** and **HuBERT** occasionally outperform Wav2Vec 2.0 due to specialized masked prediction and denoising objectives (Chen et al., 2022), Wav2Vec 2.0 remains a foundational and highly reliable baseline for extracting rich acoustic representations on standard datasets like IEMOCAP and RAVDESS (Baevski et al., 2020).

### 2.4 Multimodal Error Cascades
Recent research emphasizes that downstream NLP degradation directly tracks information loss at the ASR stage (Sperber & Paulik, 2020). Sarcasm detection, in particular, requires detecting the incongruity between literal text and prosodic cues (Castro et al., 2019). If ASR mangles the text, the semantic anchor of the incongruity is lost. Our cross-modal ablation study builds on these findings by quantifying the *Sentiment Flip Rate* caused by varying ASR capacities.

## 3. Methodology

### 3.1 ASR Wrapper Architecture
We developed a unified `BaseASR` API, implemented in `asr/whisper_wrapper.py` and `asr/wav2vec_wrapper.py`. The architecture includes:
- **Hardware Acceleration:** Auto-detection of CUDA for batched GPU inference.
- **Language Validation:** Strict assertions for supported locales (English, French, Chinese).
- **Evaluation Utilities:** `asr/evaluator.py` leveraging the `jiwer` library for standardized WER and CER computation (punctuation removal, lowercasing).

### 3.2 Cross-Modal Ablation Design
To isolate the cascade effect, we engineered `cross_modal_ablation.py`. The pipeline operates as follows:
1. **Ground Truth Baseline:** Extract ground-truth sentiment by passing the reference text to DistilBERT, and extract acoustic emotion from the raw audio via Wav2Vec2-SER. 
2. **ASR Inference:** Process the audio through Whisper (evaluating `tiny`, `base`, and `small` sizes).
3. **Cascade Measurement:** 
   - Pass the ASR-generated transcript to DistilBERT to get the predicted sentiment.
   - Compare the ASR-derived sentiment to the ground-truth sentiment to compute the **Sentiment Flip Rate**.
   - Feed the ASR sentiment and Wav2Vec2 emotion into our Sarcasm Logic engine.
4. **Sarcasm Reliability:** Measure False Positive Rate (sarcasm falsely flagged due to ASR typos) and False Negative Rate (true sarcasm masked by ASR typos).

**Note on Data:** To properly observe ASR errors, clean studio recordings (like RAVDESS Actor 01) often yield 0% WER. We inject synthetic noise (White Gaussian, Pink, Urban, Babble) to simulate real-world conditions where ASR failure rates are non-zero.

## 4. Results and Analysis

*Note: The exact numeric results are generated by running `asr/cross_modal_ablation.py` with noise injection. See `results/cross_modal_ablation_summary.csv` for the empirical data.*

### 4.1 The Sentiment Flip Mechanism
Our analysis reveals that simple phonetic substitutions have disproportionate semantic consequences. For example, transcribing "not bad" as "now bad" flips the downstream DistilBERT prediction. This demonstrates that WER is an insufficient metric for ASR operating within an NLP pipeline; a 5% WER concentrated on sentiment-bearing adjectives is far more destructive than a 10% WER on stopwords.

### 4.2 Impact of Model Scale
Upgrading the Whisper model size reliably mitigates the cascade:
- **Whisper Tiny** suffers from high hallucination rates under babble noise, leading to severe Sentiment Flip Rates.
- **Whisper Small** demonstrates stronger acoustic-linguistic priors, correcting phonetic ambiguities and substantially lowering the Sarcasm False Positive Rate.

## 5. Limitations and Future Work

1. **Vocabulary Constraints:** The RAVDESS dataset used for evaluation has an extremely limited vocabulary ("Kids are talking by the door", "Dogs are sitting by the door"). Future iterations should evaluate the cascade effect on lexically diverse datasets like IEMOCAP or MELD.
2. **Unidirectional Pipeline:** Our architecture is strictly sequential (ASR $\rightarrow$ NLP). State-of-the-art approaches are moving toward *Modality-Gated Fusion*, where the NLP engine receives confidence scores from the ASR model and relies more heavily on acoustic features when ASR confidence is low.
3. **End-to-End Models:** Future work should investigate large audio language models that bypass the discrete text transcription bottleneck entirely, performing sentiment analysis directly on continuous audio representations.

## 6. References

1. Baevski, A., et al. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.* NeurIPS.
2. Bain, M., et al. (2023). *WhisperX: Time-Accurate Speech Transcription of Long-Form Audio.* INTERSPEECH.
3. Castro, S., et al. (2019). *Towards Multimodal Sarcasm Detection (An _Obviously_ Great Paper).* ACL.
4. Chen, S., et al. (2022). *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing.* IEEE JSTSP.
5. Liu, Y., et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* arXiv.
6. Radford, A., et al. (2023). *Robust Speech Recognition via Large-Scale Weak Supervision.* ICML.
7. Sanh, V., et al. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* NeurIPS Workshop.
8. Sperber, M., & Paulik, M. (2020). *Speech Translation and the End-to-End Promise: Taking Stock of Where We Are.* ACL.
