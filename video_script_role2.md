# Role 2: Audio Preprocessing Engineer - Video Script
*Duration: ~2 minutes*

> **Note to Presenter:** Use a neutral, academic tone. Avoid using "I". Refer to "the system", "this research", or "the engineering team". Maintain a professional pace and ensure the scientific rationale is clear.

---

### [Slide 1: Title - Audio Preprocessing & DSP Architecture]
`[VISUAL: visuals/slide_r2_title.png — Title slide: Audio Preprocessing & DSP Architecture]`

**Audio (Voiceover):**
"Moving to the foundational layer of the pipeline, signal conditioning is critical for both speech recognition and emotion detection. 
The objective here was to systematically evaluate classical Digital Signal Processing techniques to prepare raw audio before it reaches the deep learning models. 
Drawing on foundational research by Oppenheim and Lim on the importance of phase and amplitude, the system evaluates two traditional denoising algorithms: the Wiener Filter and Spectral Subtraction, alongside adaptive energy-based Voice Activity Detection."

---

### [Slide 2: The Limits of Classical DSP]
`[VISUAL: visuals/slide_r2_limits.png — Limits of Classical DSP: Spectral Subtraction vs. Wiener Filter]`

**Audio (Voiceover):**
"Empirical evaluation revealed significant limitations in classical DSP when applied to complex, real-world noise.
While the Wiener Filter successfully improved the Word Error Rate for stationary white Gaussian noise, it proved detrimental in non-stationary conditions. 
More importantly, as detailed by Evans et al. in their research on Spectral Subtraction, this method introduces fundamental limitations like 'musical noise'. Our experiments confirmed this: modern attention mechanisms in models like Whisper hallucinate these artifacts as phonemes, degrading transcription accuracy by up to 27 percent. We maintain these negative results as a documented baseline to prove why magnitude-only subtraction fails for transformer-based ASR."

---

### [Slide 3: Parallel Stream Routing Architecture]
`[VISUAL: visuals/slide_parallel_routing.png — Block diagram of the Parallel Routing Architecture]`

Here is the schema of the decoupled pipeline:
```mermaid
graph TD
    A[Raw Audio Input] --> B[Silence Trimming & Normalization]
    B --> C{Parallel Routing}
    C -->|ASR Stream| D[Optional Wiener Filter if SNR < 10dB]
    C -->|SER Stream| E[Raw Normalized Audio]
    D --> F[Whisper ASR Model]
    E --> G[Wav2Vec2 SER Model]
```

**Audio (Voiceover):**
"A secondary, critical finding was that classical denoising destroys the prosodic micro-features—such as pitch, jitter, and shimmer—that are essential for Speech Emotion Recognition. Filtering dropped emotion classification accuracy by over 21 percent.
To resolve this multimodal conflict, the final implementation introduces a Parallel Stream Routing Architecture. 
The system routes optionally denoised audio exclusively to the ASR stream, while providing the Wav2Vec2 emotion model with raw audio that has only undergone silence trimming and peak normalization. 
By decoupling these streams, we preserve the necessary acoustic features for emotion detection while still isolating static noise for transcription, achieving optimal performance across both tasks."
