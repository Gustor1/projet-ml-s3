# Role 2: Audio Preprocessing Engineer - Video Script
*Duration: ~2 minutes*

> **Note to Presenter:** Use a neutral, academic tone. Avoid using "I". Refer to "the system", "this research", or "the engineering team". Maintain a professional pace and ensure the scientific rationale is clear.

---

### [Slide 1: Title - Audio Preprocessing & DSP]
![Role 2: Audio Preprocessing & DSP](/C:/Users/eliot/.gemini/antigravity-ide/brain/186b197f-c4cb-4dda-a3b6-4f9d29bd8526/slide_r2_title_1782341976320.png)

**Audio (Voiceover):**
"Moving to the foundational layer of the pipeline, signal conditioning is critical for both speech recognition and emotion detection. 
The objective here was to systematically evaluate classical Digital Signal Processing techniques to prepare raw audio before it reaches the deep learning models. 
Specifically, the system implements adaptive energy-based Voice Activity Detection, volume normalization, and evaluates two traditional denoising algorithms: the Wiener Filter and Spectral Subtraction."

---

### [Slide 2: The Limits of Classical DSP]
![The Limits of Classical DSP](/C:/Users/eliot/.gemini/antigravity-ide/brain/186b197f-c4cb-4dda-a3b6-4f9d29bd8526/slide_r2_limits_1782341996738.png)

**Audio (Voiceover):**
"Empirical evaluation revealed significant limitations in classical DSP when applied to complex, real-world noise.
While the Wiener Filter successfully improved the Word Error Rate for stationary white Gaussian noise at 5 decibels, it proved detrimental in non-stationary conditions. On urban noise and background babble, the noise estimation lagged behind transient acoustic events, distorting speech frequencies and actually *increasing* the Word Error Rate.
Similarly, Spectral Subtraction introduced spectral holes and 'musical noise', which modern attention mechanisms in models like Whisper hallucinate as phonemes, degrading transcription accuracy by up to 27 percent across all tested noise profiles."

---

### [Slide 3: Parallel Routing Architecture]
![Parallel Routing Architecture](/C:/Users/eliot/.gemini/antigravity-ide/brain/186b197f-c4cb-4dda-a3b6-4f9d29bd8526/slide_r2_routing_1782342006880.png)

**Audio (Voiceover):**
"A secondary, critical finding was that classical denoising destroys the prosodic micro-features—such as pitch, jitter, and shimmer—that are essential for Speech Emotion Recognition. Filtering dropped emotion classification accuracy by over 21 percent.
To resolve this multimodal conflict, the final implementation uses a Parallel Routing Architecture. 
The system routes optionally denoised audio exclusively to the ASR stream, while providing the SER model with raw audio that has only undergone silence trimming and peak normalization. 
Additionally, the preprocessing module extracts the fundamental frequency contour via the YIN algorithm and estimates the Signal-to-Noise ratio. These extracted acoustic features are later passed to a downstream heuristic, which fuses text sentiment and pitch to correct emotional misclassifications—yielding a 20 percent relative gain in accuracy."
