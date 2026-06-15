# asr/wav2vec_wrapper.py
import torch
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Modèles par langue
LANG_MODELS = {
    "en": "facebook/wav2vec2-base-960h",
    "fr": "facebook/wav2vec2-large-xlsr-53-french",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
}

class Wav2VecWrapper:
    def __init__(self, language: str = "en"):
        assert language in LANG_MODELS, f"Language must be one of {list(LANG_MODELS.keys())}"
        self.language = language
        model_id = LANG_MODELS[language]
        print(f"[Wav2Vec2] Loading model for '{language}': {model_id}")
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id)
        self.model.eval()
        print(f"[Wav2Vec2] Model loaded.")

    def transcribe(self, audio_path: str) -> dict:
        speech, sample_rate = sf.read(audio_path)
        if sample_rate != 16000:
            raise ValueError(f"Expected 16kHz audio, got {sample_rate}Hz. Use preprocessing/resample first.")
        inputs = self.processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return {"text": transcription.strip(), "language": self.language, "file": audio_path}

    def transcribe_batch(self, audio_paths: list) -> list:
        results = []
        for path in audio_paths:
            print(f"[Wav2Vec2] Transcribing: {path}")
            results.append(self.transcribe(path))
        return results

if __name__ == "__main__":
    wrapper = Wav2VecWrapper(language="fr")
    result = wrapper.transcribe("test.wav")
    print(f"Transcription : {result['text']}")
