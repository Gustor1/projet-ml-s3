# asr/wav2vec_wrapper.py
import torch
import librosa
import logging
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[Wav2Vec2] Loading '{model_id}' on {self.device}...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(self.device)
        self.model.eval()
        logger.info("[Wav2Vec2] Model loaded.")

    def transcribe(self, audio_path: str) -> dict:
        # librosa force le resampling à 16kHz + conversion mono automatique
        speech, _ = librosa.load(audio_path, sr=16000, mono=True)
        inputs = self.processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        # FIX : batch_decode retourne une liste
        transcription = self.processor.batch_decode(predicted_ids)[0].strip()
        return {"text": transcription, "language": self.language, "file": audio_path}

    def transcribe_batch(self, audio_paths: list) -> list:
        results = []
        for path in audio_paths:
            try:
                res = self.transcribe(path)
                results.append(res)
            except Exception as e:
                logger.error(f"[Wav2Vec2] Failed on {path}: {e}")
                results.append({"text": "", "language": self.language, "file": path, "error": str(e)})
        return results

if __name__ == "__main__":
    wrapper = Wav2VecWrapper(language="fr")
    result = wrapper.transcribe("test.wav")
    print(f"Transcription : {result['text']}")
