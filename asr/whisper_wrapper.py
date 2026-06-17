# asr/whisper_wrapper.py
import whisper
import os
import logging
from asr.base_asr import BaseASR

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = ["tiny", "base", "small", "medium"]

class WhisperWrapper(BaseASR):
    def __init__(self, model_size: str = "base", language: str = None):
        assert model_size in SUPPORTED_MODELS, f"Model must be one of {SUPPORTED_MODELS}"
        self.model_size = model_size
        self.language = language
        logger.info(f"[Whisper] Loading model '{model_size}'...")
        self.model = whisper.load_model(model_size)
        logger.info(f"[Whisper] Model loaded.")

    def transcribe(self, audio_path: str) -> dict:
        assert os.path.exists(audio_path), f"File not found: {audio_path}"
        options = {}
        if self.language:
            options["language"] = self.language
        result = self.model.transcribe(audio_path, **options)
        return {
            "text": result["text"].strip(),
            "language": result.get("language", "unknown"),
            "segments": result.get("segments", []),
            "file": audio_path
        }
