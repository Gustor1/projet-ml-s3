# asr/whisper_wrapper.py
import whisper
import torch
import os
import logging
from asr.base_asr import BaseASR

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = ["tiny", "base", "small", "medium"]
SUPPORTED_LANGUAGES = {
    "en": "english",
    "fr": "french",
    "zh": "chinese",
}


class WhisperWrapper(BaseASR):
    """
    Wrapper ASR autour d'OpenAI Whisper.
    Supporte les modèles tiny/base/small/medium et les langues EN/FR/ZH.
    Détecte automatiquement GPU/CPU.
    """

    def __init__(self, model_size: str = "base", language: str = None):
        super().__init__()
        assert model_size in SUPPORTED_MODELS, f"Model must be one of {SUPPORTED_MODELS}"
        if language is not None:
            assert language in SUPPORTED_LANGUAGES, (
                f"Language '{language}' not supported. Use one of {list(SUPPORTED_LANGUAGES.keys())}"
            )
        self.model_size = model_size
        self.language = language
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[Whisper] Loading model '{model_size}' on {self.device}...")
        self.model = whisper.load_model(model_size, device=self.device)
        logger.info(f"[Whisper] Model loaded. Language: {language or 'auto-detect'}")

    def transcribe(self, audio_path: str) -> dict:
        """
        Transcrit un fichier audio.
        Retourne: {"text": str, "language": str, "segments": list, "file": str}
        """
        assert os.path.exists(audio_path), f"File not found: {audio_path}"
        options = {}
        if self.language:
            options["language"] = self.language
        result = self.model.transcribe(audio_path, **options)
        detected_lang = result.get("language", "unknown")
        logger.debug(f"[Whisper] {os.path.basename(audio_path)} → lang={detected_lang}, "
                      f"len={len(result['text'].split())} words")
        return {
            "text": result["text"].strip(),
            "language": detected_lang,
            "segments": result.get("segments", []),
            "file": audio_path
        }
