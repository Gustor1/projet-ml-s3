# asr/base_asr.py
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BaseASR(ABC):
    """
    Classe abstraite pour tous les moteurs ASR.
    Permet de rendre benchmark.py agnostique du modèle utilisé.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """Charge le modèle et le processor."""
        pass

    @abstractmethod
    def transcribe(self, audio_path: str) -> dict:
        """
        Transcrit un fichier audio.
        Doit retourner : {"text": str, "language": str, "file": str}
        """
        pass

    def transcribe_batch(self, audio_paths: list) -> list:
        """
        Transcrit une liste de fichiers audio.
        Gère les erreurs individuellement sans stopper le batch.
        """
        results = []
        for path in audio_paths:
            try:
                res = self.transcribe(path)
                results.append(res)
            except Exception as e:
                logger.error(f"[{self.__class__.__name__}] Failed on {path}: {e}")
                results.append({"text": "", "file": path, "error": str(e)})
        return results
