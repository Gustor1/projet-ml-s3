# asr/whisper_wrapper.py
import whisper
import os

SUPPORTED_MODELS = ["tiny", "base", "small", "medium"]

class WhisperWrapper:
    def __init__(self, model_size: str = "base", language: str = None):
        """
        model_size : tiny | base | small | medium
        language   : None (auto-detect) | 'en' | 'fr' | 'zh'
        """
        assert model_size in SUPPORTED_MODELS, f"Model must be one of {SUPPORTED_MODELS}"
        self.model_size = model_size
        self.language = language
        print(f"[Whisper] Loading model '{model_size}'...")
        self.model = whisper.load_model(model_size)
        print(f"[Whisper] Model loaded.")

    def transcribe(self, audio_path: str) -> dict:
        """
        Transcrit un fichier audio.
        Retourne un dict avec 'text', 'language', 'segments'.
        """
        assert os.path.exists(audio_path), f"File not found: {audio_path}"

        options = {}
        if self.language:
            options["language"] = self.language

        result = self.model.transcribe(audio_path, **options)

        return {
            "text": result["text"].strip(),
            "language": result.get("language", "unknown"),
            "segments": result.get("segments", [])
        }

    def transcribe_batch(self, audio_paths: list) -> list:
        """
        Transcrit une liste de fichiers audio.
        Retourne une liste de dicts.
        """
        results = []
        for path in audio_paths:
            print(f"[Whisper] Transcribing: {path}")
            res = self.transcribe(path)
            res["file"] = path
            results.append(res)
        return results


if __name__ == "__main__":
    # Test rapide — remplace par un vrai fichier audio
    wrapper = WhisperWrapper(model_size="base", language="fr")
    result = wrapper.transcribe("test.wav")
    print(f"Transcription : {result['text']}")
    print(f"Langue détectée : {result['language']}")
