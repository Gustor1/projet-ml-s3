# asr/benchmark.py
import os
import json
import time
import logging
import pandas as pd
from asr.base_asr import BaseASR
from asr.evaluator import evaluate_batch, summarize, export_csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset(dataset_path: str) -> tuple:
    """
    Format attendu : [{"file": "audio.wav", "reference": "transcription ground truth"}, ...]
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filtre les fichiers inexistants sans crasher
    valid = [d for d in data if os.path.exists(d["file"])]
    skipped = len(data) - len(valid)
    if skipped > 0:
        logger.warning(f"[Dataset] {skipped} fichier(s) introuvable(s) ignoré(s).")

    audio_paths = [d["file"] for d in valid]
    references  = [d["reference"] for d in valid]
    labels      = [os.path.basename(d["file"]) for d in valid]
    return audio_paths, references, labels


def run_benchmark(
    asr_engine: BaseASR,
    dataset_path: str,
    condition: str = "raw"
) -> dict:
    """
    Lance un benchmark complet avec n'importe quel moteur BaseASR.
    Retourne le résumé des métriques.
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"[Benchmark] Condition: {condition} | Moteur: {asr_engine.__class__.__name__}")
    logger.info(f"{'='*50}")

    audio_paths, references, labels = load_dataset(dataset_path)

    start = time.time()
    results = asr_engine.transcribe_batch(audio_paths)
    elapsed = round(time.time() - start, 2)

    hypotheses = [r.get("text", "") for r in results]

    df = evaluate_batch(references, hypotheses, labels=labels)
    df["inference_time_total_s"] = elapsed
    # RTF approximatif (suppose ~5s d'audio moyen par fichier)
    df["rtf"] = round(elapsed / max(len(audio_paths) * 5, 1), 4)

    summary = summarize(df)
    export_csv(df, condition=condition)

    logger.info(f"\n[Benchmark] {condition} — WER: {summary['mean_wer']} | CER: {summary['mean_cer']} | Temps: {elapsed}s")
    return summary


def compare_conditions(
    asr_engine: BaseASR,
    dataset_raw: str,
    dataset_preprocessed: str
) -> pd.DataFrame:
    """
    Compare raw vs preprocessed avec le même moteur ASR.
    Exporte results/comparison.csv
    """
    raw  = run_benchmark(asr_engine, dataset_raw,          condition="raw")
    prep = run_benchmark(asr_engine, dataset_preprocessed, condition="preprocessed")

    comparison = pd.DataFrame([
        {"condition": "raw",          **raw},
        {"condition": "preprocessed", **prep}
    ])

    comparison["wer_improvement_%"] = round(
        (raw["mean_wer"] - prep["mean_wer"]) / max(raw["mean_wer"], 1e-6) * 100, 2
    )
    comparison["cer_improvement_%"] = round(
        (raw["mean_cer"] - prep["mean_cer"]) / max(raw["mean_cer"], 1e-6) * 100, 2
    )

    os.makedirs("results", exist_ok=True)
    comparison.to_csv("results/comparison.csv", index=False)

    logger.info(f"\n{'='*50}")
    logger.info("[Benchmark] Comparaison finale :")
    logger.info(f"\n{comparison.to_string(index=False)}")
    logger.info(f"{'='*50}")
    return comparison


if __name__ == "__main__":
    from asr.whisper_wrapper import WhisperWrapper
    engine = WhisperWrapper(model_size="base", language="fr")
    compare_conditions(
        asr_engine=engine,
        dataset_raw="data/dataset_raw.json",
        dataset_preprocessed="data/dataset_preprocessed.json"
    )
