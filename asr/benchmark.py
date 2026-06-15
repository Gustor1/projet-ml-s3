# asr/benchmark.py
import os, json, time, logging
import pandas as pd
import librosa
from asr.base_asr import BaseASR
from asr.evaluator import evaluate_batch, summarize, export_csv

logger = logging.getLogger(__name__)

def load_dataset(dataset_path: str) -> tuple:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    valid = [d for d in data if os.path.exists(d["file"])]
    skipped = len(data) - len(valid)
    if skipped > 0:
        logger.warning(f"[Dataset] {skipped} fichier(s) ignoré(s).")
    return ([d["file"] for d in valid],
            [d["reference"] for d in valid],
            [os.path.basename(d["file"]) for d in valid])

def get_audio_duration(filepath: str) -> float:
    try:
        return librosa.get_duration(path=filepath)
    except Exception:
        logger.warning(f"[RTF] Impossible de lire la durée de {filepath}")
        return 0.0

def run_benchmark(asr_engine: BaseASR, dataset_path: str, condition: str = "raw") -> dict:
    logger.info(f"[Benchmark] Condition: {condition} | Moteur: {asr_engine.__class__.__name__}")
    audio_paths, references, labels = load_dataset(dataset_path)

    start = time.time()
    results = asr_engine.transcribe_batch(audio_paths)
    elapsed = round(time.time() - start, 2)

    total_audio_duration = sum(get_audio_duration(p) for p in audio_paths)
    rtf = round(elapsed / total_audio_duration, 4) if total_audio_duration > 0 else float("inf")

    hypotheses = [r.get("text", "") for r in results]
    df = evaluate_batch(references, hypotheses, labels=labels)
    df["audio_duration_total_s"] = round(total_audio_duration, 2)
    df["inference_time_total_s"] = elapsed
    df["rtf"] = rtf

    summary = summarize(df)
    export_csv(df, condition=condition)
    logger.info(f"[Benchmark] WER: {summary['mean_wer']} | CER: {summary['mean_cer']} | RTF: {rtf}")
    return summary

def compare_conditions(asr_engine: BaseASR, dataset_raw: str, dataset_preprocessed: str) -> pd.DataFrame:
    raw  = run_benchmark(asr_engine, dataset_raw,          condition="raw")
    prep = run_benchmark(asr_engine, dataset_preprocessed, condition="preprocessed")
    comparison = pd.DataFrame([{"condition": "raw", **raw}, {"condition": "preprocessed", **prep}])
    comparison["wer_improvement_%"] = round(
        (raw["mean_wer"] - prep["mean_wer"]) / max(raw["mean_wer"], 1e-6) * 100, 2)
    comparison["cer_improvement_%"] = round(
        (raw["mean_cer"] - prep["mean_cer"]) / max(raw["mean_cer"], 1e-6) * 100, 2)
    os.makedirs("results", exist_ok=True)
    comparison.to_csv("results/comparison.csv", index=False)
    return comparison

if __name__ == "__main__":
    from asr.whisper_wrapper import WhisperWrapper
    engine = WhisperWrapper(model_size="base", language="fr")
    compare_conditions(engine, "data/dataset_raw.json", "data/dataset_preprocessed.json")
