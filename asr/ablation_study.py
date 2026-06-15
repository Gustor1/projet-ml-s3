# asr/ablation_study.py
import os, time, logging
import pandas as pd
from asr.whisper_wrapper import WhisperWrapper
from asr.benchmark import load_dataset
from asr.evaluator import evaluate_batch, summarize

logger = logging.getLogger(__name__)
MODELS = ["tiny", "base", "small", "medium"]

def run_ablation(dataset_path: str, language: str = None, condition: str = "raw") -> pd.DataFrame:
    audio_paths, references, labels = load_dataset(dataset_path)
    rows = []
    for model_size in MODELS:
        logger.info(f"[Ablation] Testing whisper-{model_size}...")
        wrapper = WhisperWrapper(model_size=model_size, language=language)
        start = time.time()
        results = wrapper.transcribe_batch(audio_paths)
        elapsed = round(time.time() - start, 2)
        hypotheses = [r.get("text", "") for r in results]
        summary = summarize(evaluate_batch(references, hypotheses, labels=labels))
        rows.append({"model": f"whisper-{model_size}", **summary, "inference_time_s": elapsed})
        logger.info(f"  WER: {summary['mean_wer']} | CER: {summary['mean_cer']} | Temps: {elapsed}s")
    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv(f"results/ablation_{condition}.csv", index=False)
    logger.info(f"[Ablation] Saved: results/ablation_{condition}.csv")
    return df

if __name__ == "__main__":
    run_ablation("data/dataset_raw.json", language="fr", condition="raw")
