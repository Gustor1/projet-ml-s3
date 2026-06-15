# asr/evaluator.py
import jiwer
import pandas as pd
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Transformation WER
TRANSFORM_WER = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords()
])

# Transformation CER — cohérente avec WER
TRANSFORM_CER = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfChars()
])

def compute_wer(reference: str, hypothesis: str) -> float:
    if not hypothesis.strip():
        words = len(reference.split())
        return float(words) if words > 0 else 0.0
    return jiwer.wer(reference, hypothesis,
                     reference_transform=TRANSFORM_WER,
                     hypothesis_transform=TRANSFORM_WER)

def compute_cer(reference: str, hypothesis: str) -> float:
    if not hypothesis.strip():
        return float(len(reference.replace(" ", "")))
    return jiwer.cer(reference, hypothesis,
                     reference_transform=TRANSFORM_CER,
                     hypothesis_transform=TRANSFORM_CER)

def evaluate_batch(references: list, hypotheses: list, labels: list = None) -> pd.DataFrame:
    assert len(references) == len(hypotheses)
    rows = []
    for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
        label = labels[i] if labels else f"sample_{i}"
        rows.append({
            "file": label,
            "reference": ref,
            "hypothesis": hyp,
            "wer": round(compute_wer(ref, hyp), 4),
            "cer": round(compute_cer(ref, hyp), 4)
        })
    return pd.DataFrame(rows)

def summarize(df: pd.DataFrame) -> dict:
    return {
        "mean_wer": round(df["wer"].mean(), 4),
        "mean_cer": round(df["cer"].mean(), 4),
        "num_samples": len(df)
    }

def export_csv(df: pd.DataFrame, condition: str = "raw", output_dir: str = "results") -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/eval_{condition}_{timestamp}.csv"
    df.to_csv(filename, index=False)
    logger.info(f"[Evaluator] Exported: {filename}")
    return filename

if __name__ == "__main__":
    refs  = ["bonjour comment allez vous", "le chat est sur le tapis"]
    hypos = ["bonjour comment allez vou",  "le chat est sur le tap"]
    df = evaluate_batch(refs, hypos, labels=["sample_1", "sample_2"])
    print(df.to_string(index=False))
    print("\nRésumé :", summarize(df))
    export_csv(df, condition="raw")
