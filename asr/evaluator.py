# asr/evaluator.py
import jiwer
import pandas as pd
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

TRANSFORM_WER = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords()
])

TRANSFORM_CER = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfChars()
])

def compute_wer(reference: str, hypothesis: str) -> float:
    if not hypothesis.strip():
        logger.warning(f"[WER] Empty hypothesis for ref: '{reference[:50]}'")
        return 1.0 if len(reference.split()) > 0 else 0.0
    return jiwer.wer(reference, hypothesis,
                     reference_transform=TRANSFORM_WER,
                     hypothesis_transform=TRANSFORM_WER)

def compute_cer(reference: str, hypothesis: str) -> float:
    if not hypothesis.strip():
        logger.warning(f"[CER] Empty hypothesis for ref: '{reference[:50]}'")
        return 1.0 if len(reference.replace(" ", "")) > 0 else 0.0
    return jiwer.cer(reference, hypothesis,
                     reference_transform=TRANSFORM_CER,
                     hypothesis_transform=TRANSFORM_CER)

def evaluate_batch(references: list, hypotheses: list, labels: list = None) -> pd.DataFrame:
    assert len(references) == len(hypotheses)
    rows = []
    for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
        label = labels[i] if labels else f"sample_{i}"
        rows.append({"file": label, "reference": ref, "hypothesis": hyp,
                     "wer": round(compute_wer(ref, hyp), 4),
                     "cer": round(compute_cer(ref, hyp), 4)})
    return pd.DataFrame(rows)

def summarize(df: pd.DataFrame) -> dict:
    return {"mean_wer": round(df["wer"].mean(), 4),
            "mean_cer": round(df["cer"].mean(), 4),
            "num_samples": len(df)}

def export_csv(df: pd.DataFrame, condition: str = "raw", output_dir: str = "results") -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/eval_{condition}_{timestamp}.csv"
    df.to_csv(filename, index=False)
    logger.info(f"[Evaluator] Exported: {filename}")
    return filename
