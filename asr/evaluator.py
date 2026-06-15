# asr/evaluator.py
import jiwer
import pandas as pd
import os
from datetime import datetime

# Transformations standard pour normaliser avant calcul
TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords()
])

def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate — métrique principale ASR."""
    return jiwer.wer(reference, hypothesis,
                     reference_transform=TRANSFORM,
                     hypothesis_transform=TRANSFORM)

def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate — utile pour le chinois (ZH)."""
    return jiwer.cer(reference, hypothesis)

def evaluate_batch(references: list, hypotheses: list, labels: list = None) -> pd.DataFrame:
    """
    Évalue une liste de paires (référence, hypothèse).
    Retourne un DataFrame avec WER, CER par sample.
    
    references  : liste de transcriptions de référence (ground truth)
    hypotheses  : liste de transcriptions prédites par le modèle
    labels      : liste de noms de fichiers (optionnel)
    """
    assert len(references) == len(hypotheses), "references et hypotheses doivent avoir la même longueur"

    rows = []
    for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
        wer = compute_wer(ref, hyp)
        cer = compute_cer(ref, hyp)
        label = labels[i] if labels else f"sample_{i}"
        rows.append({
            "file": label,
            "reference": ref,
            "hypothesis": hyp,
            "wer": round(wer, 4),
            "cer": round(cer, 4)
        })

    df = pd.DataFrame(rows)
    return df

def summarize(df: pd.DataFrame) -> dict:
    """Retourne les métriques moyennes sur tout le dataset."""
    return {
        "mean_wer": round(df["wer"].mean(), 4),
        "mean_cer": round(df["cer"].mean(), 4),
        "num_samples": len(df)
    }

def export_csv(df: pd.DataFrame, condition: str = "raw", output_dir: str = "results") -> str:
    """
    Exporte le DataFrame en CSV dans results/.
    condition : 'raw' ou 'preprocessed'
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/eval_{condition}_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"[Evaluator] Résultats exportés : {filename}")
    return filename


if __name__ == "__main__":
    # Test avec données fictives
    refs  = ["bonjour comment allez vous", "le chat est sur le tapis"]
    hypos = ["bonjour comment allez vou",  "le chat est sur le tap"]

    df = evaluate_batch(refs, hypos, labels=["sample_1", "sample_2"])
    print(df.to_string(index=False))
    print("\nRésumé :", summarize(df))
    export_csv(df, condition="raw")
