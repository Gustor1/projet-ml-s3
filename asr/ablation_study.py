# asr/ablation_study.py
import os
import json
import time
import pandas as pd
from asr.whisper_wrapper import WhisperWrapper
from asr.evaluator import evaluate_batch, summarize, export_csv

MODELS = ["tiny", "base", "small", "medium"]

def run_ablation(dataset_path: str, language: str = None, condition: str = "raw"):
    """
    Teste tous les modèles Whisper sur le même dataset.
    Mesure WER, CER et temps d'inférence par modèle.
    Exporte les résultats dans results/ablation_{condition}.csv
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    audio_paths = [d["file"] for d in data]
    references  = [d["reference"] for d in data]
    labels      = [os.path.basename(d["file"]) for d in data]

    rows = []

    for model_size in MODELS:
        print(f"\n[Ablation] Testing whisper-{model_size}...")
        wrapper = WhisperWrapper(model_size=model_size, language=language)

        start = time.time()
        results = wrapper.transcribe_batch(audio_paths)
        elapsed = round(time.time() - start, 2)

        hypotheses = [r["text"] for r in results]
        df = evaluate_batch(references, hypotheses, labels=labels)
        summary = summarize(df)

        rows.append({
            "model": f"whisper-{model_size}",
            "mean_wer": summary["mean_wer"],
            "mean_cer": summary["mean_cer"],
            "inference_time_s": elapsed,
            "num_samples": summary["num_samples"]
        })

        print(f"  WER: {summary['mean_wer']} | CER: {summary['mean_cer']} | Temps: {elapsed}s")

    ablation_df = pd.DataFrame(rows)

    os.makedirs("results", exist_ok=True)
    out_path = f"results/ablation_{condition}.csv"
    ablation_df.to_csv(out_path, index=False)

    print(f"\n{'='*50}")
    print("[Ablation] Résultats complets :")
    print(ablation_df.to_string(index=False))
    print(f"{'='*50}")
    print(f"[Ablation] Exporté : {out_path}")

    return ablation_df


if __name__ == "__main__":
    run_ablation(
        dataset_path="data/dataset_raw.json",
        language="fr",
        condition="raw"
    )
