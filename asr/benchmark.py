# asr/benchmark.py
import os
import json
import pandas as pd
from asr.whisper_wrapper import WhisperWrapper
from asr.evaluator import evaluate_batch, summarize, export_csv

def load_dataset(dataset_path: str) -> tuple:
    """
    Charge un dataset depuis un fichier JSON.
    Format attendu : [{"file": "audio.wav", "reference": "transcription ground truth"}, ...]
    Retourne (audio_paths, references, labels)
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    audio_paths = [d["file"] for d in data]
    references  = [d["reference"] for d in data]
    labels      = [os.path.basename(d["file"]) for d in data]
    return audio_paths, references, labels


def run_benchmark(
    dataset_path: str,
    model_size: str = "base",
    language: str = None,
    condition: str = "raw"
) -> dict:
    """
    Lance un benchmark complet sur un dataset.
    
    dataset_path : chemin vers le JSON du dataset
    model_size   : tiny | base | small | medium
    language     : None (auto) | 'en' | 'fr' | 'zh'
    condition    : 'raw' | 'preprocessed'
    
    Retourne un dict avec le résumé des métriques.
    """
    print(f"\n{'='*50}")
    print(f"[Benchmark] Condition : {condition} | Modèle : whisper-{model_size}")
    print(f"{'='*50}")

    audio_paths, references, labels = load_dataset(dataset_path)

    wrapper = WhisperWrapper(model_size=model_size, language=language)
    results = wrapper.transcribe_batch(audio_paths)
    hypotheses = [r["text"] for r in results]

    df = evaluate_batch(references, hypotheses, labels=labels)
    summary = summarize(df)

    print(f"\n[Benchmark] Résultats {condition}:")
    print(df[["file", "wer", "cer"]].to_string(index=False))
    print(f"\nMoyenne — WER: {summary['mean_wer']} | CER: {summary['mean_cer']} | Samples: {summary['num_samples']}")

    export_csv(df, condition=condition)
    return summary


def compare_conditions(
    dataset_raw: str,
    dataset_preprocessed: str,
    model_size: str = "base",
    language: str = None
):
    """
    Compare les métriques avant/après preprocessing.
    Génère un tableau récapitulatif dans results/comparison.csv
    """
    raw_summary  = run_benchmark(dataset_raw,          model_size, language, condition="raw")
    prep_summary = run_benchmark(dataset_preprocessed, model_size, language, condition="preprocessed")

    comparison = pd.DataFrame([
        {"condition": "raw",          **raw_summary},
        {"condition": "preprocessed", **prep_summary}
    ])

    comparison["wer_improvement_%"] = round(
        (raw_summary["mean_wer"] - prep_summary["mean_wer"]) / raw_summary["mean_wer"] * 100, 2
    )
    comparison["cer_improvement_%"] = round(
        (raw_summary["mean_cer"] - prep_summary["mean_cer"]) / raw_summary["mean_cer"] * 100, 2
    )

    os.makedirs("results", exist_ok=True)
    comparison.to_csv("results/comparison.csv", index=False)

    print("\n" + "="*50)
    print("[Benchmark] Comparaison finale :")
    print(comparison.to_string(index=False))
    print("="*50)
    return comparison


if __name__ == "__main__":
    # Exemple d'utilisation — remplace par tes vrais fichiers JSON
    compare_conditions(
        dataset_raw="data/dataset_raw.json",
        dataset_preprocessed="data/dataset_preprocessed.json",
        model_size="base",
        language="fr"
    )
