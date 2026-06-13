#!/usr/bin/env python3
"""
Ajoute le CER aux résultats existants de preprocessing_comparison.csv
"""
import csv
import jiwer
from pathlib import Path

# Charger le CSV existant
csv_path = Path("results/preprocessing_comparison.csv")
rows = []

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Ajouter le CER pour chaque ligne
print(f"Traitement de {len(rows)} lignes...")
for row in rows:
    if row["wer"] == "-1":
        row["cer"] = -1
        continue
    
    # Recharger les transcriptions depuis metadata
    import json
    with open("data/augmented_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    # Trouver la transcription correspondante
    filename = row["file_name"]
    entry = next((m for m in metadata if Path(m["file_path"]).name == filename), None)
    
    if entry:
        ref = entry["transcription"].strip().lower()
        pred = row.get("prediction", "")  # Tu devras peut-être adapter
        
        # Calculer CER
        cer = jiwer.cer(ref, pred) if ref and pred else -1
        row["cer"] = round(cer, 4)
    else:
        row["cer"] = -1

# Sauvegarder avec CER
output_path = Path("results/preprocessing_comparison_with_cer.csv")
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["file_name", "snr_db", "method", "wer", "cer", "latency_ms"])
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ CSV avec CER sauvegardé dans {output_path}")