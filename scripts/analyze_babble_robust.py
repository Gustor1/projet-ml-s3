#!/usr/bin/env python3
"""
Analyse robuste des résultats babble noise (exclusion des outliers).
"""
import pandas as pd
import numpy as np

# Charger le CSV
df = pd.read_csv("results/babble_noise_comparison.csv")

# Filtrer les résultats valides (WER < 1.0 = 100%)
df_valid = df[df["wer"] < 1.0].copy()

print(f"Total lignes: {len(df)}")
print(f"Lignes valides (WER < 100%): {len(df_valid)}")
print(f"Outliers exclus: {len(df) - len(df_valid)}")
print()

# Calculer les moyennes robustes
print("="*70)
print("MOYENNES ROBUSTES (exclusion des outliers WER > 100%)")
print("="*70)

for method in ["none", "wiener", "spectral_subtraction"]:
    method_data = df_valid[df_valid["method"] == method]
    avg_wer = method_data["wer"].mean()
    avg_cer = method_data["cer"].mean()
    avg_lat = method_data["latency_ms"].mean()
    print(f"{method:25s} | WER: {avg_wer:.2%} | CER: {avg_cer:.2%} | Latency: {avg_lat:.0f}ms")

print()
print("="*70)
print("ANALYSE PAR SNR (moyennes robustes)")
print("="*70)

pivot = df_valid.groupby(["snr_db", "method"])["wer"].mean().unstack()
print(pivot.round(4))

print()
print("="*70)
print("OUTLIERS EXCLUS (fichiers avec WER > 100%)")
print("="*70)

outliers = df[df["wer"] >= 1.0]
for _, row in outliers.iterrows():
    print(f"{row['file_name']:40s} | {row['method']:20s} | WER: {row['wer']:.2f}")