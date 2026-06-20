#!/usr/bin/env python3
"""
scripts/analyze_speakers.py
Analyse de la performance ASR par Speaker ID
Génère un rapport dans docs/speaker_analysis.md
"""
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_speakers():
    # Charger le CSV
    csv_path = Path("results/preprocessing_comparison.csv")
    if not csv_path.exists():
        logger.error(f"❌ Fichier non trouvé: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # Filtrer uniquement les résultats valides (none et wiener)
    df = df[df["wer"] >= 0].copy()
    
    if df.empty:
        logger.error("❌ Aucune donnée valide dans le CSV")
        return
    
    # Extraire le Speaker ID (ex: 6930-75918-0000_snr20dB.wav -> 6930)
    df["speaker_id"] = df["file_name"].str.split("-").str[0]
    
    logger.info(f"📊 Analyse de {df['speaker_id'].nunique()} speakers")
    logger.info(f"   Total samples: {len(df)}")
    
    # Calculer les statistiques par speaker
    speaker_stats = df.groupby("speaker_id").agg({
        "wer": ["mean", "std", "count"],
        "cer": "mean",
        "latency_ms": "mean"
    }).round(4)
    
    # Aplatir les colonnes MultiIndex
    speaker_stats.columns = ["_".join(col).strip() for col in speaker_stats.columns.values]
    speaker_stats = speaker_stats.rename(columns={
        "wer_mean": "avg_wer",
        "wer_std": "std_wer",
        "wer_count": "sample_count",
        "cer_mean": "avg_cer",
        "latency_ms_mean": "avg_latency_ms"
    })
    
    # Trier par WER moyen
    speaker_stats = speaker_stats.sort_values("avg_wer")
    
    # Afficher dans le terminal
    print("\n" + "="*60)
    print("📊 PERFORMANCE PAR SPEAKER")
    print("="*60)
    print(speaker_stats.to_string())
    print("="*60)
    
    # Analyse par méthode
    print("\n📊 PERFORMANCE PAR SPEAKER ET MÉTHODE")
    print("="*60)
    pivot_table = df.groupby(["speaker_id", "method"])["wer"].mean().unstack()
    print(pivot_table.round(4).to_string())
    print("="*60)
    
    # Sauvegarder dans un fichier Markdown
    output_path = Path("docs/speaker_analysis.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 🔊 Speaker Analysis\n\n")
        f.write("## Overview\n\n")
        f.write(f"- **Total Speakers**: {df['speaker_id'].nunique()}\n")
        f.write(f"- **Total Samples**: {len(df)}\n")
        f.write(f"- **Methods**: {', '.join(df['method'].unique())}\n\n")
        
        f.write("## Performance by Speaker\n\n")
        f.write("| Speaker ID | Avg WER | Std WER | Avg CER | Avg Latency (ms) | Samples |\n")
        f.write("|------------|---------|---------|---------|------------------|---------|\n")
        
        for speaker_id, row in speaker_stats.iterrows():
            f.write(f"| {speaker_id} | {row['avg_wer']:.2%} | {row['std_wer']:.4f} | {row['avg_cer']:.2%} | {row['avg_latency_ms']:.0f} | {int(row['sample_count'])} |\n")
        
        f.write("\n## Performance by Speaker and Method\n\n")
        f.write("```text\n")
        f.write(pivot_table.round(4).to_string())
        f.write("\n```\n\n")
        
        f.write("## Key Observations\n\n")
        
        # Trouver le meilleur et le pire speaker
        best_speaker = speaker_stats.idxmin()["avg_wer"]
        worst_speaker = speaker_stats.idxmax()["avg_wer"]
        
        f.write(f"- **Best Speaker**: {best_speaker} (WER: {speaker_stats.loc[best_speaker, 'avg_wer']:.2%})\n")
        f.write(f"- **Worst Speaker**: {worst_speaker} (WER: {speaker_stats.loc[worst_speaker, 'avg_wer']:.2%})\n")
        f.write(f"- **WER Range**: {speaker_stats['avg_wer'].max() - speaker_stats['avg_wer'].min():.2%}\n\n")
        
        # Variance
        if speaker_stats["std_wer"].mean() < 0.1:
            f.write("- ✅ **Low variance**: Performance is consistent across samples\n")
        else:
            f.write("- ⚠️ **High variance**: Performance varies significantly across samples\n")
    
    logger.info(f"✅ Rapport sauvegardé dans {output_path}")
    logger.info(f"📈 Best speaker: {best_speaker}")
    logger.info(f"📉 Worst speaker: {worst_speaker}")

if __name__ == "__main__":
    analyze_speakers()