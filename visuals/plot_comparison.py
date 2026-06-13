#!/usr/bin/env python3
"""
visuals/plot_comparison.py
Generates bar plots comparing WER across preprocessing methods and SNR levels.
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_comparison(csv_path, output_dir="visuals"):
    df = pd.read_csv(csv_path)
    df = df[df["wer"] >= 0]
    if df.empty:
        logger.warning("No valid data to plot.")
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="snr_db", y="wer", hue="method", palette="viridis", errorbar=None)
    plt.title("ASR Performance: Preprocessing Methods vs SNR")
    plt.xlabel("Signal-to-Noise Ratio (dB)")
    plt.ylabel("Word Error Rate (WER)")
    plt.xticks(ticks=[0, 1, 2], labels=["5dB", "10dB", "20dB"])
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Method")
    plt.tight_layout()

    out_path = Path(output_dir) / "preprocessing_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    logger.info(f"Plot saved to {out_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results/preprocessing_comparison.csv")
    parser.add_argument("--output_dir", type=str, default="visuals")
    args = parser.parse_args()
    plot_comparison(args.csv, args.output_dir)

if __name__ == "__main__":
    main()