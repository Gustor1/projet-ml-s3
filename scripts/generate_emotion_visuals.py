#!/usr/bin/env python3
"""
scripts/generate_emotion_visuals.py
Generates the comparison chart visuals/emotion_accuracy.png
from the results/emotion_robustness.csv evaluation data.
"""

import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    csv_path = Path("results/emotion_robustness.csv")
    clean_path = Path("data/emotion_metadata.json")
    out_img = Path("visuals/emotion_accuracy.png")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")
        
    # Load data
    df = pd.read_csv(csv_path)
    
    # Calculate baseline accuracy from clean metadata
    clean_meta = json.load(open(clean_path, "r"))
    # In evaluate_emotion_robustness.py, clean_acc was 35.71%
    # We will load that or calculate it from results
    # Since we mapped them, let's just use the hardcoded baseline from the metadata we obtained
    # or print it. We can read results/emotion_robustness.csv and use the values we reported:
    # Clean baseline: 35.71%
    clean_acc = 10.0 / 28.0 # 10 out of 28 files correct on clean baseline
    
    # Calculate accuracies per condition
    conditions = []
    
    for (noise_type, snr), group in df.groupby(["noise_type", "snr_db"]):
        acc_none = group["correct_none"].mean()
        acc_wiener = group["correct_wiener"].mean()
        acc_spec_sub = group["correct_spec_sub"].mean()
        
        noise_label = "White Noise" if noise_type == "white_gaussian" else "Urban Noise"
        cond_label = f"{noise_label}\n({int(snr)}dB)"
        
        conditions.append({"Condition": cond_label, "Method": "Raw Noisy (None)", "Accuracy": acc_none})
        conditions.append({"Condition": cond_label, "Method": "Wiener Filter", "Accuracy": acc_wiener})
        conditions.append({"Condition": cond_label, "Method": "Spectral Subtraction", "Accuracy": acc_spec_sub})
        
    plot_df = pd.DataFrame(conditions)
    
    # Set style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Use HSL tailored modern palette
    colors = ["#4A90E2", "#50E3C2", "#FF5A5F"]
    ax = sns.barplot(
        x="Condition", 
        y="Accuracy", 
        hue="Method", 
        data=plot_df, 
        palette=colors,
        edgecolor="black",
        linewidth=0.8
    )
    
    # Add horizontal line for clean speech baseline
    plt.axhline(
        y=0.3571, 
        color="#8B572A", 
        linestyle="--", 
        linewidth=1.5, 
        label="Clean Audio Baseline (35.7%)"
    )
    
    # Annotate bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.1%}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha='center', 
                va='bottom', 
                fontsize=9, 
                color='black',
                xytext=(0, 3),
                textcoords='offset points'
            )
            
    # Title and labels
    plt.title("Speech Emotion Recognition (SER) Accuracy under Noise & Preprocessing\n(Model: superb/wav2vec2-base-superb-er | Actor 01 Dataset)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Experimental Condition", fontsize=11, fontweight='bold', labelpad=10)
    plt.ylabel("Classification Accuracy", fontsize=11, fontweight='bold')
    plt.ylim(0, 1.0)
    
    # Format y-axis as percentage
    import matplotlib.ticker as mtick
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    plt.legend(title="Method", loc="upper right")
    plt.tight_layout()
    
    out_img.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_img)
    plt.close()
    
    logger.info(f"Successfully generated visual plot: {out_img}")

if __name__ == "__main__":
    main()
