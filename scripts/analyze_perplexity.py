import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

def analyze_perplexity(csv_path, output_plot=None):
    df = pd.read_csv(csv_path)
    df_valid = df[(df['wer'] >= 0) & (df['perplexity'] > 0)].copy()
    
    print("=" * 70)
    print("PERPLEXITY vs WER ANALYSIS")
    print("=" * 70)
    
    # 1. Summary by SNR and method
    print("\n📊 Perplexity by SNR and Method:")
    print(f"{'SNR':>6s} | {'Method':>20s} | {'Avg WER':>10s} | {'Avg Perp':>10s} | {'Median Perp':>12s}")
    print("-" * 70)
    for snr in sorted(df_valid['snr_db'].unique()):
        for method in ['none', 'wiener', 'spectral_subtraction']:
            subset = df_valid[(df_valid['snr_db'] == snr) & (df_valid['method'] == method)]
            if len(subset) > 0:
                print(f"{snr:>6.0f} | {method:>20s} | {subset['wer'].mean():>9.1%} | {subset['perplexity'].mean():>10.1f} | {subset['perplexity'].median():>12.1f}")
    
    # 2. Correlation
    df_corr = df_valid[df_valid['wer'] < 10]  # Exclude extreme outliers for correlation
    pearson_r, pearson_p = stats.pearsonr(df_corr['perplexity'], df_corr['wer'])
    spearman_r, spearman_p = stats.spearmanr(df_corr['perplexity'], df_corr['wer'])
    
    print(f"\n📈 Correlation (excluding WER > 1000%):")
    print(f"  Pearson r = {pearson_r:.3f} (p = {pearson_p:.4f}) — N = {len(df_corr)}")
    print(f"  Spearman ρ = {spearman_r:.3f} (p = {spearman_p:.4f})")
    
    # 3. Hallucination analysis
    hallucinations = df_valid[df_valid['wer'] > 1.0]
    non_hall = df_valid[df_valid['wer'] <= 1.0]
    
    print(f"\n🚨 Hallucination Analysis (WER > 100%):")
    print(f"  Hallucinations: {len(hallucinations)} samples")
    print(f"    Avg perplexity: {hallucinations['perplexity'].mean():.1f}")
    print(f"    Median: {hallucinations['perplexity'].median():.1f}")
    print(f"    Range: {hallucinations['perplexity'].min():.1f} - {hallucinations['perplexity'].max():.1f}")
    print(f"  Non-hallucinations: {len(non_hall)} samples")
    print(f"    Avg perplexity: {non_hall['perplexity'].mean():.1f}")
    print(f"    Median: {non_hall['perplexity'].median():.1f}")
    print(f"    Max: {non_hall['perplexity'].max():.1f}")
    
    # 4. Threshold analysis
    print(f"\n🔍 Predictive Thresholds:")
    print(f"{'Threshold':>12s} | {'Precision':>10s} | {'Recall':>8s} | {'F1':>6s}")
    print("-" * 50)
    for t in [100, 500, 1000, 2000, 5000, 10000]:
        pred_hall = df_valid['perplexity'] > t
        true_hall = df_valid['wer'] > 1.0
        tp = ((pred_hall) & (true_hall)).sum()
        fp = ((pred_hall) & (~true_hall)).sum()
        fn = ((~pred_hall) & (true_hall)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"{t:>12.0f} | {precision:>10.2%} | {recall:>8.2%} | {f1:>6.2f}")
    
    # 5. Plot
    if output_plot:
        fig, ax = plt.subplots(figsize=(10, 7))
        colors = {'none': '#3498db', 'wiener': '#2ecc71', 'spectral_subtraction': '#e74c3c'}
        for method in ['none', 'wiener', 'spectral_subtraction']:
            subset = df_valid[df_valid['method'] == method]
            ax.scatter(subset['perplexity'], subset['wer'] * 100, 
                       c=colors[method], label=method, alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
        
        # Regression line
        z = np.polyfit(df_corr['perplexity'], df_corr['wer'] * 100, 1)
        p = np.poly1d(z)
        x_line = np.linspace(df_corr['perplexity'].min(), df_corr['perplexity'].max(), 100)
        ax.plot(x_line, p(x_line), "k--", alpha=0.5, linewidth=2, label=f'Linear fit (r={pearson_r:.2f})')
        
        ax.axhline(y=100, color='orange', linestyle=':', linewidth=2, label='Hallucination threshold (WER=100%)')
        ax.axvline(x=1000, color='purple', linestyle='-.', linewidth=1.5, alpha=0.7, label='Perplexity threshold = 1000')
        
        ax.set_xlabel('Decoder Perplexity', fontsize=13)
        ax.set_ylabel('WER (%)', fontsize=13)
        ax.set_title('Decoder Perplexity vs WER: Hallucination Predictor', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_plot, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"\n✅ Plot saved to {output_plot}")
    
    print("\n" + "=" * 70)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Input CSV with perplexity column")
    parser.add_argument("--output", type=str, default=None, help="Output plot path")
    args = parser.parse_args()
    analyze_perplexity(args.csv, args.output)

if __name__ == "__main__":
    main()