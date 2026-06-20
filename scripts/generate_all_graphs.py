#!/usr/bin/env python3
"""
scripts/generate_all_graphs.py
Génère toutes les visualisations pour les rapports d'expériences.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150

def load_and_clean(csv_path, exclude_outliers=False):
    df = pd.read_csv(csv_path)
    if exclude_outliers and 'wer' in df.columns:
        # Exclure les hallucinations (WER > 1.0) pour les graphiques de tendance
        df = df[df['wer'] < 1.0].copy()
    return df

def plot_single_experiment(df, title, output_path):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='snr_db', y='wer', hue='method', marker='o', 
                 palette={'none': '#2196F3', 'wiener': '#4CAF50', 'spectral_subtraction': '#F44336'},
                 linewidth=2.5, markersize=8)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Word Error Rate (WER)', fontsize=12)
    plt.xticks([5, 10, 20], ['5dB', '10dB', '20dB'])
    plt.legend(title='Method')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Généré: {output_path}")

def plot_heatmap(df_white, df_pink, df_urban, df_babble, output_path):
    def get_delta(df):
        # Grouper par snr_db et method pour avoir les moyennes (évite les index dupliqués)
        grouped = df.groupby(['snr_db', 'method'])['wer'].mean()
        
        # Extraire les valeurs pour 'none' et 'wiener'
        none_wer = grouped.xs('none', level='method')
        wiener_wer = grouped.xs('wiener', level='method')
        
        # Delta en points de pourcentage (pp)
        delta = (wiener_wer - none_wer) * 100 
        return delta

    deltas = {
        'White Gaussian': get_delta(df_white),
        'Pink (1/f)': get_delta(df_pink),
        'Urban Real': get_delta(df_urban),
        'Babble': get_delta(df_babble)
    }
    
    heatmap_df = pd.DataFrame(deltas).T
    heatmap_df.columns = [f'{int(snr)}dB' for snr in sorted(heatmap_df.columns)]
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap='RdYlGn', center=0, 
                cbar_kws={'label': 'ΔWER (Wiener - None) en points de pourcentage (pp)'})
    plt.title('Impact du Filtre de Wiener sur l\'ASR (ΔWER)', fontsize=14, fontweight='bold')
    plt.ylabel('Type de Bruit')
    plt.xlabel('Niveau SNR')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Généré: {output_path}")

def plot_perplexity(df_perp, output_path):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_perp, x='perplexity', y='wer', hue='method', alpha=0.7, s=80)
    plt.xscale('log')
    plt.title('Perplexité du Décodeur vs WER (Bruit Babble, hors hallucinations)', fontsize=14, fontweight='bold')
    plt.xlabel('Perplexité du Décodeur (échelle log)', fontsize=12)
    plt.ylabel('Word Error Rate (WER)', fontsize=12)
    plt.legend(title='Method')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Généré: {output_path}")

def main():
    results_dir = Path('results')
    visuals_dir = Path('visuals')
    visuals_dir.mkdir(exist_ok=True)
    
    # Chargement des données (Babble nettoyé des outliers pour les tendances)
    df_white = load_and_clean(results_dir / 'preprocessing_comparison.csv')
    df_pink = load_and_clean(results_dir / 'pink_noise_comparison.csv')
    df_urban = load_and_clean(results_dir / 'urban_noise_comparison.csv')
    df_babble = load_and_clean(results_dir / 'babble_noise_comparison.csv', exclude_outliers=True)
    df_perp = load_and_clean(results_dir / 'test_perplexity_v2.csv', exclude_outliers=True)
    
    # 1. Graphiques individuels par expérience
    plot_single_experiment(df_white, 'Exp 2: Bruit Blanc Gaussien', visuals_dir / 'exp2_white_noise.png')
    plot_single_experiment(df_pink, 'Exp 3: Bruit Rose (1/f)', visuals_dir / 'exp3_pink_noise.png')
    plot_single_experiment(df_urban, 'Exp 4: Bruit Urbain Réel', visuals_dir / 'exp4_urban_noise.png')
    plot_single_experiment(df_babble, 'Exp 5: Bruit Babble (Statistiques Robustes)', visuals_dir / 'exp5_babble_noise.png')
    
    # 2. Heatmap comparative (Wiener vs None)
    plot_heatmap(df_white, df_pink, df_urban, df_babble, visuals_dir / 'wiener_impact_heatmap.png')
    
    # 3. Graphique de perplexité (Insight mécanistique)
    plot_perplexity(df_perp, visuals_dir / 'perplexity_vs_wer.png')
    
    print("\n🎉 Toutes les visualisations ont été générées avec succès dans le dossier 'visuals/' !")

if __name__ == '__main__':
    main()