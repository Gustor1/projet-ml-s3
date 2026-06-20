#!/usr/bin/env python3
"""
scripts/generate_all_visuals.py
Génère des visualisations pour toutes les expériences de comparaison de bruit.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.style.use('seaborn-v0_8-darkgrid')

def create_comparison_plot(csv_path, output_path, title, noise_type):
    """Crée un graphique de comparaison pour un type de bruit."""
    df = pd.read_csv(csv_path)
    
    # Filtrer les outliers si nécessaire (pour babble)
    if 'babble' in csv_path:
        df = df[df['wer'] < 1.0]
    
    # Calculer les moyennes par SNR et méthode
    pivot = df.groupby(['snr_db', 'method'])['wer'].mean().unstack()
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Couleurs pour chaque méthode
    colors = {'none': '#2196F3', 'wiener': '#4CAF50', 'spectral_subtraction': '#F44336'}
    markers = {'none': 'o', 'wiener': 's', 'spectral_subtraction': '^'}
    
    snr_levels = sorted(pivot.index)
    
    for method in ['none', 'wiener', 'spectral_subtraction']:
        if method in pivot.columns:
            values = [pivot.loc[snr, method] * 100 for snr in snr_levels]
            ax.plot(snr_levels, values, marker=markers[method], color=colors[method], 
                   linewidth=2.5, markersize=10, label=method.replace('_', ' ').title())
    
    # Configuration du graphique
    ax.set_xlabel('SNR (dB)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Word Error Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'{title}\n{noise_type}', fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(snr_levels)
    ax.set_ylim(0, max([pivot.max().max() * 100 * 1.1, 50]))
    
    # Ajouter des annotations pour les valeurs
    for method in ['none', 'wiener', 'spectral_subtraction']:
        if method in pivot.columns:
            for snr in snr_levels:
                value = pivot.loc[snr, method] * 100
                ax.annotate(f'{value:.1f}%', 
                           xy=(snr, value), 
                           xytext=(5, 5), 
                           textcoords='offset points',
                           fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Graphique sauvegardé: {output_path}")
    plt.close()

def main():
    visuals_dir = Path("visuals")
    visuals_dir.mkdir(exist_ok=True)
    
    # Expérience 3: Pink Noise
    print("📊 Génération du graphique Pink Noise...")
    create_comparison_plot(
        "results/pink_noise_comparison.csv",
        visuals_dir / "pink_noise_comparison.png",
        "Experiment 3: Preprocessing Comparison",
        "Pink Noise (1/f Spectrum)"
    )
    
    # Expérience 4: Urban Noise
    print("📊 Génération du graphique Urban Noise...")
    create_comparison_plot(
        "results/urban_noise_comparison.csv",
        visuals_dir / "urban_noise_comparison.png",
        "Experiment 4: Preprocessing Comparison",
        "Real Urban Noise (Traffic, Café, Street)"
    )
    
    # Expérience 5: Babble Noise
    print("📊 Génération du graphique Babble Noise...")
    create_comparison_plot(
        "results/babble_noise_comparison.csv",
        visuals_dir / "babble_noise_comparison.png",
        "Experiment 5: Preprocessing Comparison",
        "Babble Noise (Cocktail Party Problem)"
    )
    
    # Graphique comparatif final (tous les types de bruit)
    print("📊 Génération du graphique comparatif final...")
    create_final_comparison_plot(visuals_dir / "all_noise_types_comparison.png")
    
    print("\n✅ Tous les graphiques ont été générés avec succès!")

def create_final_comparison_plot(output_path):
    """Crée un graphique comparatif de tous les types de bruit."""
    # Charger tous les CSV
    files = {
        'White Gaussian': 'results/preprocessing_comparison.csv',
        'Pink Noise': 'results/pink_noise_comparison.csv',
        'Urban Noise': 'results/urban_noise_comparison.csv',
        'Babble Noise': 'results/babble_noise_comparison.csv'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (noise_name, csv_path) in enumerate(files.items()):
        df = pd.read_csv(csv_path)
        
        # Filtrer outliers pour babble
        if 'babble' in csv_path:
            df = df[df['wer'] < 1.0]
        
        # Calculer moyennes par méthode
        method_means = df.groupby('method')['wer'].mean() * 100
        
        # Graphique en barres
        methods = ['none', 'wiener', 'spectral_subtraction']
        values = [method_means.get(m, 0) for m in methods]
        colors = ['#2196F3', '#4CAF50', '#F44336']
        
        bars = axes[idx].bar(methods, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        axes[idx].set_title(f'{noise_name}', fontsize=14, fontweight='bold')
        axes[idx].set_ylabel('Average WER (%)', fontsize=11)
        axes[idx].set_ylim(0, 50)
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Ajouter les valeurs sur les barres
        for bar, value in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                          f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Rotation des labels
        axes[idx].set_xticklabels(['None\n(Baseline)', 'Wiener', 'Spectral\nSubtraction'], fontsize=10)
    
    plt.suptitle('Final Comparison: Preprocessing Effectiveness Across All Noise Types', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Graphique comparatif final sauvegardé: {output_path}")
    plt.close()

if __name__ == "__main__":
    main()