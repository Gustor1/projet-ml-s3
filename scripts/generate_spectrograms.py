#!/usr/bin/env python3
"""
scripts/generate_spectrograms.py
Génère des spectrogrammes comparatifs pour visualiser la distortion spectrale
causée par le filtre de Wiener sur le bruit rose.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, wiener
from scipy.io import wavfile
from pathlib import Path

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

def load_audio(file_path):
    """Charge un fichier audio WAV."""
    sr, audio = wavfile.read(file_path)
    if audio.dtype == np.int16:
        audio = audio / 32768.0
    elif audio.dtype == np.int32:
        audio = audio / 2147483648.0
    return audio, sr

def compute_spectrogram(audio, sr, nperseg=512, noverlap=256):
    """Calcule le spectrogramme STFT."""
    f, t, Zxx = stft(audio, fs=sr, nperseg=nperseg, noverlap=noverlap, window='hann')
    magnitude_db = 20 * np.log10(np.abs(Zxx) + 1e-10)
    return f, t, magnitude_db

def plot_spectrogram(ax, f, t, magnitude_db, title, sr, formant_bands=(1000, 4000)):
    """Trace un spectrogramme avec les bandes de formants."""
    im = ax.pcolormesh(t, f, magnitude_db, shading='gouraud', cmap='viridis', vmin=-80, vmax=0)
    ax.set_ylabel('Frequency (Hz)', fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_ylim(0, sr/2)
    
    # Lignes de formants
    for fb in formant_bands:
        ax.axhline(y=fb, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Formant band' if fb == formant_bands[0] else '')
    
    # Colorbar
    plt.colorbar(im, ax=ax, label='Magnitude (dB)')

def main():
    """Génère les spectrogrammes comparatifs."""
    output_dir = Path("visuals")
    output_dir.mkdir(exist_ok=True)
    
    # Fichiers générés par extract_audio_samples.py
    pink_noisy_file = Path("data/audio_samples/6930-75918-0000_pink_snr5dB.wav")
    wiener_filtered_file = Path("data/audio_samples/6930-75918-0000_pink_snr5dB_wiener.wav")
    
    # Charger les audio
    print("🔊 Chargement des fichiers audio...")
    pink_audio, sr = load_audio(pink_noisy_file)
    wiener_audio, _ = load_audio(wiener_filtered_file)
    
    # Calculer les spectrogrammes
    print("📊 Calcul des spectrogrammes...")
    f, t, pink_spec = compute_spectrogram(pink_audio, sr)
    _, _, wiener_spec = compute_spectrogram(wiener_audio, sr)
    
    # Créer la figure comparative (2 panneaux : noisy vs wiener)
    print("🎨 Génération des visualisations...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    plot_spectrogram(axes[0], f, t, pink_spec, "(a) Pink Noise (5dB SNR) - Before Wiener", sr)
    plot_spectrogram(axes[1], f, t, wiener_spec, "(b) After Wiener Filter - Spectral Tilt Distortion", sr)
    
    plt.suptitle("Spectral Tilt Distortion: Wiener Filter on Pink Noise\nDemonstrating high-frequency formant attenuation (F2/F3 at 2.5-3.5kHz)", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / "spectrogram_pink_wiener.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Spectrogramme sauvegardé: {output_path}")
    plt.close()
    
    print("\n🎉 Visualisation générée avec succès!")
    
if __name__ == "__main__":
    main()