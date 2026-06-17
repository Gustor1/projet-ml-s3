#!/usr/bin/env python3
"""
scripts/generate_urban_noise.py
Génère des bruits réalistes imitant des environnements urbains :
- traffic.wav : Bruit de trafic (brown noise + modulation basse fréquence)
- cafe.wav : Brouhaha de café (bruit filtré passe-bande + modulation aléatoire)
- street.wav : Rue animée (mix brown + pink + impulsions aléatoires)
"""
import numpy as np
import soundfile as sf
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_traffic_noise(duration_s, sr=16000, seed=42):
    """
    Bruit de trafic : brown noise (1/f²) avec modulation lente.
    Imitation du ronronnement des moteurs et du roulement sur la route.
    """
    np.random.seed(seed)
    n_samples = int(duration_s * sr)
    
    # Brown noise (intégration de bruit blanc)
    white = np.random.randn(n_samples)
    brown = np.cumsum(white)
    brown = brown / np.max(np.abs(brown))
    
    # Modulation lente (0.1-2 Hz) pour simuler le passage de véhicules
    t = np.arange(n_samples) / sr
    modulation = 0.7 + 0.3 * np.sin(2 * np.pi * 0.5 * t) * np.sin(2 * np.pi * 0.13 * t)
    
    traffic = brown * modulation
    return traffic / np.max(np.abs(traffic))

def generate_cafe_noise(duration_s, sr=16000, seed=43):
    """
    Brouhaha de café : bruit filtré passe-bande (300-3000 Hz) + modulation aléatoire.
    Imitation des conversations lointaines et du bruit de vaisselle.
    """
    np.random.seed(seed)
    n_samples = int(duration_s * sr)
    
    # Bruit blanc filtré passe-bande
    white = np.random.randn(n_samples)
    
    # Filtre passe-bande simple (moyenne mobile)
    from scipy.signal import butter, filtfilt
    nyq = sr / 2
    low, high = 300 / nyq, 3000 / nyq
    b, a = butter(4, [low, high], btype='band')
    filtered = filtfilt(b, a, white)
    
    # Modulation aléatoire rapide (conversations)
    modulation = 0.5 + 0.5 * np.abs(np.random.randn(n_samples))
    modulation = np.convolve(modulation, np.ones(100)/100, mode='same')  # Lissage
    
    cafe = filtered * modulation
    return cafe / np.max(np.abs(cafe))

def generate_street_noise(duration_s, sr=16000, seed=44):
    """
    Rue animée : mix brown + pink + impulsions aléatoires (klaxons, pas).
    """
    np.random.seed(seed)
    n_samples = int(duration_s * sr)
    
    # Brown noise (fond de trafic)
    white = np.random.randn(n_samples)
    brown = np.cumsum(white)
    brown = brown / np.max(np.abs(brown))
    
    # Pink noise (vent, environnement)
    num_rows = 16
    pink_array = np.random.randn(num_rows, n_samples // num_rows + 1)
    pink = np.sum(pink_array, axis=0)
    pink = np.repeat(pink, num_rows)[:n_samples]
    pink = pink / np.max(np.abs(pink))
    
    # Impulsions aléatoires (klaxons, pas)
    impulses = np.zeros(n_samples)
    num_impulses = int(duration_s * 2)  # ~2 impulsions par seconde
    impulse_positions = np.random.randint(0, n_samples, num_impulses)
    for pos in impulse_positions:
        if pos + 100 < n_samples:
            impulses[pos:pos+100] += np.random.randn(100) * 0.3
    
    # Mix
    street = 0.6 * brown + 0.3 * pink + 0.1 * impulses
    return street / np.max(np.abs(street))

def main():
    output_dir = Path("data/urban_noise")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    duration_s = 30  # 30 secondes de bruit (suffisant pour boucler)
    sr = 16000
    
    generators = {
        "traffic.wav": generate_traffic_noise,
        "cafe.wav": generate_cafe_noise,
        "street.wav": generate_street_noise,
    }
    
    for filename, gen_func in generators.items():
        output_path = output_dir / filename
        if output_path.exists():
            logger.info(f"⏭️  Déjà existant: {output_path}")
            continue
        
        logger.info(f"🔊 Génération: {filename} ({duration_s}s)")
        noise = gen_func(duration_s, sr)
        sf.write(str(output_path), noise, sr)
        logger.info(f"✅ Créé: {output_path}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Bruits urbains générés dans {output_dir}")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()