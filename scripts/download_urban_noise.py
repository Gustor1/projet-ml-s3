#!/usr/bin/env python3
"""
scripts/download_urban_noise.py
Télécharge des fichiers de bruit urbain libre de droits depuis Freesound ou Zenodo.
"""
import urllib.request
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sources de bruit urbain libre de droits
# Option 1: DEMAND dataset (Zenodo) - bruit réel enregistré
# Option 2: Freesound via API (nécessite API key)
# Option 3: Samples depuis GitHub repos publics

URBAN_NOISE_URLS = {
    "traffic.wav": "https://zenodo.org/record/1226381/files/traffic.wav?download=1",
    "cafe.wav": "https://zenodo.org/record/1226381/files/cafe.wav?download=1",
    "street.wav": "https://zenodo.org/record/1226381/files/street.wav?download=1",
}

def download_file(url, output_path):
    """Télécharge un fichier depuis une URL."""
    try:
        logger.info(f"Téléchargement: {output_path.name}")
        urllib.request.urlretrieve(url, output_path)
        logger.info(f"✅ Téléchargé: {output_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur téléchargement {output_path}: {e}")
        return False

def main():
    output_dir = Path("data/urban_noise")
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for filename, url in URBAN_NOISE_URLS.items():
        output_path = output_dir / filename
        if output_path.exists():
            logger.info(f"⏭️  Déjà existant: {output_path}")
            success_count += 1
        else:
            if download_file(url, output_path):
                success_count += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Téléchargement terminé: {success_count}/{len(URBAN_NOISE_URLS)} fichiers")
    logger.info(f"📁 Dossier: {output_dir}")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()