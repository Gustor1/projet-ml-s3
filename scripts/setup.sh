#!/usr/bin/env bash
set -e

echo "[INFO] Creating virtual environment .venv (if not exists)..."
python3 -m venv .venv

echo "[INFO] Activating virtual environment..."
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[INFO] Installing Python dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[INFO] Running a quick sanity check..."
python main.py --config configs/config.yaml

echo "[INFO] Setup completed successfully."
