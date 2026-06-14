# Local Audio Preprocessing for Better ASR Performance

This project implements a complete pipeline for **local audio preprocessing** (noise reduction, VAD, enhancement, echo handling, etc.) to improve the performance of an **ASR (Automatic Speech Recognition)** model.

The repository is organized by roles:
- `preprocessing/` – audio preprocessing modules
- `asr/` – ASR model integration and evaluation
- `experiments/` – experiments, ablation studies, and plots
- `optimization/` – profiling and real-time performance
- `demo/` – interactive demo and final video assets
- `configs/` – configuration files
- `utils/` – shared utilities

## Setup (Ubuntu / WSL)

```bash
git clone https://github.com/Gustor1/projet-ml-s3.git
cd projet-ml-s3

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Run baseline pipeline entrypoint

For now, the main script only loads the config and prepares the future pipeline integration:

```bash
python main.py --config configs/config.yaml
```

You should see:

```text
[INFO] Loaded config for project: local-audio-preprocessing-asr
[INFO] Placeholder main() finished. Pipeline integration will be added later.
```

## Project structure (high level)

```text
projet-ml-s3/
  asr/
  preprocessing/
  experiments/
  optimization/
  demo/
  utils/
  configs/
  scripts/
  data/
    raw/
    processed/
  models/
  logs/
  main.py
  requirements.txt
  README.md
```

Each role will progressively fill its own directory with code and experiments.
