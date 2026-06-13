import argparse
from pathlib import Path

import yaml


def load_config(config_path: str):
  config_path = Path(config_path)
  if not config_path.exists():
    raise FileNotFoundError(f"Config file not found: {config_path}")
  with config_path.open("r", encoding="utf-8") as f:
    return yaml.safe_load(f)


def main(config_path: str):
  config = load_config(config_path)
  project_name = config["project"]["name"]
  print(f"[INFO] Loaded config for project: {project_name}")

  # TODO: branch to different pipeline steps when they are implemented
  # Example (plus tard, par les autres rôles) :
  # from preprocessing.pipeline import run_preprocessing
  # from asr.whisper_wrapper import run_asr
  #
  # run_preprocessing(config)
  # run_asr(config)

  print("[INFO] Placeholder main() finished. Pipeline integration will be added later.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Local audio preprocessing + ASR pipeline")
  parser.add_argument(
    "--config",
    type=str,
    default="configs/config.yaml",
    help="Path to YAML config file",
  )
  args = parser.parse_args()
  main(args.config)
