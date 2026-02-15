from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from rlnf_rrt.engine.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RLNF-RRT flow model")
    parser.add_argument("--config", type=str, default="configs/train/default.toml")
    args = parser.parse_args()
    train(config_path=args.config)


if __name__ == "__main__":
    main()
