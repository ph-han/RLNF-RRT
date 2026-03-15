from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from rlnf_rrt.engine.rl_train import rl_train


def main() -> None:
    parser = argparse.ArgumentParser(description="REINFORCE sub-goal policy training")
    parser.add_argument("--config", type=str, default="configs/rl/default.toml")
    args = parser.parse_args()
    rl_train(config_path=args.config)


if __name__ == "__main__":
    main()
