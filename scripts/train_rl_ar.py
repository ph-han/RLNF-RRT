from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from rlnf_rrt.engine.rl_train_ar import rl_train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="REINFORCE with Autoregressive sub-goal policy (Count-first AR)"
    )
    parser.add_argument("--config", type=str, default="configs/rl/reinforce_ar.toml")
    args = parser.parse_args()
    rl_train(config_path=args.config)


if __name__ == "__main__":
    main()
