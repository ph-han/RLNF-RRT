"""SplitDecisionAgent RL 학습 진입점."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlnf_rrt.engine.train_recursive_rl import recurive_subgoal_train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SplitDecisionAgent (recursive binary split)")
    parser.add_argument("--config", type=str, default="configs/split/default.toml")
    args = parser.parse_args()
    recurive_subgoal_train(config_path=args.config)


if __name__ == "__main__":
    main()
