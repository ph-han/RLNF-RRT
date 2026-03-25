from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from rlnf_rrt.engine.sl_train import sl_train


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised learning: gt_path midpoint prediction")
    parser.add_argument("--config", type=str, default="configs/sl/default.toml")
    args = parser.parse_args()
    sl_train(config_path=args.config)


if __name__ == "__main__":
    main()
