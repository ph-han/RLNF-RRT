from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from rlnf_rrt.engine.evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RLNF-RRT flow model")
    parser.add_argument("--config", type=str, default="configs/eval/default.toml")
    args = parser.parse_args()
    evaluate(config_path=args.config)


if __name__ == "__main__":
    main()
