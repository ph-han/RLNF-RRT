from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rlnf_rrt.utils.config import load_toml, resolve_project_path


def _read_row(meta_csv: Path, idx: int) -> dict[str, str]:
    with open(meta_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if idx < 0 or idx >= len(rows):
        raise IndexError(f"index out of range: {idx} (dataset size={len(rows)})")
    return rows[idx]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export RLNF dataset sample as neural-rrt map format (free=0, obstacle=1, start=2, goal=3)."
    )
    parser.add_argument("--eval-config", type=str, default="configs/eval/default.toml")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=str, required=True, help="Output .png path")
    args = parser.parse_args()

    cfg = load_toml(args.eval_config)
    data_cfg = cfg["data"]
    split = args.split if args.split is not None else str(data_cfg.get("split", "test"))
    data_root = resolve_project_path(data_cfg.get("data_root", "data"))

    split_dir = data_root / split
    meta_csv = split_dir / "meta.csv"
    row = _read_row(meta_csv, args.index)

    map_path = split_dir / "map" / row["map_file"]
    sg_path = split_dir / "start_goal" / row["start_goal_file"]

    map_np = np.array(Image.open(map_path).convert("L"), dtype=np.uint8)
    start_goal = np.load(sg_path).astype(np.int32)  # shape (2,2), pixel coords

    out = np.zeros_like(map_np, dtype=np.uint8)
    out[map_np == 0] = 1

    h, w = out.shape
    sx = int(np.clip(start_goal[0, 0], 0, w - 1))
    sy = int(np.clip(start_goal[0, 1], 0, h - 1))
    gx = int(np.clip(start_goal[1, 0], 0, w - 1))
    gy = int(np.clip(start_goal[1, 1], 0, h - 1))
    out[sy, sx] = 2
    out[gy, gx] = 3

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = resolve_project_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, mode="L").save(output_path)

    print(f"saved_map={output_path}")
    print(f"split={split} index={args.index}")
    print(f"start=({sx},{sy}) goal=({gx},{gy})")


if __name__ == "__main__":
    main()
