"""Score map 시각화 스크립트.

Usage:
    uv run scripts/debug/score_map_viewer.py --index 0 --split train
    uv run scripts/debug/score_map_viewer.py --index 5 --decay-rate 0.02 --show
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rlnf_rrt.data.dataset import RLNFDataset
from rlnf_rrt.engine.rl_train import _make_gt_score_map
from rlnf_rrt.utils.config import load_toml, resolve_project_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rl/default.toml")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--decay-rate", type=float, default=None, help="decay rate override (default: from config)")
    parser.add_argument("--save", default="outputs/figures/score_map.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    cfg = load_toml(args.config)
    data_cfg = cfg["data"]
    rl_cfg   = cfg["rl"]

    data_root  = resolve_project_path(data_cfg.get("data_root", "data"))
    decay_rate = args.decay_rate or float(rl_cfg.get("gt_score_decay_rate", 0.1))

    ds = RLNFDataset(
        split=args.split,
        data_root=data_root,
        num_points=int(data_cfg["num_points"]),
        clearance=int(data_cfg["clearance"]),
        step_size=int(data_cfg["step_size"]),
    )

    sample = ds[args.index]
    cond_image = sample["cond_image"]   # (3, H, W)
    gt_path    = sample["gt_path"]      # (N, 2)
    start      = sample["start"]        # (2,)
    goal       = sample["goal"]         # (2,)

    H, W = cond_image.shape[-2:]

    # obstacle_mask: channel 0 <= 0.5 → obstacle
    obstacle_mask = (cond_image[0] <= 0.5).numpy().astype(np.uint8)

    score_map = _make_gt_score_map(gt_path, H, W, obstacle_mask=obstacle_mask, decay_rate=decay_rate)

    # --- plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"split={args.split}  index={args.index}  decay_rate={decay_rate}", fontsize=12)

    # 1) 원본 맵 (channel 0)
    ax = axes[0]
    ax.imshow(cond_image[0].numpy(), cmap="gray", origin="upper", vmin=0, vmax=1)
    ax.set_title("Map (ch0: free=1, obstacle=0)")
    ax.axis("off")

    # 2) Score map (RdYlGn: 빨강=-1, 노랑=0, 초록=1)
    ax = axes[1]
    sm = ax.imshow(score_map, cmap="RdYlGn", origin="upper", vmin=score_map.min(), vmax=score_map.max())
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    # GT 경로 overlay
    pts = gt_path.numpy()
    px  = pts[:, 0] * (W - 1)
    py  = pts[:, 1] * (H - 1)
    ax.plot(px, py, "b-", linewidth=1, alpha=0.7, label="GT path")

    # start / goal
    ax.scatter([start[0] * (W-1)], [start[1] * (H-1)], c="cyan",  s=60, zorder=5, label="start")
    ax.scatter([goal[0]  * (W-1)], [goal[1]  * (H-1)], c="magenta", s=60, zorder=5, label="goal")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("Score map  (green=+5, red=-10)")
    ax.axis("off")

    # 3) Score 히스토그램 (obstacle 제외)
    ax = axes[2]
    free_scores = score_map[obstacle_mask == 0].flatten()
    s_min, s_max = float(free_scores.min()), float(free_scores.max())
    ax.hist(free_scores, bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="red", linestyle="--", linewidth=1, label="0")
    ax.axvline(s_min, color="orange", linestyle=":", linewidth=1, label=f"min={s_min:.3f}")
    ax.axvline(s_max, color="green",  linestyle=":", linewidth=1, label=f"max={s_max:.3f}")
    ax.legend(fontsize=7)
    ax.set_title(f"Score distribution (free space only)\n[{s_min:.3f}, {s_max:.3f}]")
    ax.set_xlabel("score")
    ax.set_ylabel("count")

    plt.tight_layout()

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"saved → {save_path}")

    if args.show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
