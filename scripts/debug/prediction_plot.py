from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rlnf_rrt.data.dataset import RLNFDataset
from rlnf_rrt.utils.config import load_toml, resolve_project_path


def _find_latest_prediction_dir(eval_cfg: dict) -> Path:
    pred_root = resolve_project_path(eval_cfg.get("prediction_dir", "outputs/predictions"))
    if not pred_root.exists():
        raise FileNotFoundError(f"Prediction root not found: {pred_root}")

    ckpt_stem = Path(eval_cfg["checkpoint"]).stem
    entries = sorted(
        [p for p in pred_root.iterdir() if p.is_dir() and p.name.startswith(f"{ckpt_stem}_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not entries:
        raise FileNotFoundError(
            f"No prediction directory found under {pred_root} with prefix '{ckpt_stem}_'"
        )
    return entries[0]


def _load_predictions(pred_dir: Path, ds_idx: int) -> list[np.ndarray]:
    files = sorted(pred_dir.glob(f"pred_{ds_idx:07d}_s*.npy"))
    return [np.load(f).astype(np.float32) for f in files]


def _indices_with_predictions(pred_dir: Path) -> list[int]:
    out: set[int] = set()
    for f in pred_dir.glob("pred_*_s*.npy"):
        try:
            out.add(int(f.name.split("_")[1]))
        except (ValueError, IndexError):
            continue
    return sorted(out)


def _choose_indices(
    available: list[int],
    explicit: list[int] | None,
    num_examples: int,
    seed: int,
) -> list[int]:
    if explicit:
        picked = [i for i in explicit if i in set(available)]
        if not picked:
            raise RuntimeError("None of the requested indices have prediction files.")
        return picked[:num_examples]

    if len(available) <= num_examples:
        return available

    rng = random.Random(seed)
    return sorted(rng.sample(available, k=num_examples))


def main() -> None:
    parser = argparse.ArgumentParser(description="Matplotlib prediction figure generator")
    parser.add_argument("--eval-config", type=str, default="configs/eval/default.toml")
    parser.add_argument("--pred-dir", type=str, default=None)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save", type=str, default="outputs/figures/prediction_grid.png")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    cfg = load_toml(args.eval_config)
    data_cfg = cfg["data"]
    eval_cfg = cfg["eval"]

    pred_dir = resolve_project_path(args.pred_dir) if args.pred_dir else _find_latest_prediction_dir(eval_cfg)
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")

    ds = RLNFDataset(
        split=str(data_cfg.get("split", "test")),
        data_root=resolve_project_path(data_cfg.get("data_root", "data")),
        noise_std=0.0,
        num_points=int(data_cfg["num_points"]),
        clearance=int(data_cfg["clearance"]),
        step_size=int(data_cfg["step_size"]),
    )
    if len(ds) == 0:
        raise RuntimeError("Dataset is empty with current config filters.")

    available = _indices_with_predictions(pred_dir)
    if not available:
        raise RuntimeError(f"No prediction files found in {pred_dir}")

    picked = _choose_indices(available, args.indices, args.num_examples, args.seed)
    if not picked:
        raise RuntimeError("No valid indices selected.")

    n = len(picked)
    ncols = 2
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 8 * nrows))
    axes_arr = np.atleast_1d(axes).reshape(-1)

    # free-space=light gray, obstacle=dark gray
    cmap = ListedColormap(["#3b3336", "#dfdfdf"])

    for ax_i, ax in enumerate(axes_arr):
        if ax_i >= n:
            ax.axis("off")
            continue

        ds_idx = picked[ax_i]
        sample = ds[ds_idx]

        map_np = sample["map"].squeeze(0).numpy()
        start = sample["start"].numpy()
        goal = sample["goal"].numpy()
        gt = sample["gt_path"].numpy()

        preds = _load_predictions(pred_dir, ds_idx)
        if args.max_samples is not None:
            preds = preds[: max(0, args.max_samples)]

        ax.imshow(map_np, cmap=cmap, vmin=0.0, vmax=1.0, origin="lower", extent=(0, 1, 0, 1), interpolation="nearest")

        if len(gt) > 0:
            ax.scatter(gt[:, 0], gt[:, 1], s=10, c="#18a71e", alpha=0.65, label="GT Path", edgecolors="none")

        if preds:
            pred_all = np.concatenate(preds, axis=0)
            ax.scatter(
                pred_all[:, 0],
                pred_all[:, 1],
                s=12,
                c="#1a2cff",
                alpha=0.45,
                label=f"Samples (n={len(preds)})",
                edgecolors="none",
            )

        ax.scatter([start[0]], [start[1]], s=220, c="red", label="Start", edgecolors="none", zorder=3)
        ax.scatter([goal[0]], [goal[1]], s=220, c="lime", label="Goal", edgecolors="none", zorder=3)

        ax.set_title(f"Example {ds_idx}", fontsize=18, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(axis="both", labelsize=12)

        if ax_i in (0, n - 1):
            ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=13)

    fig.tight_layout()

    save_path = resolve_project_path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved figure: {save_path}")
    print(f"Prediction dir: {pred_dir}")
    print(f"Indices: {picked}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
