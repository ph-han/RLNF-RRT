from __future__ import annotations

import argparse
from pathlib import Path
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


def _load_sample_predictions(pred_dir: Path, ds_idx: int) -> list[np.ndarray]:
    files = sorted(pred_dir.glob(f"pred_{ds_idx:07d}_s*.npy"))
    preds: list[np.ndarray] = []
    for f in files:
        preds.append(np.load(f).astype(np.float32))
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Prediction overlay viewer")
    parser.add_argument("--eval-config", type=str, default="configs/eval/default.toml")
    parser.add_argument("--pred-dir", type=str, default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=900)
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
        print("Dataset is empty with current config filters.")
        return

    idx = max(0, min(args.index, len(ds) - 1))

    # Matplotlib setup for 4 subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    axes_arr = np.atleast_1d(axes).reshape(-1)
    
    try:
        fig.canvas.manager.set_window_title("RLNF Prediction Viewer")
    except Exception:
        pass
    
    cmap = ListedColormap(["#3b3336", "#dfdfdf"])

    print(f"Prediction dir: {pred_dir}")
    print("Controls: n/right(next), p/left(prev), j(jump), q/esc(quit)")
    
    state = {"idx": idx}

    def draw_page() -> None:
        current_idx = state["idx"]
        fig.suptitle(f"Prediction Viewer - Index {current_idx} to {current_idx + 3} / {len(ds) - 1}", fontsize=16)

        for ax_i, ax in enumerate(axes_arr):
            ax.clear()
            ds_idx = current_idx + ax_i

            if ds_idx >= len(ds):
                ax.axis("off")
                continue

            ax.axis("on")
            sample = ds[ds_idx]
            
            map_np = sample["map"].squeeze(0).numpy()
            start = sample["start"].numpy()
            goal = sample["goal"].numpy()
            gt = sample["gt_path"].numpy()

            preds = _load_sample_predictions(pred_dir, ds_idx)

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

            ax.set_title(f"Example {ds_idx}", fontsize=14, fontweight="bold")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(axis="both", labelsize=10)
            
            if ax_i == 0:
                ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=10)

        fig.tight_layout()
        fig.canvas.draw_idle()

    def on_key(event) -> None:
        if event.key in ("n", "right", "down"):
            state["idx"] = min(len(ds) - 1, state["idx"] + 4)
            draw_page()
        elif event.key in ("p", "left", "up"):
            state["idx"] = max(0, state["idx"] - 4)
            draw_page()
        elif event.key == "j":
            raw = input(f"Jump to index (0~{len(ds)-1}): ").strip()
            if raw.isdigit():
                state["idx"] = max(0, min(int(raw), len(ds) - 1))
                draw_page()
        elif event.key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw_page()
    plt.show()


if __name__ == "__main__":
    main()
