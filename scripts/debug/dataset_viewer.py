import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rlnf_rrt.data.dataset import RLNFDataset


def _to_pixel_coords(points_xy: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    points_xy = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if points_xy.size == 0:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)

    if float(points_xy.max()) > 1.0 or float(points_xy.min()) < 0.0:
        points_xy = points_xy.copy()
        points_xy[:, 0] = points_xy[:, 0] / max(1, (w - 1))
        points_xy[:, 1] = points_xy[:, 1] / max(1, (h - 1))

    px = np.clip(points_xy[:, 0] * (w - 1), 0, w - 1).astype(np.int32)
    py = np.clip(points_xy[:, 1] * (h - 1), 0, h - 1).astype(np.int32)
    return px, py


def draw_sample(ax: plt.Axes, sample: dict[str, np.ndarray | object], idx: int, total: int) -> None:
    cond_tensor = sample["cond_image"][:1]  # (1, H, W)
    start_tensor = sample["start"]
    goal_tensor = sample["goal"]
    path_tensor = sample["gt_path"]

    cond_np = cond_tensor.numpy()
    map_np = cond_np[:1, :, :].squeeze(0)
    start = start_tensor.numpy()
    goal = goal_tensor.numpy()
    path = path_tensor.numpy()

    h, w = map_np.shape
    ax.clear()
    ax.imshow(map_np, cmap="gray", vmin=0.0, vmax=1.0, origin="lower", interpolation="nearest")

    if len(path) > 0:
        px, py = _to_pixel_coords(path, h, w)
        ax.scatter(px, py, s=8, c="green", alpha=0.75, label="GT Path", edgecolors="none")

    sx = int(np.clip(start[0] * (w - 1), 0, w - 1))
    sy = int(np.clip(start[1] * (h - 1), 0, h - 1))
    gx = int(np.clip(goal[0] * (w - 1), 0, w - 1))
    gy = int(np.clip(goal[1] * (h - 1), 0, h - 1))
    ax.scatter([sx], [sy], s=55, c="red", label="Start", edgecolors="none", zorder=3)
    ax.scatter([gx], [gy], s=55, c="deepskyblue", label="Goal", edgecolors="none", zorder=3)

    ax.set_title(f"idx: {idx}/{total - 1}   map: {h}x{w}   gt_path: {len(path)}")
    ax.set_xlabel("green=gt_path   keys: n/right=next, p/left=prev, j=jump, q/esc=quit")
    ax.set_xlim(0, w - 1)
    ax.set_ylim(0, h - 1)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)


def main() -> None:
    parser = argparse.ArgumentParser(description="RLNF dataset viewer")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test", "test_circle"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--num-points", type=int, default=128)
    parser.add_argument("--clearance", type=int, default=2)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=800)
    args = parser.parse_args()

    dataset = RLNFDataset(
        split=args.split,
        noise_std=args.noise_std,
        num_points=args.num_points,
        clearance=args.clearance,
        step_size=args.step_size,
    )

    if len(dataset) == 0:
        print("No samples found with current filters.")
        return

    idx = max(0, min(args.index, len(dataset) - 1))
    fig_size = max(float(args.window_size) / 100.0, 4.0)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    try:
        fig.canvas.manager.set_window_title(f"RLNF Dataset Viewer [{args.split}]")
    except Exception:
        pass

    print("Viewer started.")
    print("Controls: n/right(next), p/left(prev), j(jump), q/esc(quit)")

    state = {"idx": idx}

    def redraw() -> None:
        sample = dataset[state["idx"]]
        draw_sample(ax, sample, state["idx"], len(dataset))
        fig.tight_layout()
        fig.canvas.draw_idle()

    def on_key(event) -> None:
        if event.key in ("n", "right"):
            state["idx"] = (state["idx"] + 1) % len(dataset)
            redraw()
        elif event.key in ("p", "left"):
            state["idx"] = (state["idx"] - 1 + len(dataset)) % len(dataset)
            redraw()
        elif event.key == "j":
            raw = input(f"Jump to index (0~{len(dataset)-1}): ").strip()
            if raw.isdigit():
                state["idx"] = max(0, min(int(raw), len(dataset) - 1))
                redraw()
        elif event.key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show()


if __name__ == "__main__":
    main()
