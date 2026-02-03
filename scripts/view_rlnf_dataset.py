"""
Example:
  python scripts/view_rlnf_dataset.py --split train --idx 0 --show
  python scripts/view_rlnf_dataset.py --split valid --num 10 --save_dir outputs/dataset_preview
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

file_path = Path(__file__).resolve()
project_root = file_path.parents[1]
sys.path.append(str(project_root / "src"))

from rlnf_rrt.data_pipeline.custom_dataset import RLNFDataset


def _to_xy(points: np.ndarray, w: int, h: int) -> np.ndarray:
    pts = points.reshape(-1, 2)
    xy = pts.copy()
    xy[:, 0] = xy[:, 0] * w
    xy[:, 1] = xy[:, 1] * h
    return xy


def render_sample(sample: dict, idx: int, show_gt: bool = True):
    map_img = sample["map"].squeeze().numpy()
    h, w = map_img.shape

    start = _to_xy(sample["start"].numpy(), w, h)[0]
    goal = _to_xy(sample["goal"].numpy(), w, h)[0]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(map_img, cmap="gray", vmin=0, vmax=1, origin="lower")
    ax.scatter(start[0], start[1], c="lime", s=80, marker="o", label="start", edgecolors="black")
    ax.scatter(goal[0], goal[1], c="red", s=80, marker="X", label="goal", edgecolors="black")
    if show_gt and "gt" in sample:
        gt = _to_xy(sample["gt"].numpy(), w, h)
        ax.scatter(gt[:, 0], gt[:, 1], c="cyan", s=10, alpha=0.6, label="gt")
    ax.set_title(f"Sample {idx}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


class DatasetViewer:
    def __init__(self, dataset, start_idx: int = 0, show_gt: bool = True):
        self.dataset = dataset
        self.idx = start_idx
        self.show_gt = show_gt

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.update_plot()
        plt.show()

    def update_plot(self):
        self.ax.clear()
        sample = self.dataset[self.idx]
        map_img = sample["map"].squeeze().numpy()
        h, w = map_img.shape

        start = _to_xy(sample["start"].numpy(), w, h)[0]
        goal = _to_xy(sample["goal"].numpy(), w, h)[0]

        self.ax.imshow(map_img, cmap="gray", vmin=0, vmax=1, origin="lower")
        self.ax.scatter(start[0], start[1], c="lime", s=80, marker="o", label="start", edgecolors="black")
        self.ax.scatter(goal[0], goal[1], c="red", s=80, marker="X", label="goal", edgecolors="black")
        if self.show_gt and "gt" in sample:
            print(sample["gt"].numpy())
            gt = _to_xy(sample["gt"].numpy(), w, h)
            self.ax.scatter(gt[:, 0], gt[:, 1], c="cyan", s=10, alpha=0.6, label="gt")
            
        self.ax.set_title(f"Sample {self.idx}")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.legend(loc="upper right")
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key in ["n", " ", "right"]:
            self.idx = (self.idx + 1) % len(self.dataset)
            self.update_plot()
        elif event.key in ["p", "left"]:
            self.idx = (self.idx - 1) % len(self.dataset)
            self.update_plot()
        elif event.key in ["q", "escape"]:
            plt.close(self.fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize RLNFDataset samples.")
    parser.add_argument("--dataset_root", type=str, default="data")
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"])
    parser.add_argument("--idx", type=int, default=0, help="Single sample index (used if --num not set)")
    parser.add_argument("--num", type=int, default=None, help="Number of samples to save (from start index)")
    parser.add_argument("--show", action="store_true", help="Show a single sample window")
    parser.add_argument("--save_dir", type=str, default=None, help="If set, save images here")
    parser.add_argument("--no_gt", action="store_true", help="Hide ground-truth points")
    parser.add_argument("--browse", action="store_true", help="Enable keyboard browsing mode")

    args = parser.parse_args()

    dataset = RLNFDataset(dataset_root_path=args.dataset_root, split=args.split)

    if args.num is not None and args.save_dir is None:
        raise ValueError("--num requires --save_dir to be set.")

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        start_idx = args.idx
        end_idx = min(len(dataset), start_idx + args.num if args.num is not None else start_idx + 1)
        for i in range(start_idx, end_idx):
            sample = dataset[i]
            fig = render_sample(sample, i, show_gt=not args.no_gt)
            out_path = os.path.join(args.save_dir, f"sample_{i}.png")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
        print(f"Saved {end_idx - start_idx} samples to {args.save_dir}")
        return

    if args.browse:
        DatasetViewer(dataset, start_idx=args.idx, show_gt=not args.no_gt)
        return

    if args.show:
        sample = dataset[args.idx]
        fig = render_sample(sample, args.idx, show_gt=not args.no_gt)
        plt.show()
        plt.close(fig)
        return

    sample = dataset[args.idx]
    fig = render_sample(sample, args.idx, show_gt=not args.no_gt)
    out_path = f"sample_{args.idx}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
