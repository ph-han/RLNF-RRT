from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rlnf_rrt.data.dataset import RLNFDataset
from rlnf_rrt.models.flow import Flow
from rlnf_rrt.utils.config import load_toml, resolve_project_path
from rlnf_rrt.utils.utils import get_device


def _build_model_from_ckpt(ckpt: dict, device: torch.device) -> Flow:
    cfg = ckpt.get("config")
    if cfg is None or "model" not in cfg:
        raise RuntimeError("Checkpoint missing model config. Re-train with current train.py.")

    m = cfg["model"]
    model = Flow(
        num_blocks=int(m["num_blocks"]),
        latent_dim=int(m["latent_dim"]),
        hidden_dim=int(m["hidden_dim"]),
        s_max=float(m["s_max"]),
        channels=tuple(int(x) for x in m["channels"]),
        norm=str(m["norm"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot forward flow transformation steps")
    parser.add_argument("--eval-config", type=str, default="configs/eval/default.toml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=4)
    parser.add_argument("--save", type=str, default="outputs/figures/forward_flow_example.png")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--show", action="store_true")
    return parser


def _stats_text(points: np.ndarray) -> str:
    mu = points.mean(axis=0)
    var = points.var(axis=0)
    corr = np.corrcoef(points[:, 0], points[:, 1])[0, 1] if len(points) > 1 else 0.0
    r = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    return (
        f"mu=({mu[0]:.2f},{mu[1]:.2f})\n"
        f"var=({var[0]:.2f},{var[1]:.2f})\n"
        f"corr={corr:.2f}\n"
        f"r(mean/std)=({r.mean():.2f}/{r.std():.2f})"
    )


def main() -> None:
    args = _make_parser().parse_args()

    cfg = load_toml(args.eval_config)
    data_cfg = cfg["data"]
    eval_cfg = cfg["eval"]

    ckpt_path = resolve_project_path(args.checkpoint or eval_cfg["checkpoint"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    split = args.split or str(data_cfg.get("split", "test"))
    data_root = resolve_project_path(data_cfg.get("data_root", "data"))

    ds = RLNFDataset(
        split=split,
        data_root=data_root,
        noise_std=0.0,
        num_points=int(data_cfg["num_points"]),
        clearance=int(data_cfg["clearance"]),
        step_size=int(data_cfg["step_size"]),
    )
    if len(ds) == 0:
        raise RuntimeError("Dataset is empty with current config filters.")

    idx = max(0, min(args.index, len(ds) - 1))
    sample = ds[idx]

    device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device)
    model = _build_model_from_ckpt(ckpt, device)

    map_img = sample["map"].unsqueeze(0).to(device)
    start = sample["start"].unsqueeze(0).to(device)
    goal = sample["goal"].unsqueeze(0).to(device)
    x = sample["gt_path"].unsqueeze(0).to(device)

    with torch.no_grad():
        cond = model.cond_encoder(map_img, start, goal)
        states: list[np.ndarray] = [x.squeeze(0).cpu().numpy()]
        z = x
        for block in model.flows:
            z, _ = block(z, cond)
            states.append(z.squeeze(0).cpu().numpy())

    num_steps = len(states)
    ncols = min(5, num_steps)
    nrows = math.ceil(num_steps / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.2 * ncols, 4.6 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    latent_all = np.concatenate(states[1:], axis=0)
    lim = max(2.5, float(np.quantile(np.abs(latent_all), 0.995)) * 1.15)

    map_np = sample["map"].squeeze(0).numpy()
    gt_np = sample["gt_path"].numpy()
    start_np = sample["start"].numpy()
    goal_np = sample["goal"].numpy()

    cmap = ListedColormap(["#3a3337", "#dcdcdc"])

    for step, ax in enumerate(axes):
        if step >= num_steps:
            ax.axis("off")
            continue

        if step == 0:
            ax.imshow(
                map_np,
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                origin="lower",
                extent=(0, 1, 0, 1),
                interpolation="nearest",
            )
            ax.scatter(gt_np[:, 0], gt_np[:, 1], s=14, c="#149b23", alpha=0.65, label="GT Path (x)")
            ax.scatter([start_np[0]], [start_np[1]], s=180, c="red", label="Start", zorder=3)
            ax.scatter([goal_np[0]], [goal_np[1]], s=180, c="lime", label="Goal", zorder=3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title("Step 0: Data Space (x)", fontsize=12, fontweight="bold")
            ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
            continue

        pts = states[step]
        is_last = step == (num_steps - 1)

        theta = np.linspace(0.0, 2.0 * np.pi, 256)
        ax.plot(np.cos(theta), np.sin(theta), "r--", alpha=0.5, linewidth=1.4)
        ax.plot(2.0 * np.cos(theta), 2.0 * np.sin(theta), "r:", alpha=0.35, linewidth=1.3)
        ax.axhline(0.0, color="gray", alpha=0.25, linewidth=1.0)
        ax.axvline(0.0, color="gray", alpha=0.25, linewidth=1.0)

        if is_last:
            ax.scatter(pts[:, 0], pts[:, 1], s=14, c="purple", alpha=0.65, label="Latent (z)")
            ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
            ax.text(
                0.03,
                0.03,
                _stats_text(pts),
                transform=ax.transAxes,
                fontsize=9,
                va="bottom",
                ha="left",
                bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none"},
            )
            ax.set_title(f"Step {step}: Latent (z) [Block {step}]", fontsize=12, fontweight="bold")
        else:
            ax.scatter(pts[:, 0], pts[:, 1], s=14, c="#1c2dff", alpha=0.72)
            ax.set_title(f"Step {step}: After Block {step}", fontsize=12, fontweight="bold")

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(
        f"Forward Flow Transformation (Example {idx})\nData (x) -> Noise (z)",
        fontsize=20,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    save_path = resolve_project_path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")

    print(f"Saved figure: {save_path}")
    print(f"checkpoint: {ckpt_path}")
    print(f"split/index: {split}/{idx}")
    print(f"num_blocks: {len(model.flows)}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
