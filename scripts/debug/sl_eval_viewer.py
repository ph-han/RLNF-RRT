from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")  # 서버 환경: 디스플레이 없이 파일 저장
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rlnf_rrt.data.dataset import RLNFDataset
from rlnf_rrt.models.subgoal_policy import SubGoalPolicy
from rlnf_rrt.utils.config import load_toml, resolve_project_path
from rlnf_rrt.utils.utils import get_device


def _find_checkpoint(cfg: dict) -> Path:
    output_cfg = cfg["output"]
    run_name = str(output_cfg.get("run_name", "sl_midpoint")).strip()
    ckpt_root = resolve_project_path(output_cfg.get("checkpoint_root", "outputs/checkpoints"))
    run_dir = ckpt_root / run_name

    for name in ("best.pt", "last.pt"):
        p = run_dir / name
        if p.exists():
            return p

    raise FileNotFoundError(
        f"No checkpoint found in {run_dir}. "
        "Run training first or specify --checkpoint explicitly."
    )


def _load_model(cfg: dict, checkpoint: Path, device: torch.device) -> SubGoalPolicy:
    policy_cfg = cfg["policy"]
    model = SubGoalPolicy(
        latent_dim=int(policy_cfg.get("latent_dim", 64)),
        hidden_dim=int(policy_cfg.get("hidden_dim", 128)),
        backbone=str(policy_cfg.get("backbone", "resnet34")),
        num_subgoals=int(policy_cfg.get("num_subgoals", 1)),
    ).to(device)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    val_nll = ckpt.get("val_nll", float("nan"))
    print(f"Loaded: {checkpoint}  (epoch={epoch}, val_nll={val_nll:.4f})")
    return model


@torch.no_grad()
def _predict(
    model: SubGoalPolicy,
    sample: dict,
    device: torch.device,
) -> np.ndarray:
    cond_image = sample["cond_image"].unsqueeze(0).to(device)
    start = sample["start"].unsqueeze(0).to(device)
    goal = sample["goal"].unsqueeze(0).to(device)

    alpha, beta_param = model(cond_image, start, goal)
    pred = alpha / (alpha + beta_param)  # Beta distribution mean
    return pred.squeeze(0).cpu().numpy()


def _draw_page(
    fig: Figure,
    axes_arr: np.ndarray,
    ds: RLNFDataset,
    model: SubGoalPolicy,
    device: torch.device,
    base: int,
    mid_idx: int,
    ckpt_name: str,
    split: str,
    cmap,
) -> None:
    fig.suptitle(
        f"SL Midpoint Eval — {ckpt_name}  |  split={split}  "
        f"[{base}~{min(base+3, len(ds)-1)}]",
        fontsize=13,
    )

    for ax_i, ax in enumerate(axes_arr):
        ax.clear()
        ds_idx = base + ax_i
        if ds_idx >= len(ds):
            ax.axis("off")
            continue

        sample = ds[ds_idx]
        map_np = sample["cond_image"][0].numpy()
        start_np = sample["start"].numpy()
        goal_np = sample["goal"].numpy()
        gt_path_np = sample["gt_path"].numpy()
        gt_mid = gt_path_np[mid_idx]

        pred = _predict(model, sample, device)
        l2 = float(np.linalg.norm(pred - gt_mid))

        ax.imshow(
            map_np, cmap=cmap, vmin=0.0, vmax=1.0,
            origin="lower", extent=(0, 1, 0, 1), interpolation="nearest",
        )
        ax.scatter(gt_path_np[:, 0], gt_path_np[:, 1],
                   s=6, c="#18a71e", alpha=0.4, edgecolors="none", label="GT path")
        ax.scatter([gt_mid[0]], [gt_mid[1]],
                   s=180, c="#00ff88", marker="*", edgecolors="black", linewidths=0.5,
                   zorder=4, label=f"GT mid ({mid_idx})")
        ax.scatter([pred[0]], [pred[1]],
                   s=160, c="#ff3333", marker="X", edgecolors="black", linewidths=0.5,
                   zorder=5, label="Pred")
        ax.plot([gt_mid[0], pred[0]], [gt_mid[1], pred[1]],
                color="#ffaa00", linewidth=1.2, linestyle="--", alpha=0.8, zorder=3)
        ax.scatter([start_np[0]], [start_np[1]], s=200, c="red",
                   edgecolors="none", zorder=6, label="Start")
        ax.scatter([goal_np[0]], [goal_np[1]], s=200, c="lime",
                   edgecolors="none", zorder=6, label="Goal")

        ax.set_title(f"#{ds_idx}  L2={l2:.4f}", fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=9)

        if ax_i == 0:
            ax.legend(loc="upper right", frameon=True, framealpha=0.85, fontsize=8)

    fig.tight_layout()


def main() -> None:
    parser = argparse.ArgumentParser(description="SL midpoint prediction viewer (server mode: saves images)")
    parser.add_argument("--config", type=str, default="configs/sl/default.toml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--index", type=int, default=0, help="시작 인덱스")
    parser.add_argument("--num", type=int, default=4, help="저장할 샘플 수 (4의 배수 권장)")
    parser.add_argument("--out-dir", type=str, default=None, help="저장 디렉토리 (기본: outputs/figures)")
    args = parser.parse_args()

    cfg = load_toml(args.config)
    data_cfg = cfg["data"]
    device = get_device()

    ckpt_path = resolve_project_path(args.checkpoint) if args.checkpoint else _find_checkpoint(cfg)
    model = _load_model(cfg, ckpt_path, device)

    data_root = resolve_project_path(data_cfg.get("data_root", "data"))
    ds = RLNFDataset(
        split=args.split,
        data_root=data_root,
        num_points=int(data_cfg["num_points"]),
        clearance=int(data_cfg["clearance"]),
        step_size=int(data_cfg["step_size"]),
    )
    if len(ds) == 0:
        print("Dataset is empty.")
        return

    num_points = int(data_cfg["num_points"])
    mid_idx = num_points // 2

    save_dir = resolve_project_path(args.out_dir) if args.out_dir else resolve_project_path("outputs/figures")
    save_dir.mkdir(parents=True, exist_ok=True)

    cmap = ListedColormap(["#3b3336", "#dfdfdf"])

    # 4개씩 묶어서 페이지 단위로 저장
    num_pages = max(1, (args.num + 3) // 4)
    start_idx = max(0, min(args.index, len(ds) - 1))

    print(f"dataset={args.split}  size={len(ds)}  mid_index={mid_idx}")
    print(f"saving {num_pages} page(s) starting from index {start_idx} → {save_dir}")

    for page in range(num_pages):
        base = start_idx + page * 4
        if base >= len(ds):
            break

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        axes_arr = np.atleast_1d(axes).reshape(-1)

        _draw_page(
            fig=fig,
            axes_arr=axes_arr,
            ds=ds,
            model=model,
            device=device,
            base=base,
            mid_idx=mid_idx,
            ckpt_name=ckpt_path.name,
            split=args.split,
            cmap=cmap,
        )

        end_i = min(len(ds), base + 4)
        fname = f"sl_eval_{ckpt_path.stem}_{args.split}_{base:05d}-{end_i:05d}.png"
        out = save_dir / fname
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")

    print("done.")


if __name__ == "__main__":
    main()
