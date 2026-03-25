"""
Beta PDF 그리드 시각화 스크립트.

RL policy가 예측한 Beta 분포를 (H×W) 그리드에서 계산하고,
장애물 마스킹 전후 heatmap을 나란히 보여줍니다.

Usage:
  python scripts/debug/beta_pdf_vis.py \
    --rl-checkpoint outputs/checkpoints/reinforce_resample/best.pt \
    --index 4 --show
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rlnf_rrt.data.dataset import RLNFDataset
from rlnf_rrt.models.subgoal_policy import SubGoalPolicy
from rlnf_rrt.utils.config import load_toml, resolve_project_path
from rlnf_rrt.utils.utils import get_device


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualize Beta PDF grid from RL policy")
    p.add_argument("--eval-config",    type=str, default="configs/eval/default.toml")
    p.add_argument("--split",          type=str, default=None, choices=["train", "val", "test"])
    p.add_argument("--index",          type=int, default=4)
    p.add_argument("--rl-checkpoint",  type=str,
                   default="outputs/checkpoints/rl_20260315_152232/best.pt")
    p.add_argument("--save",           type=str, default="outputs/figures/beta_pdf_vis.png")
    p.add_argument("--dpi",            type=int, default=180)
    p.add_argument("--show",           action="store_true")
    return p


def _load_rl_policy(ckpt_path: str, device: torch.device) -> SubGoalPolicy:
    ckpt = torch.load(resolve_project_path(ckpt_path), map_location=device, weights_only=False)
    policy_cfg = ckpt["config"]["policy"]
    policy = SubGoalPolicy(
        latent_dim=int(policy_cfg.get("latent_dim", 64)),
        hidden_dim=int(policy_cfg.get("hidden_dim", 128)),
        backbone=str(policy_cfg.get("backbone", "resnet34")),
        num_subgoals=1,
    ).to(device)
    raw_sd = ckpt.get("actor_state_dict") or ckpt.get("model_state_dict")
    prefix = "module.0.module."
    stripped = {k[len(prefix):]: v for k, v in raw_sd.items() if k.startswith(prefix)}
    sg_sd = stripped if stripped else raw_sd
    policy.load_state_dict(sg_sd)
    policy.eval()
    return policy


def main() -> None:
    args = _make_parser().parse_args()

    cfg      = load_toml(args.eval_config)
    data_cfg = cfg["data"]
    split    = args.split or str(data_cfg.get("split", "test"))
    data_root = resolve_project_path(data_cfg.get("data_root", "data"))

    ds     = RLNFDataset(data_root=data_root, split=split)
    idx    = max(0, min(args.index, len(ds) - 1))
    sample = ds[idx]

    device = get_device()

    cond_image = sample["cond_image"].unsqueeze(0).to(device)  # (1, 3, H, W)
    start      = sample["start"].unsqueeze(0).to(device)       # (1, 2)
    goal       = sample["goal"].unsqueeze(0).to(device)        # (1, 2)

    map_np        = sample["cond_image"][0].cpu().numpy()      # (H, W) binary map
    obstacle_mask = map_np <= 0.5                              # True = obstacle
    H, W          = map_np.shape

    start_np = sample["start"].cpu().numpy()   # (2,)
    goal_np  = sample["goal"].cpu().numpy()    # (2,)

    # ── Run RL policy ──────────────────────────────────────────────────
    rl_policy = _load_rl_policy(args.rl_checkpoint, device)
    with torch.no_grad():
        alpha, beta_param = rl_policy(
            cond_image.squeeze(0), start.squeeze(0), goal.squeeze(0)
        )  # (2,), (2,)

    print(f"alpha = {alpha.cpu().numpy()},  beta = {beta_param.cpu().numpy()}")

    # ── Compute Beta PDF grid on GPU ───────────────────────────────────
    xs = torch.linspace(0.001, 0.999, W, device=device)   # (W,)
    ys = torch.linspace(0.001, 0.999, H, device=device)   # (H,)
    dist_x = torch.distributions.Beta(alpha[0].clamp(min=0.1), beta_param[0].clamp(min=0.1))
    dist_y = torch.distributions.Beta(alpha[1].clamp(min=0.1), beta_param[1].clamp(min=0.1))
    log_p = dist_y.log_prob(ys)[:, None] + dist_x.log_prob(xs)[None, :]  # (H, W) on GPU

    # Raw heatmap (no masking)
    p_raw = (log_p - log_p.max()).exp().cpu().numpy()  # (H, W), max=1

    # Masked heatmap
    log_p_masked = log_p.clone()
    log_p_masked[torch.tensor(obstacle_mask, device=device)] = float("-inf")
    finite_max = log_p_masked[log_p_masked.isfinite()].max()
    p_masked = (log_p_masked - finite_max).exp().cpu().numpy()
    p_masked[obstacle_mask] = 0.0

    # Selected sub-goal (argmax of masked grid)
    flat_idx = log_p_masked.argmax()
    yi_sel = (flat_idx // W).item()
    xi_sel = (flat_idx % W).item()
    sg_norm = (xi_sel / (W - 1), yi_sel / (H - 1))   # normalized
    sg_px   = (xi_sel, yi_sel)                         # pixel (x, y)

    print(f"Selected sub-goal (normalized): {sg_norm}")
    print(f"Selected sub-goal (pixel):      {sg_px}")

    # ── Plot ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Beta PDF Grid  |  index={idx}  |  α={alpha.cpu().numpy().round(3)},  β={beta_param.cpu().numpy().round(3)}",
                 fontsize=10)

    cmap = "hot"

    def _marker(ax, pt_norm, color, marker, label, zorder=5):
        px = pt_norm[0] * (W - 1)
        py = pt_norm[1] * (H - 1)
        ax.plot(px, py, marker=marker, color=color, markersize=9,
                markeredgecolor="white", markeredgewidth=0.8,
                label=label, zorder=zorder)

    for ax, p_data, title in [
        (axes[0], p_raw,    "Raw Beta PDF (장애물 무시)"),
        (axes[1], p_masked, "Masked Beta PDF (장애물 = 0)"),
    ]:
        ax.imshow(map_np, cmap="gray", vmin=0, vmax=1, origin="upper")
        im = ax.imshow(p_data, cmap=cmap, alpha=0.65, origin="upper",
                       norm=mcolors.PowerNorm(gamma=0.4, vmin=0, vmax=1))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="relative prob")
        _marker(ax, start_norm := start_np, "lime",    "^", "start")
        _marker(ax, goal_np,                "#e74c3c", "v", "goal")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        ax.legend(loc="lower right", fontsize=7, framealpha=0.7)

    # Add selected sub-goal star on right panel
    axes[1].plot(xi_sel, yi_sel, marker="*", color="cyan", markersize=14,
                 markeredgecolor="black", markeredgewidth=0.6,
                 label="selected ★", zorder=6)
    axes[1].legend(loc="lower right", fontsize=7, framealpha=0.7)

    plt.tight_layout()

    save_path = Path(resolve_project_path(args.save))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved → {save_path}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
