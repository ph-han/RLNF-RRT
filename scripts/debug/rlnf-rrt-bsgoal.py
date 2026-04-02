import sys
from pathlib import Path
import argparse

import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rlnf_rrt.data.dataset import RLNFDataset
from rlnf_rrt.models.flow import Flow
from rlnf_rrt.models.subgoal_policy import SubGoalPolicy
from rlnf_rrt.utils.config import load_toml, resolve_project_path
from rlnf_rrt.utils.utils import get_device


def _build_model_from_ckpt(ckpt: dict, device: torch.device, backbone: str = "resnet34", is_pe: bool = False) -> Flow:
    cfg = ckpt.get("config")
    if cfg is None or "model" not in cfg:
        raise RuntimeError("Checkpoint missing model config. Re-train with current train.py.")

    m = cfg["model"]
    model = Flow(
        num_blocks=int(m["num_blocks"]),
        latent_dim=int(m["latent_dim"]),
        hidden_dim=int(m["hidden_dim"]),
        s_max=float(m["s_max"]),
        backbone=backbone,
        is_pe=is_pe
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Binary-search hierarchical sub-goal path visualization")
    parser.add_argument("--eval-config", type=str, default="configs/eval/default.toml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=4)
    parser.add_argument("--all", action="store_true", help="Run visualization for all samples in the split")
    parser.add_argument("--depth", type=int, default=2, help="Binary search depth (depth=2 → 3 sub-goals, 4 segments)")
    parser.add_argument("--save", type=str, default="outputs/figures/bsgoal-example.png")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--rl-checkpoint", type=str,
        default="outputs/checkpoints/rl_20260315_152232/best.pt",
        help="RL policy checkpoint path for sub-goal prediction",
    )
    return parser


def _beta_grid_argmax(
    alpha_x: torch.Tensor, beta_x: torch.Tensor,
    alpha_y: torch.Tensor, beta_y: torch.Tensor,
    obstacle_mask: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Beta PDF 그리드 argmax — obstacle-free 픽셀 중 확률 최고점을 반환 (normalized [0,1])."""
    H, W = obstacle_mask.shape
    xs = torch.linspace(0.001, 0.999, W, device=device)
    ys = torch.linspace(0.001, 0.999, H, device=device)
    dist_x = torch.distributions.Beta(alpha_x.clamp(min=0.1), beta_x.clamp(min=0.1))
    dist_y = torch.distributions.Beta(alpha_y.clamp(min=0.1), beta_y.clamp(min=0.1))
    log_p = dist_y.log_prob(ys)[:, None] + dist_x.log_prob(xs)[None, :]  # (H, W)
    log_p[torch.tensor(obstacle_mask, device=device)] = float("-inf")
    flat_idx = log_p.argmax()
    yi, xi = flat_idx // W, flat_idx % W
    return torch.tensor([xi / (W - 1), yi / (H - 1)], device=device)


def _load_rl_policy(ckpt_path: str, device: torch.device) -> SubGoalPolicy:
    ckpt = torch.load(resolve_project_path(ckpt_path), map_location=device, weights_only=False)
    policy_cfg = ckpt["config"]["policy"]
    policy = SubGoalPolicy(
        latent_dim=int(policy_cfg.get("latent_dim", 128)),
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


def _to_pixel(pts: np.ndarray, H: int, W: int):
    """Normalized [0,1] coords → pixel (x, y)."""
    return pts[:, 0] * (W - 1), pts[:, 1] * (H - 1)


def _make_cond(cond_base: torch.Tensor, start_pt: np.ndarray, goal_pt: np.ndarray,
               H: int, W: int) -> torch.Tensor:
    """channel 1을 start(+1)/goal(-1)로 교체한 cond_image 반환. (1, 3, H, W)"""
    cond = cond_base.clone()
    cond[0, 1] = 0.0
    sy = int(round(float(start_pt[1]) * (H - 1)))
    sx = int(round(float(start_pt[0]) * (W - 1)))
    gy = int(round(float(goal_pt[1]) * (H - 1)))
    gx = int(round(float(goal_pt[0]) * (W - 1)))
    cond[0, 1, sy, sx] = 1.0
    cond[0, 1, gy, gx] = -1.0
    return cond


def _predict_subgoal(
    cond_base: torch.Tensor,
    start_np: np.ndarray,
    goal_np: np.ndarray,
    rl_policy: SubGoalPolicy,
    device: torch.device,
    obstacle_mask: np.ndarray,
    H: int,
    W: int,
) -> np.ndarray:
    """구간 [start_np, goal_np] 사이의 sub-goal 1개 예측. 반환: (2,) normalized coords."""
    seg_cond = _make_cond(cond_base, start_np, goal_np, H, W)
    start_t = torch.from_numpy(start_np).float().to(device)
    goal_t = torch.from_numpy(goal_np).float().to(device)
    with torch.no_grad():
        alpha, beta_param = rl_policy(seg_cond.squeeze(0), start_t, goal_t)
        sg = _beta_grid_argmax(alpha[0], beta_param[0], alpha[1], beta_param[1], obstacle_mask, device)
    return sg.cpu().numpy()


def _binary_subgoals(
    cond_base: torch.Tensor,
    start_np: np.ndarray,
    goal_np: np.ndarray,
    rl_policy: SubGoalPolicy,
    device: torch.device,
    obstacle_mask: np.ndarray,
    H: int,
    W: int,
    depth: int = 2,
) -> list:
    """Binary search 방식으로 sub-goal 트리 구성.
    depth=1 → [SG_mid]  (1개)
    depth=2 → [SG_left, SG_mid, SG_right]  (3개, 경로 순서)
    depth=N → 2^N - 1개
    """
    sg_mid = _predict_subgoal(cond_base, start_np, goal_np, rl_policy, device, obstacle_mask, H, W)
    if depth <= 1:
        return [sg_mid]
    left = _binary_subgoals(cond_base, start_np, sg_mid, rl_policy, device, obstacle_mask, H, W, depth - 1)
    right = _binary_subgoals(cond_base, sg_mid, goal_np, rl_policy, device, obstacle_mask, H, W, depth - 1)
    return left + [sg_mid] + right


def _gen_seg_paths(waypoints, cond_image, model, device, num_points, num_samples):
    """waypoints 리스트로 각 segment의 샘플 경로를 생성. list of (num_samples, pts_per_seg, 2)."""
    num_segments = len(waypoints) - 1
    pts_per_seg = max(2, num_points // num_segments)
    all_seg_paths = []
    for i in range(num_segments):
        s_t = torch.from_numpy(waypoints[i]).float().unsqueeze(0).to(device)
        g_t = torch.from_numpy(waypoints[i + 1]).float().unsqueeze(0).to(device)
        H, W = cond_image.shape[2], cond_image.shape[3]
        seg_cond = _make_cond(cond_image, waypoints[i], waypoints[i + 1], H, W)
        seg = []
        with torch.no_grad():
            for _ in range(num_samples):
                z = torch.randn((1, pts_per_seg, 2), device=device, dtype=cond_image.dtype)
                pred, _ = model.inverse(seg_cond, s_t, g_t, z)
                seg.append(pred.squeeze(0).clamp(0, 1).cpu().numpy())
        all_seg_paths.append(np.stack(seg))
    return all_seg_paths


def _draw_panel(ax, map_np, gt_path, H, W, start_np, goal_np, title,
                full_paths=None, seg_paths=None, seg_colors=None, subgoals=None, num_samples=4):
    ax.imshow(map_np, cmap="gray", origin="upper", vmin=0, vmax=1)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

    px, py = _to_pixel(gt_path, H, W)
    ax.plot(px, py, "b-", lw=1.5, alpha=0.9, label="GT", zorder=4)

    if full_paths is not None:
        all_pts = full_paths.reshape(-1, 2)
        px, py = _to_pixel(all_pts, H, W)
        ax.scatter(px, py, s=4, color="#9b59b6", alpha=0.25, zorder=2,
                   linewidths=0, label=f"Pred ×{num_samples}")

    if seg_paths is not None:
        for seg_i, (sp, color) in enumerate(zip(seg_paths, seg_colors)):
            all_pts = sp.reshape(-1, 2)
            px, py = _to_pixel(all_pts, H, W)
            ax.scatter(px, py, s=4, color=color, alpha=0.3, zorder=2,
                       linewidths=0, label=f"Seg{seg_i+1}")

    if subgoals is not None:
        mid_idx = len(subgoals) // 2
        for sg_i, sg in enumerate(subgoals):
            sx, sy = sg[0] * (W - 1), sg[1] * (H - 1)
            is_mid = (sg_i == mid_idx)
            marker = "D" if is_mid else "^"
            ms = 10 if is_mid else 8
            color = "#f39c12" if is_mid else "#e8d44d"
            ax.plot(sx, sy, marker, color=color, ms=ms, zorder=5, label=f"SG{sg_i+1}")
            ax.annotate(f"SG{sg_i+1}", (sx, sy), xytext=(5, 4),
                        textcoords="offset points", fontsize=8,
                        color=color, fontweight="bold")

    sx, sy = start_np[0] * (W - 1), start_np[1] * (H - 1)
    gx, gy = goal_np[0] * (W - 1), goal_np[1] * (H - 1)
    ax.plot(sx, sy, "^", color="#2ecc71", ms=10, zorder=6, label="Start")
    ax.plot(gx, gy, "*", color="#e74c3c", ms=13, zorder=6, label="Goal")
    ax.legend(loc="best", fontsize=7, framealpha=0.85)


def _visualize_one(idx: int, sample, model, rl_policy, device, args, save_path: Path) -> None:
    cond_image = sample["cond_image"].unsqueeze(0).to(device)  # (1, 3, H, W)
    start = sample["start"].unsqueeze(0).to(device)
    goal = sample["goal"].unsqueeze(0).to(device)
    gt_path = sample["gt_path"].cpu().numpy()

    map_np = sample["cond_image"][0].cpu().numpy()
    H, W = map_np.shape
    start_np = sample["start"].cpu().numpy()
    goal_np = sample["goal"].cpu().numpy()
    obstacle_mask = (sample["cond_image"][0].cpu().numpy() <= 0.5)

    num_points = 128
    num_samples = 4
    base_colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db", "#9b59b6", "#1abc9c", "#e67e22"]

    # --- Sub-goal prediction ---
    # Panel 2: single sub-goal (depth=1)
    sg1_list = _binary_subgoals(cond_image, start_np, goal_np, rl_policy, device, obstacle_mask, H, W, depth=1)
    # Panel 3: binary sub-goals (depth=args.depth, default 2)
    sg_bin_list = _binary_subgoals(cond_image, start_np, goal_np, rl_policy, device, obstacle_mask, H, W, depth=args.depth)

    print(f"[{idx}] single SG: {sg1_list[0]}")
    print(f"[{idx}] binary SGs (depth={args.depth}): {[sg.tolist() for sg in sg_bin_list]}")

    # --- Warm-up ---
    for _ in range(20):
        z = torch.randn((1, num_points, 2), device=device, dtype=cond_image.dtype)
        _ = model.inverse(cond_image, start, goal, z)

    # --- Panel 1: direct paths ---
    full_paths = []
    with torch.no_grad():
        for _ in range(num_samples):
            z = torch.randn((1, num_points, 2), device=device, dtype=cond_image.dtype)
            pred, _ = model.inverse(cond_image, start, goal, z)
            full_paths.append(pred.squeeze(0).clamp(0, 1).cpu().numpy())
    full_paths = np.stack(full_paths)

    # --- Panel 2: single sub-goal segments ---
    wp1 = [start_np] + sg1_list + [goal_np]
    seg_paths_1 = _gen_seg_paths(wp1, cond_image, model, device, num_points, num_samples)
    seg_colors_1 = [base_colors[i % len(base_colors)] for i in range(len(seg_paths_1))]

    # --- Panel 3: binary sub-goal segments ---
    wp_bin = [start_np] + sg_bin_list + [goal_np]
    seg_paths_bin = _gen_seg_paths(wp_bin, cond_image, model, device, num_points, num_samples)
    seg_colors_bin = [base_colors[i % len(base_colors)] for i in range(len(seg_paths_bin))]

    # --- Visualization: 3 panels ---
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    _draw_panel(axes[0], map_np, gt_path, H, W, start_np, goal_np,
                title="Direct: Start → Goal",
                full_paths=full_paths, num_samples=num_samples)

    _draw_panel(axes[1], map_np, gt_path, H, W, start_np, goal_np,
                title="Single Sub-goal (depth=1)",
                seg_paths=seg_paths_1, seg_colors=seg_colors_1,
                subgoals=sg1_list, num_samples=num_samples)

    _draw_panel(axes[2], map_np, gt_path, H, W, start_np, goal_np,
                title=f"Binary Sub-goal (depth={args.depth}, {len(sg_bin_list)} SGs)",
                seg_paths=seg_paths_bin, seg_colors=seg_colors_bin,
                subgoals=sg_bin_list, num_samples=num_samples)

    plt.suptitle(
        f"Flow Model – Sub-goal Comparison  (sample {idx})",
        fontsize=13, y=1.01
    )
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.close(fig)


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

    ds = RLNFDataset(data_root=data_root, split=split)

    device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone = eval_cfg.get("backbone", "resnet34")
    is_pe = eval_cfg.get("is_pe", False)
    model = _build_model_from_ckpt(ckpt, device, backbone=backbone, is_pe=is_pe)
    rl_policy = _load_rl_policy(args.rl_checkpoint, device)

    if args.all:
        out_dir = Path(args.save)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving {len(ds)} figures to {out_dir}/")
        for i in range(len(ds)):
            print(f"[{i+1}/{len(ds)}] index {i}")
            save_path = out_dir / f"{i:04d}.png"
            try:
                _visualize_one(i, ds[i], model, rl_policy, device, args, save_path)
            except Exception as e:
                print(f"  ERROR at index {i}: {e}")
    else:
        idx = max(0, min(args.index, len(ds) - 1))
        save_path = Path(args.save)
        _visualize_one(idx, ds[idx], model, rl_policy, device, args, save_path)
        if args.show:
            plt.show()


if __name__ == "__main__":
    main()
