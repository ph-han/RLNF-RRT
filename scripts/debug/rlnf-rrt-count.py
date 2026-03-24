"""
Debug/visualization script for AutoregressiveSubGoalPolicyCount (Count-first AR).
Usage:
  python scripts/debug/rlnf-rrt-count.py \
    --rl-checkpoint outputs/checkpoints/reinforce_ar/best.pt \
    --index 4 --show
"""
import sys
from pathlib import Path
import argparse

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import Categorical, Beta

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rlnf_rrt.data.dataset import RLNFDataset
from rlnf_rrt.models.flow import Flow
from rlnf_rrt.models.subgoal_policy import AutoregressiveSubGoalPolicyCount
from rlnf_rrt.utils.config import load_toml, resolve_project_path
from rlnf_rrt.utils.utils import get_device


def _build_flow_from_ckpt(ckpt: dict, device: torch.device, backbone: str = "resnet34", is_pe: bool = False) -> Flow:
    cfg = ckpt.get("config")
    if cfg is None or "model" not in cfg:
        raise RuntimeError("Checkpoint missing model config.")
    m = cfg["model"]
    model = Flow(
        num_blocks=int(m["num_blocks"]),
        latent_dim=int(m["latent_dim"]),
        hidden_dim=int(m["hidden_dim"]),
        s_max=float(m["s_max"]),
        backbone=backbone,
        is_pe=is_pe,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


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


def _load_rl_policy(ckpt_path: str, device: torch.device) -> tuple[AutoregressiveSubGoalPolicyCount, int]:
    """Returns (policy, max_subgoals)."""
    ckpt = torch.load(resolve_project_path(ckpt_path), map_location=device, weights_only=False)
    policy_cfg = ckpt["config"]["policy"]
    max_subgoals = int(policy_cfg.get("max_subgoals", 4))
    policy = AutoregressiveSubGoalPolicyCount(
        max_subgoals=max_subgoals,
        latent_dim=int(policy_cfg.get("latent_dim", 128)),
        hidden_dim=int(policy_cfg.get("hidden_dim", 128)),
        backbone=str(policy_cfg.get("backbone", "resnet34")),
    ).to(device)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()
    return policy, max_subgoals


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Variable sub-goal count policy visualization")
    parser.add_argument("--eval-config", type=str, default="configs/eval/default.toml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=4)
    parser.add_argument("--save", type=str, default="outputs/figures/sub-goal-ar-example.png")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--rrt", action="store_true")
    parser.add_argument("--iter-num", type=int, default=3000)
    parser.add_argument("--expand-size", type=float, default=2.0)
    parser.add_argument("--near-distance", type=float, default=15.0)
    parser.add_argument("--clearance", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--goal-bias", type=float, default=0.1)
    parser.add_argument("--flow-guidance-prob", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rl-checkpoint", type=str,
        default="outputs/checkpoints/reinforce_ar/best.pt",
    )
    return parser


def _to_pixel(pts: np.ndarray, H: int, W: int):
    return pts[:, 0] * (W - 1), pts[:, 1] * (H - 1)


def _make_cond(cond_base: torch.Tensor, start_pt: np.ndarray, goal_pt: np.ndarray,
               H: int, W: int) -> torch.Tensor:
    cond = cond_base.clone()
    cond[0, 1] = 0.0
    sy = int(round(float(start_pt[1]) * (H - 1)))
    sx = int(round(float(start_pt[0]) * (W - 1)))
    gy = int(round(float(goal_pt[1]) * (H - 1)))
    gx = int(round(float(goal_pt[0]) * (W - 1)))
    cond[0, 1, sy, sx] = 1.0
    cond[0, 1, gy, gx] = -1.0
    return cond


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

    idx = max(0, min(args.index, len(ds) - 1))
    sample = ds[idx]

    device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone = eval_cfg.get("backbone", "resnet34")
    is_pe = eval_cfg.get("is_pe", False)
    model = _build_flow_from_ckpt(ckpt, device, backbone=backbone, is_pe=is_pe)

    cond_image = sample["cond_image"].unsqueeze(0).to(device)
    start = sample["start"].unsqueeze(0).to(device)
    goal  = sample["goal"].unsqueeze(0).to(device)
    gt_path = sample["gt_path"].cpu().numpy()

    map_np = sample["cond_image"][0].cpu().numpy()
    H, W = map_np.shape

    start_np = sample["start"].cpu().numpy()
    goal_np  = sample["goal"].cpu().numpy()

    # ── Sub-goal prediction (AR policy) ───────────────────────────────
    rl_policy, max_subgoals = _load_rl_policy(args.rl_checkpoint, device)
    obstacle_mask = (sample["cond_image"][0].cpu().numpy() <= 0.5)

    with torch.no_grad():
        img_s  = cond_image.squeeze(0)   # (3, H, W)
        st_s   = start.squeeze(0)        # (2,)
        goal_s = goal.squeeze(0)         # (2,)

        # 1. map encode (한 번만)
        map_feat = rl_policy.encode_map(img_s.unsqueeze(0))  # (1, latent)

        # 2. count head → argmax for deterministic k
        feat0 = torch.cat([map_feat, st_s.unsqueeze(0), goal_s.unsqueeze(0)], dim=-1)
        h0 = rl_policy.mlp(feat0)
        count_logits = rl_policy.count_head(h0).squeeze(0)   # (max_K+1,)
        k = int(count_logits.argmax().item())

        # 3. AR: forward_step 순서대로 — 이전 SG를 prev로 전달
        sgs = []
        prev = st_s.unsqueeze(0)  # (1, 2)
        for _ in range(k):
            alpha_i, beta_i = rl_policy.forward_step(
                map_feat, prev, goal_s.unsqueeze(0)
            )  # (1, 2) each
            sg = _beta_grid_argmax(
                alpha_i[0, 0], beta_i[0, 0],  # x
                alpha_i[0, 1], beta_i[0, 1],  # y
                obstacle_mask, device,
            )
            sgs.append(sg)
            prev = sg.unsqueeze(0)

        sub_goal_2d = torch.stack(sgs) if k > 0 else torch.zeros(0, 2, device=device)

    sub_goals_np = sub_goal_2d.cpu().numpy()  # (k, 2)
    print(f"RL policy predicted k={k} sub-goals")
    for i, sg in enumerate(sub_goals_np):
        print(f"  SG{i+1}: {sg}")

    # ── Waypoints & segmented paths ───────────────────────────────────
    waypoints = [start_np] + list(sub_goals_np) + [goal_np]
    num_segments = len(waypoints) - 1

    num_points   = gt_path.shape[0]
    pts_per_seg  = max(2, num_points // max(num_segments, 1))
    num_samples  = 6

    # Generate segment paths
    # Colors: cycle through a palette for each segment
    seg_palette = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]
    seg_colors = [seg_palette[i % len(seg_palette)] for i in range(num_segments)]

    import time as _time
    all_seg_paths = []
    gpu_time_seg = 0.0
    for i in range(num_segments):
        s_t = torch.from_numpy(waypoints[i]).float().unsqueeze(0).to(device)
        g_t = torch.from_numpy(waypoints[i + 1]).float().unsqueeze(0).to(device)
        seg_cond = _make_cond(cond_image, waypoints[i], waypoints[i + 1], H, W)
        seg = []
        _t0 = _time.perf_counter()
        with torch.no_grad():
            for _ in range(num_samples):
                z = torch.randn((1, pts_per_seg, 2), device=device, dtype=cond_image.dtype)
                pred, _ = model.inverse(seg_cond, s_t, g_t, z)
                seg.append(pred.squeeze(0).clamp(0, 1).cpu().numpy())
        gpu_time_seg += _time.perf_counter() - _t0
        all_seg_paths.append(np.stack(seg))

    # Full path (direct)
    full_paths = []
    _t0 = _time.perf_counter()
    with torch.no_grad():
        for _ in range(num_samples):
            z = torch.randn((1, num_points, 2), device=device, dtype=cond_image.dtype)
            pred, _ = model.inverse(cond_image, start, goal, z)
            full_paths.append(pred.squeeze(0).clamp(0, 1).cpu().numpy())
    gpu_time_direct = _time.perf_counter() - _t0
    full_paths = np.stack(full_paths)

    # ── RRT* (optional) ───────────────────────────────────────────────
    rrt_direct_node = None
    rrt_direct_metrics = None
    rrt_subgoal_nodes = []
    rrt_subgoal_metrics = None

    if args.rrt:
        from rlnf_rrt.utils.nf_rrt import FlowRRTStar
        rrt_kwargs = dict(
            seed=args.seed, clearance=args.clearance, step_size=1,
            iter_num=args.iter_num, cspace_img_path=f"{split}:{idx}",
            is_neural_mode=True, near_distance=args.near_distance,
            expand_size=args.expand_size, threshold=args.threshold,
            num_guidance_samples=num_samples, goal_bias=args.goal_bias,
            flow_guidance_prob=args.flow_guidance_prob,
        )

        planner_direct = FlowRRTStar(**rrt_kwargs)
        planner_direct.set_problem_from_sample(sample)
        planner_direct.set_external_pred_paths(list(full_paths))
        rrt_direct_node, cpu_t, _, tot_t = planner_direct.planning(
            model=None, device=None, is_rewiring=True, is_break=True, is_draw=False
        )
        rrt_direct_metrics = dict(
            cpu=cpu_t, gpu=gpu_time_direct, total=cpu_t + gpu_time_direct,
            nodes=len(planner_direct.paths),
            success=planner_direct.is_goal,
            cost=rrt_direct_node.cost if rrt_direct_node else None,
        )
        status = "SUCCESS" if planner_direct.is_goal else "FAILED"
        cost_str = f"{rrt_direct_node.cost:.1f}" if rrt_direct_node else "N/A"
        print(f"[RRT* Direct] {status}  nodes={len(planner_direct.paths)}  cost={cost_str}  time={tot_t:.3f}s")

        sg_cpu, sg_gpu, sg_total, sg_nodes = 0.0, 0.0, 0.0, 0
        sg_success_all = True
        sg_cost_total = 0.0
        for seg_i in range(num_segments):
            seg_preds = list(all_seg_paths[seg_i])
            s_wp = waypoints[seg_i]
            g_wp = waypoints[seg_i + 1]
            seg_sample = {
                "cond_image": sample["cond_image"],
                "start": torch.from_numpy(s_wp).float(),
                "goal":  torch.from_numpy(g_wp).float(),
                "gt_path": torch.from_numpy(seg_preds[0]).float(),
            }
            planner_seg = FlowRRTStar(**rrt_kwargs)
            planner_seg.set_problem_from_sample(seg_sample)
            planner_seg.set_external_pred_paths(seg_preds)
            best, cpu_t, gpu_t, tot_t = planner_seg.planning(
                model=None, device=None, is_rewiring=True, is_break=True, is_draw=False
            )
            sg_cpu += cpu_t; sg_gpu += gpu_t; sg_total += tot_t
            sg_nodes += len(planner_seg.paths)
            if not planner_seg.is_goal:
                sg_success_all = False
            if best is not None:
                sg_cost_total += best.cost
            status = "SUCCESS" if planner_seg.is_goal else "FAILED"
            cost_str = f"{best.cost:.1f}" if best else "N/A"
            print(f"[RRT* Seg {seg_i+1}] {status}  nodes={len(planner_seg.paths)}  cost={cost_str}  time={tot_t:.3f}s")
            rrt_subgoal_nodes.append(best)
        rrt_subgoal_metrics = dict(
            cpu=sg_cpu, gpu=gpu_time_seg, total=sg_cpu + gpu_time_seg,
            nodes=sg_nodes, success=sg_success_all,
            cost=sg_cost_total if sg_success_all else None,
        )

    def _extract_path_px(node, H, W):
        xs, ys = [], []
        curr = node
        while curr:
            xs.append(curr.x); ys.append(curr.y)
            curr = curr.parent
        return xs[::-1], ys[::-1]

    def _draw_rrt_metrics(ax, metrics: dict) -> None:
        success = metrics["success"]
        cost = metrics["cost"]
        cost_str = f"{cost:.1f}" if cost is not None else "N/A"
        status_str = "SUCCESS" if success else "FAILED"
        text = (
            f"── RRT* ──\n"
            f"Status : {status_str}\n"
            f"CPU    : {metrics['cpu']:.3f}s\n"
            f"GPU    : {metrics['gpu']:.3f}s\n"
            f"Total  : {metrics['total']:.3f}s\n"
            f"Nodes  : {metrics['nodes']}\n"
            f"Cost   : {cost_str}"
        )
        color = "#00e5ff" if success else "#ff6b6b"
        ax.text(
            0.02, 0.02, text, transform=ax.transAxes,
            va="bottom", ha="left", fontsize=7.5, color="white",
            fontfamily="monospace",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "black",
                  "alpha": 0.65, "edgecolor": color, "linewidth": 1.2},
        )

    # ── Visualization ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    titles = ["Direct: Start → Goal", f"Sub-goal Guided (k={k})"]

    for ax, title in zip(axes, titles):
        ax.imshow(map_np, cmap="gray", origin="upper", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])

        px, py = _to_pixel(gt_path, H, W)
        ax.plot(px, py, "b-", lw=1.5, alpha=0.9, label="GT path", zorder=4)

        if title.startswith("Direct"):
            all_pts = full_paths.reshape(-1, 2)
            px, py = _to_pixel(all_pts, H, W)
            ax.scatter(px, py, s=4, color="#9b59b6", alpha=0.25, zorder=2,
                       linewidths=0, label=f"Pred pts ×{num_samples}")
            if args.rrt and rrt_direct_node is not None:
                rxs, rys = _extract_path_px(rrt_direct_node, H, W)
                ax.plot(rxs, rys, "-", color="#00e5ff", lw=2.0, zorder=5, label="RRT* path")
            if args.rrt and rrt_direct_metrics is not None:
                _draw_rrt_metrics(ax, rrt_direct_metrics)
        else:
            if k == 0:
                # No sub-goals: show direct flow paths
                all_pts = full_paths.reshape(-1, 2)
                px, py = _to_pixel(all_pts, H, W)
                ax.scatter(px, py, s=4, color="#9b59b6", alpha=0.25, zorder=2,
                           linewidths=0, label=f"Pred pts ×{num_samples}")
                ax.text(0.5, 0.5, "k=0: direct path", transform=ax.transAxes,
                        ha="center", va="center", fontsize=11, color="yellow",
                        bbox={"facecolor": "black", "alpha": 0.5})
            else:
                for seg_i, (seg_paths, color) in enumerate(zip(all_seg_paths, seg_colors)):
                    all_pts = seg_paths.reshape(-1, 2)
                    px, py = _to_pixel(all_pts, H, W)
                    ax.scatter(px, py, s=4, color=color, alpha=0.3, zorder=2,
                               linewidths=0, label=f"Seg {seg_i+1}")
                # Sub-goal markers
                for sg_i, sg in enumerate(sub_goals_np):
                    sx_px = sg[0] * (W - 1)
                    sy_px = sg[1] * (H - 1)
                    ax.plot(sx_px, sy_px, "D", color="#f39c12", ms=9, zorder=5,
                            label=f"SG{sg_i+1}")
                    ax.annotate(f"SG{sg_i+1}", (sx_px, sy_px), xytext=(5, 4),
                                textcoords="offset points", fontsize=8,
                                color="#f39c12", fontweight="bold")

            if args.rrt:
                rrt_labeled = False
                for seg_node in rrt_subgoal_nodes:
                    if seg_node is not None:
                        rxs, rys = _extract_path_px(seg_node, H, W)
                        label = "RRT* path" if not rrt_labeled else None
                        ax.plot(rxs, rys, "-", color="#00e5ff", lw=2.0, zorder=5, label=label)
                        rrt_labeled = True
                if rrt_subgoal_metrics is not None:
                    _draw_rrt_metrics(ax, rrt_subgoal_metrics)

        sx_px = start_np[0] * (W - 1); sy_px = start_np[1] * (H - 1)
        gx_px = goal_np[0]  * (W - 1); gy_px = goal_np[1]  * (H - 1)
        ax.plot(sx_px, sy_px, "^", color="#2ecc71", ms=10, zorder=6, label="Start")
        ax.plot(gx_px, gy_px, "*", color="#e74c3c", ms=13, zorder=6, label="Goal")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    plt.suptitle(
        f"Count Policy – k={k}/{max_subgoals} sub-goals  (sample {idx})",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved → {save_path}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
