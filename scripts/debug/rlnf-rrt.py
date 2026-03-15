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
from rlnf_rrt.utils.config import load_toml, resolve_project_path
from rlnf_rrt.utils.utils import get_device
from flow_rrt_star import FlowRRTStar


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
    parser = argparse.ArgumentParser(description="Plot forward flow transformation steps")
    parser.add_argument("--eval-config", type=str, default="configs/eval/default.toml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=4)
    parser.add_argument("--save", type=str, default="outputs/figures/sub-goal-example.png")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--show", action="store_true")
    # RRT* arguments
    parser.add_argument("--rrt", action="store_true", help="Run RRT* planning and overlay result")
    parser.add_argument("--iter-num", type=int, default=3000)
    parser.add_argument("--expand-size", type=float, default=2.0)
    parser.add_argument("--near-distance", type=float, default=15.0)
    parser.add_argument("--clearance", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--goal-bias", type=float, default=0.1)
    parser.add_argument("--flow-guidance-prob", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    return parser

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
    model = _build_model_from_ckpt(ckpt, device, backbone=backbone, is_pe=is_pe)

    cond_image = sample["cond_image"].unsqueeze(0).to(device)  # (1, 3, H, W)
    start = sample["start"].unsqueeze(0).to(device)            # (1, 2)
    goal = sample["goal"].unsqueeze(0).to(device)              # (1, 2)
    gt_path = sample["gt_path"].cpu().numpy()                  # (N, 2)

    map_np = sample["cond_image"][0].cpu().numpy()             # (H, W) binary map
    H, W = map_np.shape

    # Sub-goals (normalized [0,1] coords): shape (num_sg, 2)
    sub_goals_np = np.array([[0.563, 0.528], [0.146, 0.569]], dtype=np.float32)

    # Waypoints: start → sg0 → sg1 → goal
    start_np = sample["start"].cpu().numpy()
    goal_np = sample["goal"].cpu().numpy()
    waypoints = [start_np, sub_goals_np[0], sub_goals_np[1], goal_np]

    num_points = gt_path.shape[0]
    num_segments = len(waypoints) - 1
    pts_per_seg = max(2, num_points // num_segments)
    num_samples = 6

    import time as _time

    # --- Generate paths ---
    for _ in range(20):
        z = torch.randn((1, num_points, 2), device=device, dtype=cond_image.dtype)
        _ = model.inverse(cond_image, start, goal, z)

    # Segmented paths: start→sg0, sg0→sg1, sg1→goal
    seg_colors = ["#e74c3c", "#e67e22", "#2ecc71"]
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
        all_seg_paths.append(np.stack(seg))  # (num_samples, pts_per_seg, 2)

    # Full path: start → goal
    full_paths = []
    _t0 = _time.perf_counter()
    with torch.no_grad():
        for _ in range(num_samples):
            z = torch.randn((1, num_points, 2), device=device, dtype=cond_image.dtype)
            pred, _ = model.inverse(cond_image, start, goal, z)
            full_paths.append(pred.squeeze(0).clamp(0, 1).cpu().numpy())
    gpu_time_direct = _time.perf_counter() - _t0
    full_paths = np.stack(full_paths)  # (num_samples, N, 2)

    # --- RRT* Planning ---
    rrt_direct_node = None
    rrt_direct_metrics = None   # dict with cpu, gpu, total, nodes, success, cost
    rrt_subgoal_nodes = []      # best_node per segment
    rrt_subgoal_metrics = None  # aggregated metrics for sub-goal run
    if args.rrt:
        clearance = args.clearance
        step_size = 1
        rrt_kwargs = dict(
            seed=args.seed,
            clearance=clearance,
            step_size=step_size,
            iter_num=args.iter_num,
            cspace_img_path=f"{split}:{idx}",
            is_neural_mode=True,
            near_distance=args.near_distance,
            expand_size=args.expand_size,
            threshold=args.threshold,
            num_guidance_samples=num_samples,
            goal_bias=args.goal_bias,
            flow_guidance_prob=args.flow_guidance_prob,
        )

        # Direct: start → goal using full_paths heatmap
        planner_direct = FlowRRTStar(**rrt_kwargs)
        planner_direct.set_problem_from_sample(sample)
        planner_direct.set_external_pred_paths(list(full_paths))
        rrt_direct_node, cpu_t, gpu_t, tot_t = planner_direct.planning(
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

        # Sub-goal: one RRT* per segment, chain them
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
            nodes=sg_nodes,
            success=sg_success_all,
            cost=sg_cost_total if sg_success_all else None,
        )

    def _extract_path_px(node, H, W):
        """Node 체인을 픽셀 (x, y) 리스트로 변환."""
        xs, ys = [], []
        curr = node
        while curr:
            xs.append(curr.x)
            ys.append(curr.y)
            curr = curr.parent
        return xs[::-1], ys[::-1]

    def _draw_rrt_metrics(ax, metrics: dict) -> None:
        """RRT* 결과 지표를 ax 좌측 하단에 텍스트 박스로 표시."""
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
            0.02, 0.02, text,
            transform=ax.transAxes,
            va="bottom", ha="left",
            fontsize=7.5,
            color="white",
            fontfamily="monospace",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "black", "alpha": 0.65, "edgecolor": color, "linewidth": 1.2},
        )

    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    titles = ["Direct: Start → Goal", "Sub-goal Guided"]
    for ax, title in zip(axes, titles):
        ax.imshow(map_np, cmap="gray", origin="upper", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

        # GT path as line
        px, py = _to_pixel(gt_path, H, W)
        ax.plot(px, py, "b-", lw=1.5, alpha=0.9, label="GT path", zorder=4)

        if title.startswith("Direct"):
            # Scatter all predicted points: (num_samples * N, 2)
            all_pts = full_paths.reshape(-1, 2)
            px, py = _to_pixel(all_pts, H, W)
            ax.scatter(px, py, s=4, color="#9b59b6", alpha=0.25, zorder=2,
                       linewidths=0, label=f"Pred pts ×{num_samples}")
            # Overlay RRT* direct path
            if args.rrt and rrt_direct_node is not None:
                rxs, rys = _extract_path_px(rrt_direct_node, H, W)
                ax.plot(rxs, rys, "-", color="#00e5ff", lw=2.0, zorder=5, label="RRT* path")
            if args.rrt and rrt_direct_metrics is not None:
                _draw_rrt_metrics(ax, rrt_direct_metrics)
        else:
            for seg_i, (seg_paths, color) in enumerate(zip(all_seg_paths, seg_colors)):
                # (num_samples, pts_per_seg, 2) → (num_samples*pts_per_seg, 2)
                all_pts = seg_paths.reshape(-1, 2)
                px, py = _to_pixel(all_pts, H, W)
                ax.scatter(px, py, s=4, color=color, alpha=0.3, zorder=2,
                           linewidths=0, label=f"Seg {seg_i+1} pts")
            # Sub-goal markers
            for sg_i, sg in enumerate(sub_goals_np):
                sx, sy = sg[0] * (W - 1), sg[1] * (H - 1)
                ax.plot(sx, sy, "D", color="#f39c12", ms=9, zorder=5,
                        label=f"SG{sg_i+1}")
                ax.annotate(f"SG{sg_i+1}", (sx, sy), xytext=(5, 4),
                            textcoords="offset points", fontsize=8,
                            color="#f39c12", fontweight="bold")
            # Overlay RRT* sub-goal paths
            if args.rrt:
                rrt_labeled = False
                for seg_i, seg_node in enumerate(rrt_subgoal_nodes):
                    if seg_node is not None:
                        rxs, rys = _extract_path_px(seg_node, H, W)
                        label = "RRT* path" if not rrt_labeled else None
                        ax.plot(rxs, rys, "-", color="#00e5ff", lw=2.0, zorder=5, label=label)
                        rrt_labeled = True
                if rrt_subgoal_metrics is not None:
                    _draw_rrt_metrics(ax, rrt_subgoal_metrics)

        # Start / Goal markers
        sx, sy = start_np[0] * (W - 1), start_np[1] * (H - 1)
        gx, gy = goal_np[0] * (W - 1), goal_np[1] * (H - 1)
        ax.plot(sx, sy, "^", color="#2ecc71", ms=10, zorder=6, label="Start")
        ax.plot(gx, gy, "*", color="#e74c3c", ms=13, zorder=6, label="Goal")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    plt.suptitle(f"Flow Model – Sub-goal Path Guidance  (sample {idx})",
                 fontsize=13, y=1.01)
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