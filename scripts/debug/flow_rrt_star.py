from __future__ import annotations

import argparse
import random
import sys
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rlnf_rrt.data.dataset import RLNFDataset
from rlnf_rrt.models.flow import Flow
from rlnf_rrt.utils.config import load_toml, resolve_project_path
from rlnf_rrt.utils.node import Node
from rlnf_rrt.utils.utils import get_device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FlowRRTStar:
    def __init__(
        self,
        seed,
        clearance,
        step_size,
        iter_num,
        cspace_img_path,
        is_neural_mode=True,
        near_distance=15,
        robot_radius=1,
        expand_size=2,
        threshold=0.05,
        num_guidance_samples=64,
        goal_bias=0.1,
        flow_guidance_prob=0.5,
    ):
        self.clearance = clearance
        self.step_size = step_size
        self.init_x, self.init_y = None, None
        self.goal_x, self.goal_y = None, None
        self.iter_num = iter_num
        self.near_distance = near_distance
        self.cspace_img_path = cspace_img_path
        self.is_goal = False
        self.robot_radius = robot_radius
        self.expand_size = expand_size
        self.seed = seed
        self.is_neural_mode = is_neural_mode
        self.paths = []
        self.gamma_rrt = 20.0
        self.threshold = float(threshold)
        self.num_guidance_samples = int(num_guidance_samples)
        self.goal_bias = float(goal_bias)
        self.flow_guidance_prob = float(flow_guidance_prob)

        self.cond_image = None
        self.start = None
        self.goal = None
        self.gt_path = None
        self.grid_map = None
        self.plot_map = None
        self.non_uniform_map = None
        self.external_pred_paths = None

    def set_problem_from_sample(self, sample: dict[str, torch.Tensor]) -> None:
        self.cond_image = sample["cond_image"].clone().float()
        self.start = sample["start"].clone().float()
        self.goal = sample["goal"].clone().float()
        self.gt_path = sample["gt_path"].clone().float()

    def set_external_pred_paths(self, pred_paths: list[np.ndarray] | None) -> None:
        self.external_pred_paths = pred_paths

    def flow_model(self, model, device, is_draw):
        if self.cond_image is None or self.start is None or self.goal is None:
            raise RuntimeError("Problem is not set. Call set_problem_from_sample first.")

        cond_image = self.cond_image
        map_np = cond_image[0].cpu().numpy().astype(np.float32)
        h, w = map_np.shape
        self.grid_map = (map_np < 0.5).astype(np.uint8)

        self.init_x = float(self.start[0].item() * (w - 1))
        self.init_y = float(self.start[1].item() * (h - 1))
        self.goal_x = float(self.goal[0].item() * (w - 1))
        self.goal_y = float(self.goal[1].item() * (h - 1))

        base = np.stack([map_np, map_np, map_np], axis=-1)
        self.plot_map = base.copy()

        total_time = 0.0
        out = np.ones((h, w), dtype=np.float32)

        if is_draw:
            plt.cla()
            plt.imshow(base, interpolation="nearest")

        if self.is_neural_mode:
            # Optionally reuse pre-saved predictions from outputs/predictions/<pred-name>.
            if self.external_pred_paths is not None:
                heat = np.zeros((h, w), dtype=np.float32)
                for pred in self.external_pred_paths:
                    pred_arr = np.asarray(pred, dtype=np.float32)
                    if pred_arr.size == 0:
                        continue
                    pred_arr = np.clip(pred_arr, 0.0, 1.0)
                    px = np.clip(np.round(pred_arr[:, 0] * (w - 1)).astype(int), 0, w - 1)
                    py = np.clip(np.round(pred_arr[:, 1] * (h - 1)).astype(int), 0, h - 1)
                    heat[py, px] += 1.0
                if heat.max() > 0:
                    heat = heat / heat.max()
                out = heat

                if is_draw:
                    alpha = out.copy()
                    alpha[out < self.threshold] = 0.0
                    alpha = np.clip(alpha * 5.0, 0.0, 0.9)
                    plt.imshow(out, cmap="plasma", alpha=alpha, interpolation="bilinear")
                return out, total_time

            if model is None or device is None:
                raise RuntimeError("Model/device are required when is_neural_mode=True.")

            model.eval()
            cond = cond_image.unsqueeze(0).to(device)
            start = self.start.unsqueeze(0).to(device)
            goal = self.goal.unsqueeze(0).to(device)
            t = int(self.gt_path.shape[0])

            heat = np.zeros((h, w), dtype=np.float32)
            with torch.no_grad():
                t0 = time.perf_counter()
                for _ in range(self.num_guidance_samples):
                    z = torch.randn((1, t, 2), device=device, dtype=cond.dtype)
                    pred_path, _ = model.inverse(cond, start, goal, z)
                    pred = pred_path.squeeze(0).clamp(0.0, 1.0).cpu().numpy()
                    px = np.clip(np.round(pred[:, 0] * (w - 1)).astype(int), 0, w - 1)
                    py = np.clip(np.round(pred[:, 1] * (h - 1)).astype(int), 0, h - 1)
                    heat[py, px] += 1.0
                t1 = time.perf_counter()
            total_time = t1 - t0

            if heat.max() > 0:
                heat = heat / heat.max()
            out = heat

            if is_draw:
                alpha = out.copy()
                alpha[out < self.threshold] = 0.0
                alpha = np.clip(alpha * 5.0, 0.0, 0.9)
                plt.imshow(out, cmap="plasma", alpha=alpha, interpolation="bilinear")

        if is_draw:
            plt.plot(self.init_x, self.init_y, "ob")
            plt.plot(self.goal_x, self.goal_y, "xr")

        return out, total_time

    def prepare_non_uniform(self, threshold=0.5):
        h, w = self.non_uniform_map.shape
        flat = self.non_uniform_map.reshape(-1).astype(np.float64)
        if flat.sum() == 0:
            flat = np.ones_like(flat)
        flat[flat < threshold] = 0.0
        if flat.sum() == 0:
            flat = np.ones_like(flat)
        self.flat_prob = flat / flat.sum()
        self.h, self.w = h, w

    def sample_from_non_uniform_map(self):
        idx = np.random.choice(len(self.flat_prob), p=self.flat_prob)
        return idx // self.w, idx % self.w

    def get_random_node(self):
        p = random.random()
        if p < self.goal_bias:
            x, y = self.goal_x, self.goal_y
        elif self.is_neural_mode and random.random() < self.flow_guidance_prob:
            y, x = self.sample_from_non_uniform_map()
        else:
            x = random.uniform(1, self.w - 1)
            y = random.uniform(1, self.h - 1)
        return Node(x, y)

    def get_nearest_node(self, rand):
        distances = [np.hypot(node.x - rand.x, node.y - rand.y) for node in self.paths]
        return self.paths[int(np.argmin(distances))]

    def steer(self, near, rand):
        dist = np.hypot(rand.x - near.x, rand.y - near.y)
        if dist <= self.expand_size:
            return Node(rand.x, rand.y, parent=near, cost=near.cost + dist), dist

        theta = np.arctan2(rand.y - near.y, rand.x - near.x)
        new_x = near.x + self.expand_size * np.cos(theta)
        new_y = near.y + self.expand_size * np.sin(theta)
        return Node(new_x, new_y, parent=near, cost=near.cost + self.expand_size), self.expand_size

    def is_collision(self, node):
        if not (0 <= node.x < self.w and 0 <= node.y < self.h):
            return True
        x, y = int(round(node.x)), int(round(node.y))
        clr = int(round(self.clearance))

        y_start, y_end = max(0, y - clr), min(self.h, y + clr + 1)
        x_start, x_end = max(0, x - clr), min(self.w, x + clr + 1)
        if np.any(self.grid_map[y_start:y_end, x_start:x_end] == 1):
            return True
        return False

    def get_near_ids(self, new):
        n = len(self.paths)
        if n <= 1:
            return [0]
        r = min(self.gamma_rrt * np.sqrt(np.log(n) / n), self.near_distance)

        node_idxs = []
        for i, node in enumerate(self.paths):
            if np.hypot(node.x - new.x, node.y - new.y) <= r:
                node_idxs.append(i)
        return node_idxs

    def choose_parent(self, near_by_vertices, nearest, new):
        candi_parent, cost_min = nearest, new.cost
        for near_id in near_by_vertices:
            near = self.paths[near_id]
            dist = np.hypot(near.x - new.x, near.y - new.y)
            if not self.is_collision(new) and near.cost + dist < cost_min:
                candi_parent, cost_min = near, near.cost + dist
        return candi_parent, cost_min

    def update_subtree_cost(self, start_node):
        queue = deque([start_node])
        while queue:
            curr = queue.popleft()
            for child in curr.children:
                dist = np.hypot(child.x - curr.x, child.y - curr.y)
                child.cost = curr.cost + dist
                queue.append(child)

    def rewire(self, near_by_vertices, parent, new):
        for near_id in near_by_vertices:
            near = self.paths[near_id]
            if near is parent:
                continue

            dist = np.hypot(new.x - near.x, new.y - near.y)
            if new.cost + dist < near.cost:
                if not self.is_collision(near):
                    if near.parent and near in near.parent.children:
                        near.parent.children.remove(near)
                    near.parent = new
                    near.cost = new.cost + dist
                    new.children.append(near)
                    self.update_subtree_cost(near)

    def planning(self, model=None, device=None, is_rewiring=True, is_break=False, is_draw=False):
        if self.is_neural_mode:
            self.non_uniform_map, gpu_time = self.flow_model(model, device, is_draw)
        else:
            _, gpu_time = self.flow_model(None, None, is_draw)
            self.non_uniform_map = np.ones_like(self.grid_map, dtype=np.float32)

        self.prepare_non_uniform(threshold=self.threshold)
        init_node = Node(self.init_x, self.init_y)
        self.paths = [init_node]
        best_node = None

        start = time.perf_counter()
        for _ in range(self.iter_num):
            rand = self.get_random_node()
            nearest = self.get_nearest_node(rand)
            new, _ = self.steer(nearest, rand)

            if not self.is_collision(new):
                near_ids = self.get_near_ids(new)
                parent, cost = self.choose_parent(near_ids, nearest, new)

                new.parent = parent
                new.cost = cost
                parent.children.append(new)
                self.paths.append(new)

                if is_rewiring:
                    self.rewire(near_ids, parent, new)

                if is_draw:
                    self.plot_explore_edge(new)

                dist_to_goal = np.hypot(new.x - self.goal_x, new.y - self.goal_y)
                if dist_to_goal <= self.expand_size:
                    final_node = Node(self.goal_x, self.goal_y, parent=new, cost=new.cost + dist_to_goal)
                    if not self.is_collision(final_node):
                        if best_node is None or final_node.cost < best_node.cost:
                            best_node = final_node
                            self.is_goal = True
                            if is_break:
                                break

        end = time.perf_counter()
        cpu_time = end - start
        total_time = cpu_time + gpu_time
        return best_node, cpu_time, gpu_time, total_time

    def plot_explore_edge(self, new):
        if new.parent:
            plt.plot([new.parent.x, new.x], [new.parent.y, new.y], "-g", alpha=0.3)
            if len(self.paths) % 5 == 0:
                plt.pause(0.001)


def plot_final_path(final_node):
    xlist, ylist = [], []
    curr = final_node
    while curr:
        xlist.append(curr.x)
        ylist.append(curr.y)
        curr = curr.parent
    plt.plot(xlist[::-1], ylist[::-1], "-r", linewidth=2)


def _draw_metrics(cpu_time: float, gpu_time: float, total_time: float, nodes: int, cost: float | None) -> None:
    cost_str = f"{cost:.2f}" if cost is not None else "N/A"
    text = (
        f"CPU: {cpu_time:.4f}s\n"
        f"GPU: {gpu_time:.4f}s\n"
        f"Total: {total_time:.4f}s\n"
        f"Nodes: {nodes}\n"
        f"Cost: {cost_str}"
    )
    ax = plt.gca()
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color="white",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
    )


def _build_model_from_ckpt(ckpt: dict, device: torch.device, backbone: str = "resnet34") -> Flow:
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
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _load_prediction_paths(pred_name: str, eval_cfg: dict, ds_idx: int) -> tuple[Path, list[np.ndarray]]:
    pred_root = resolve_project_path(eval_cfg.get("prediction_dir", "outputs/predictions"))
    pred_dir = pred_root / pred_name
    if not pred_dir.exists() or not pred_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")

    files = sorted(pred_dir.glob(f"pred_{ds_idx:07d}_s*.npy"))
    if not files:
        raise FileNotFoundError(
            f"No prediction files found for index={ds_idx} under {pred_dir} "
            f"(expected pattern: pred_{ds_idx:07d}_s*.npy)"
        )
    preds = [np.load(f).astype(np.float32) for f in files]
    return pred_dir, preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Flow-guided RRT* test")
    parser.add_argument("--eval-config", type=str, default="configs/eval/default.toml")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--iter-num", type=int, default=5000)
    parser.add_argument("--expand-size", type=float, default=2.0)
    parser.add_argument("--near-distance", type=float, default=15.0)
    parser.add_argument("--clearance", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-guidance-samples", type=int, default=16)
    parser.add_argument("--pred-name", type=str, default=None, help="Subdirectory name under outputs/predictions.")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--goal-bias", type=float, default=0.1, help="Probability of directly sampling the goal.")
    parser.add_argument(
        "--flow-guidance-prob",
        type=float,
        default=0.6,
        help="Probability of sampling from flow heatmap (when not goal-sampling).",
    )
    parser.add_argument("--uniform", action="store_true", help="Disable neural guidance and run uniform RRT* sampling.")
    parser.add_argument("--no-rewire", action="store_true")
    parser.add_argument("--break-on-goal", default=True, action="store_true", help="Stop planning immediately when a path to goal is found.")
    parser.add_argument("--draw", action="store_true")
    args = parser.parse_args()
    if not (0.0 <= args.goal_bias <= 1.0):
        raise ValueError(f"--goal-bias must be in [0, 1], got {args.goal_bias}")
    if not (0.0 <= args.flow_guidance_prob <= 1.0):
        raise ValueError(f"--flow-guidance-prob must be in [0, 1], got {args.flow_guidance_prob}")

    set_seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    cfg = load_toml(args.eval_config)
    data_cfg = cfg["data"]
    eval_cfg = cfg["eval"]

    split = args.split if args.split is not None else str(data_cfg.get("split", "test"))
    clearance = int(args.clearance if args.clearance is not None else data_cfg.get("clearance", 2))
    step_size = int(args.step_size if args.step_size is not None else data_cfg.get("step_size", 1))

    ds = RLNFDataset(
        split=split,
        data_root=resolve_project_path(data_cfg.get("data_root", "data")),
        noise_std=0.0,
        num_points=int(data_cfg["num_points"]),
        clearance=clearance,
        step_size=step_size,
    )
    if len(ds) == 0:
        raise RuntimeError(f"Dataset is empty for split='{split}', clearance={clearance}, step_size={step_size}.")

    idx = max(0, min(args.index, len(ds) - 1))
    sample = ds[idx]

    model = None
    pred_dir = None
    pred_paths = None
    if args.pred_name is not None:
        pred_dir, pred_paths = _load_prediction_paths(args.pred_name, eval_cfg, idx)
        print(f"prediction_dir: {pred_dir}")
        print(f"prediction_samples: {len(pred_paths)}")
    elif not args.uniform:
        ckpt_path = resolve_project_path(eval_cfg["checkpoint"])
        ckpt = torch.load(ckpt_path, map_location=device)
        model = _build_model_from_ckpt(ckpt, device, backbone=eval_cfg.get("backbone", "resnet34"))
        print(f"checkpoint: {ckpt_path}")

    planner = FlowRRTStar(
        seed=args.seed,
        clearance=clearance,
        step_size=step_size,
        iter_num=args.iter_num,
        cspace_img_path=f"{split}:{idx}",
        is_neural_mode=(not args.uniform),
        near_distance=args.near_distance,
        expand_size=args.expand_size,
        threshold=args.threshold,
        num_guidance_samples=args.num_guidance_samples,
        goal_bias=args.goal_bias,
        flow_guidance_prob=args.flow_guidance_prob,
    )
    planner.set_problem_from_sample(sample)
    planner.set_external_pred_paths(pred_paths)
    best_node, cpu_time, gpu_time, total_time = planner.planning(
        model=model,
        device=device,
        is_rewiring=(not args.no_rewire),
        is_break=args.break_on_goal,
        is_draw=args.draw,
    )

    print(f"sample: split={split}, index={idx}")
    print(
        f"times: cpu={cpu_time:.4f}s gpu={gpu_time:.4f}s total={total_time:.4f}s "
        f"nodes={len(planner.paths)} cost={(best_node.cost if best_node is not None else float('nan')):.2f}"
    )
    if best_node:
        print(f"Success! Time: {total_time:.4f}s, Nodes: {len(planner.paths)}, Cost: {best_node.cost:.2f}")
        # plt.imshow(planner.plot_map, interpolation="nearest")
        plot_final_path(best_node)
        plt.plot(planner.init_x, planner.init_y, "ob", label="Start")
        plt.plot(planner.goal_x, planner.goal_y, "xr", label="Goal")
        _draw_metrics(cpu_time, gpu_time, total_time, len(planner.paths), best_node.cost)
        plt.legend()
        plt.title("Flow-guided RRT*")
        plt.show()
    else:
        plt.imshow(planner.plot_map, interpolation="nearest")
        plt.plot(planner.init_x, planner.init_y, "ob", label="Start")
        plt.plot(planner.goal_x, planner.goal_y, "xr", label="Goal")
        _draw_metrics(cpu_time, gpu_time, total_time, len(planner.paths), None)
        plt.legend()
        plt.title("Flow-guided RRT* (failed)")
        plt.show()
        print("Failed to find path.")


if __name__ == "__main__":
    main()
