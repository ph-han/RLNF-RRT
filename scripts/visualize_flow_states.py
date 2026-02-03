"""
Example:
  python scripts/visualize_flow_states.py --ckpt outputs/model.pt --split valid --idx 0 --num_samples 5000 --max_panels 8 --device cuda
"""

import argparse
import math
import os
from typing import List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation

from rlnf_rrt.models.CustomPlannerFlows import CustomPlannerFlows
from rlnf_rrt.data_pipeline.utils import get_device

try:
    from rlnf_rrt.data_pipeline.custom_dataset import RLNFDataset
except Exception:
    RLNFDataset = None


def _strip_module_prefix(state_dict: dict) -> dict:
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _parse_masks_string(masks_str: str) -> List[List[float]]:
    masks = []
    for layer in masks_str.split(";"):
        layer = layer.strip()
        if not layer:
            continue
        mask_vals = [float(x.strip()) for x in layer.split(",") if x.strip() != ""]
        masks.append(mask_vals)
    return masks


def _load_masks_from_file(path: str) -> List[List[float]]:
    if path.endswith(".pt") or path.endswith(".pth"):
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict) and "masks" in data:
            return data["masks"]
        return data
    if path.endswith(".npy"):
        data = np.load(path, allow_pickle=True)
        return data.tolist()
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return _parse_masks_string(text)


def _infer_masks_from_state(state_dict: dict) -> Optional[List[List[float]]]:
    masks = []
    i = 0
    while True:
        key = f"flow.layers.{i}.mask"
        if key not in state_dict:
            break
        mask_tensor = state_dict[key].detach().cpu().float()
        masks.append(mask_tensor.tolist())
        i += 1
    return masks if masks else None


def _infer_hidden_and_condition_dim(state_dict: dict) -> Tuple[Optional[int], Optional[int]]:
    key = "flow.layers.0.s_net.0.weight"
    if key not in state_dict:
        return None, None
    weight = state_dict[key]
    hidden_dim = int(weight.shape[0])
    condition_input_dim = int(weight.shape[1])
    mask_key = "flow.layers.0.mask"
    if mask_key not in state_dict:
        return hidden_dim, None
    input_dim = int(state_dict[mask_key].numel())
    condition_dim = condition_input_dim - input_dim
    return hidden_dim, condition_dim


def load_model(args, device: torch.device) -> CustomPlannerFlows:
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    state_dict = _strip_module_prefix(state_dict)

    masks = None
    if isinstance(ckpt, dict) and "masks" in ckpt:
        masks = ckpt["masks"]
    if masks is None:
        masks = _infer_masks_from_state(state_dict)
    if masks is None and args.masks is not None:
        masks = _parse_masks_string(args.masks)
    if masks is None and args.masks_file is not None:
        masks = _load_masks_from_file(args.masks_file)
    if masks is None:
        raise ValueError("Could not infer masks from checkpoint. Provide --masks or --masks_file.")

    hidden_dim = args.hidden_dim
    env_latent_dim = args.env_latent_dim
    if hidden_dim is None or env_latent_dim is None:
        inferred_hidden_dim, inferred_condition_dim = _infer_hidden_and_condition_dim(state_dict)
        if hidden_dim is None:
            hidden_dim = inferred_hidden_dim
        if env_latent_dim is None and inferred_condition_dim is not None:
            env_latent_dim = inferred_condition_dim - args.state_dim * 2
    if hidden_dim is None or env_latent_dim is None:
        raise ValueError("Could not infer hidden_dim/env_latent_dim from checkpoint. Provide CLI args.")

    model = CustomPlannerFlows(masks, hidden_dim, env_latent_dim, state_dim=args.state_dim).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def get_condition(model: CustomPlannerFlows, args, device: torch.device):
    map_img = None
    start = None
    goal = None
    if RLNFDataset is not None:
        try:
            dataset = RLNFDataset(dataset_root_path=args.dataset_root, split=args.split)
            sample = dataset[args.idx]
            map_img = sample["map"].unsqueeze(0).to(device)
            start = sample["start"].unsqueeze(0).to(device)
            goal = sample["goal"].unsqueeze(0).to(device)
        except Exception:
            map_img = None

    if map_img is None:
        map_img = torch.zeros(1, 1, args.map_size, args.map_size, device=device, dtype=torch.float32)
        start = torch.zeros(1, args.state_dim, device=device, dtype=torch.float32)
        goal = torch.zeros(1, args.state_dim, device=device, dtype=torch.float32)

    with torch.no_grad():
        condition = model._get_condition(map_img, start, goal)
    return condition


def collect_states(model: CustomPlannerFlows, condition: torch.Tensor, num_samples: int, include_squash: bool):
    device = condition.device
    input_dim = model.flow.layers[0].input_dim
    z = torch.randn(num_samples, input_dim, device=device, dtype=torch.float32)
    cond = condition.expand(num_samples, -1)

    states = [z]
    with torch.no_grad():
        for layer in model.flow.layers:
            z, _ = layer(z, cond)
            states.append(z)
        if include_squash:
            tanh_z = torch.tanh(z)
            z_squash = (tanh_z + 1.0) / 2.0
            states.append(z_squash)
    return states


def _pca_project(states_np: List[np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    all_data = np.concatenate(states_np, axis=0)
    mean = all_data.mean(axis=0, keepdims=True)
    centered = all_data - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    projected = [(s - mean) @ components.T for s in states_np]
    return projected, mean, components


def _select_indices(num_states: int, max_panels: int) -> List[int]:
    if max_panels is None or num_states <= max_panels:
        return list(range(num_states))
    idxs = np.linspace(0, num_states - 1, max_panels, dtype=int).tolist()
    seen = set()
    uniq = []
    for idx in idxs:
        if idx not in seen:
            uniq.append(idx)
            seen.add(idx)
    return uniq


def plot_states(states: List[torch.Tensor], labels: List[str], max_panels: int, save_path: str):
    states_np = [s.detach().cpu().numpy().reshape(s.shape[0], -1) for s in states]
    input_dim = states_np[0].shape[1]
    sel = _select_indices(len(states_np), max_panels)
    states_sel = [states_np[i] for i in sel]
    labels_sel = [labels[i] for i in sel]

    if input_dim > 2:
        states_sel, _, _ = _pca_project(states_sel)
        input_dim = 2

    if input_dim == 1:
        all_vals = np.concatenate(states_sel, axis=0).flatten()
        x_min, x_max = float(all_vals.min()), float(all_vals.max())
    else:
        all_vals = np.concatenate(states_sel, axis=0)
        x_min, x_max = float(all_vals[:, 0].min()), float(all_vals[:, 0].max())
        y_min, y_max = float(all_vals[:, 1].min()), float(all_vals[:, 1].max())

    n_panels = len(states_sel)
    cols = int(math.ceil(math.sqrt(n_panels)))
    rows = int(math.ceil(n_panels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for ax in axes[n_panels:]:
        ax.axis("off")

    for ax, data, title in zip(axes, states_sel, labels_sel):
        if input_dim == 1:
            ax.hist(data.flatten(), bins=50, density=True)
            ax.set_xlim(x_min, x_max)
        else:
            x = data[:, 0]
            y = data[:, 1]
            bins = 60
            hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[x_min, x_max], [y_min, y_max]], density=True)
            xcenters = 0.5 * (xedges[:-1] + xedges[1:])
            ycenters = 0.5 * (yedges[:-1] + yedges[1:])
            xx, yy = np.meshgrid(xcenters, ycenters)
            ax.contour(xx, yy, hist.T, levels=8)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        ax.set_title(title)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_gif(states: List[torch.Tensor], labels: List[str], save_path: str):
    states_np = [s.detach().cpu().numpy().reshape(s.shape[0], -1) for s in states]
    input_dim = states_np[0].shape[1]

    if input_dim > 2:
        states_np, _, _ = _pca_project(states_np)
        input_dim = 2

    if input_dim == 1:
        all_vals = np.concatenate(states_np, axis=0).flatten()
        x_min, x_max = float(all_vals.min()), float(all_vals.max())
    else:
        all_vals = np.concatenate(states_np, axis=0)
        x_min, x_max = float(all_vals[:, 0].min()), float(all_vals[:, 0].max())
        y_min, y_max = float(all_vals[:, 1].min()), float(all_vals[:, 1].max())

    fig, ax = plt.subplots(figsize=(5, 5))

    def _plot_frame(idx: int):
        ax.clear()
        data = states_np[idx]
        title = labels[idx]
        if input_dim == 1:
            ax.hist(data.flatten(), bins=50, density=True)
            ax.set_xlim(x_min, x_max)
        else:
            x = data[:, 0]
            y = data[:, 1]
            bins = 60
            hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[x_min, x_max], [y_min, y_max]], density=True)
            xcenters = 0.5 * (xedges[:-1] + xedges[1:])
            ycenters = 0.5 * (yedges[:-1] + yedges[1:])
            xx, yy = np.meshgrid(xcenters, ycenters)
            ax.contour(xx, yy, hist.T, levels=8)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        ax.set_title(title)

    anim = animation.FuncAnimation(fig, _plot_frame, frames=len(states_np), interval=300)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    anim.save(save_path, writer=animation.PillowWriter(fps=3))
    plt.close(fig)


def build_labels(num_layers: int, include_squash: bool) -> List[str]:
    labels = [f"z{idx}" for idx in range(num_layers + 1)]
    if include_squash:
        labels.append("squash")
    return labels


def resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg is None:
        return get_device()
    device_arg = device_arg.lower()
    if device_arg == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if device_arg == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def main():
    parser = argparse.ArgumentParser(description="Visualize flow states per coupling layer.")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--max_panels", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--include_squash", action="store_true")
    parser.add_argument("--save_gif", action="store_true")

    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--env_latent_dim", type=int, default=None)
    parser.add_argument("--state_dim", type=int, default=2)
    parser.add_argument("--masks", type=str, default=None, help="Semicolon-separated masks, e.g. '1,0;0,1'")
    parser.add_argument("--masks_file", type=str, default=None, help="Path to masks file (.pt/.pth/.npy or text)")

    parser.add_argument("--dataset_root", type=str, default="data")
    parser.add_argument("--map_size", type=int, default=224)

    args = parser.parse_args()

    device = resolve_device(args.device)
    model = load_model(args, device)
    condition = get_condition(model, args, device)
    states = collect_states(model, condition, args.num_samples, args.include_squash)
    labels = build_labels(len(model.flow.layers), args.include_squash)

    out_path = os.path.join("outputs", f"flow_states_idx{args.idx}.png")
    plot_states(states, labels, args.max_panels, out_path)
    if args.save_gif:
        gif_path = os.path.join("outputs", f"flow_states_idx{args.idx}.gif")
        save_gif(states, labels, gif_path)


if __name__ == "__main__":
    main()
