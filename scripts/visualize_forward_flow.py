#!/usr/bin/env python
"""
Visualize forward flow transformations (Data -> Noise).

Shows how ground truth data transforms from data distribution x -> z1 -> z2 -> ... -> z_final (Gaussian)

Usage:
    uv run python scripts/visualize_forward_flow.py
    uv run python scripts/visualize_forward_flow.py --num_examples 5
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from rlnf_rrt.data_pipeline.dataset import RLNFDataset
from rlnf_rrt.models.conditional_flow_planner import ConditionalFlowPlanner


def infer_conditioning_mode(checkpoint):
    config = checkpoint.get("config", None)
    if config is not None and hasattr(config, "conditioning_mode"):
        return getattr(config, "conditioning_mode")
    state_dict = checkpoint.get("model_state_dict", {})
    has_film = any(".film1." in k or ".film2." in k for k in state_dict.keys())
    return "film" if has_film else "concat"


class FastRLNFDataset(RLNFDataset):
    """Optimized dataset loader that only reads minimal necessary samples."""
    def __init__(self, split:str="test", noise_std:float=0.0, max_samples:int=100):
        self.split:str = split
        self.noise_std:float = noise_std
        self.data_path:str = f"{PROJECT_ROOT}/data/{split}"
        
        # Read full meta.csv to allow true random sampling from the distribution
        self.meta_data:pd.DataFrame = pd.read_csv(f"{self.data_path}/meta.csv")
        
        self.map_list: list[str] = []
        self.start_goal_list: list[str] = []
        self.gt_list: list[str] = []

        # experiment: only use clearance 1 and step size 1
        filtered_meta = self.meta_data[  
            (self.meta_data["clearance"] == 1) &
            (self.meta_data["step_size"] == 1)
        ]
        
        # Randomly sample
        if len(filtered_meta) > max_samples:
            self.filtered_meta = filtered_meta.sample(n=max_samples)
        else:
            self.filtered_meta = filtered_meta

        for _, row in self.filtered_meta.iterrows():
            self.map_list.append(f"{self.data_path}/map/{row['map_file']}")
            self.start_goal_list.append(f"{self.data_path}/start_goal/{row['start_goal_file']}")
            self.gt_list.append(f"{self.data_path}/gt_path/{row['gt_path_file']}")
            
    # __len__ and __getitem__ are inherited from RLNFDataset and work fine


def latent_normality_metrics(z: np.ndarray) -> dict:
    """Compute simple diagnostics for z ~ N(0, I) in 2D."""
    eps = 1e-6
    n, d = z.shape
    mu = z.mean(axis=0)
    cov = np.cov(z, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=np.float32)
    cov_reg = cov + np.eye(cov.shape[0]) * eps
    inv_cov = np.linalg.inv(cov_reg)

    centered = z - mu
    d2 = np.einsum("ni,ij,nj->n", centered, inv_cov, centered)
    mardia_kurtosis = float(np.mean(d2 ** 2))
    expected_mardia_kurtosis = float(d * (d + 2))

    r = np.linalg.norm(z, axis=1)
    r_mean = float(r.mean())
    r_std = float(r.std())
    # For chi distribution with k=2 (radius of N(0,I) in 2D)
    chi2_mean = float(np.sqrt(np.pi / 2.0))
    chi2_std = float(np.sqrt((4.0 - np.pi) / 2.0))

    var_diag = np.diag(cov)
    corr = float(np.corrcoef(z, rowvar=False)[0, 1]) if d >= 2 else 0.0

    return {
        "mean_x": float(mu[0]) if d >= 1 else 0.0,
        "mean_y": float(mu[1]) if d >= 2 else 0.0,
        "var_x": float(var_diag[0]) if d >= 1 else 0.0,
        "var_y": float(var_diag[1]) if d >= 2 else 0.0,
        "corr_xy": corr,
        "r_mean": r_mean,
        "r_std": r_std,
        "chi2_r_mean_ref": chi2_mean,
        "chi2_r_std_ref": chi2_std,
        "mardia_kurtosis": mardia_kurtosis,
        "mardia_kurtosis_ref": expected_mardia_kurtosis,
    }


def visualize_forward_steps(model, dataset, device, example_idx=None, save_path=None):
    """Visualize intermediate transformations through each coupling block in forward direction (x -> z)."""
    model.eval()
    
    # Get example
    if example_idx is None:
        example_idx = np.random.randint(len(dataset))
    
    data = dataset[example_idx]
    
    map_img = data["map"].unsqueeze(0).to(device)
    start = data["start"].unsqueeze(0).to(device)
    goal = data["goal"].unsqueeze(0).to(device)
    gt_path = data["gt_path"].unsqueeze(0).to(device)  # (1, T, 2)
    
    # Get intermediate transformations (Forward: x -> z)
    with torch.no_grad():
        intermediates = model.forward_with_intermediates(gt_path, map_img, start, goal)
    
    # Convert to numpy
    intermediates = [z[0].cpu().numpy() for z in intermediates]  # Each: (T, 2)
    start_np = start[0].cpu().numpy()
    goal_np = goal[0].cpu().numpy()
    map_np = map_img[0, 0].cpu().numpy()
    
    num_steps = len(intermediates)
    
    # Create figure
    cols = min(num_steps, 5)
    rows = (num_steps + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    
    if num_steps == 1:
        axes = [axes]
    elif rows == 1:
        # If result is array of axes (cols > 1), fine. If 1 col, it's array.
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]
    else:
        axes = axes.flatten()
    
    latent_metrics = latent_normality_metrics(intermediates[-1])
    for i, (ax, z) in enumerate(zip(axes, intermediates)):
        is_data_space = (i == 0)
        is_latent_space = (i == num_steps - 1)
        
        # Plot map (always useful as reference, even if z moves away)
        ax.imshow(map_np, cmap='gray', origin='lower', extent=[0, 1, 0, 1], alpha=0.3)
        
        # Plot trajectory
        color = 'green' if is_data_space else ('purple' if is_latent_space else 'blue')
        label = 'GT Path (x)' if is_data_space else ('Latent (z)' if is_latent_space else f'Step {i}')
        
        ax.scatter(z[:, 0], z[:, 1], c=color, s=15, alpha=0.7, label=label, zorder=5)
        
        # Plot distribution reference (Gaussian circle) for latent steps
        if not is_data_space and i > 0:
            # Draw unit circle to show standard normal scale
            ax.add_patch(Circle((0, 0), 1, fill=False, edgecolor='red', linestyle='--', alpha=0.5, label='Std Normal'))
            ax.add_patch(Circle((0, 0), 2, fill=False, edgecolor='red', linestyle=':', alpha=0.3))
        
        # Plot start/goal
        if is_data_space:
            ax.add_patch(Circle(start_np, 0.02, color='red', zorder=10, label='Start'))
            ax.add_patch(Circle(goal_np, 0.02, color='lime', zorder=10, label='Goal'))
        
        # Set limits
        if is_data_space:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f'Step {i}: Data Space (x)', fontsize=11, fontweight='bold')
        else:
            # Determine limits based on data range to ensure visibility
            # But keep 0,0 centered
            max_val = max(np.abs(z).max(), 2.5) # At least 2.5 (2.5 sigma)
            limit = max_val * 1.1
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            
            title = f'Step {i}: Latent (z) [Block {i}]' if is_latent_space else f'Step {i}: After Block {i}'
            ax.set_title(title, fontsize=11, fontweight='bold')
            
            # Add origin lines
            ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
            ax.axvline(0, color='gray', linestyle='-', alpha=0.3, linewidth=1)

        ax.set_aspect('equal')
        
        # Only add legend if it fits, or just once
        if i == 0 or i == num_steps - 1:
            ax.legend(fontsize=8, loc='upper right')

        if is_latent_space:
            text = (
                f"mu=({latent_metrics['mean_x']:.2f},{latent_metrics['mean_y']:.2f})\n"
                f"var=({latent_metrics['var_x']:.2f},{latent_metrics['var_y']:.2f})\n"
                f"corr={latent_metrics['corr_xy']:.2f}\n"
                f"r(mean/std)={latent_metrics['r_mean']:.2f}/{latent_metrics['r_std']:.2f}\n"
                f"mardia={latent_metrics['mardia_kurtosis']:.2f} (ref {latent_metrics['mardia_kurtosis_ref']:.2f})"
            )
            ax.text(
                0.03,
                0.03,
                text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="bottom",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )
    
    # Hide unused subplots
    for ax in axes[num_steps:]:
        ax.axis('off')
    
    fig.suptitle(f'Forward Flow Transformation (Example {example_idx})\nData (x) → Noise (z)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved forward flow steps to: {save_path}")
    plt.close()
    return latent_metrics


def main(args):
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("   Train a model first using: uv run python scripts/train_flow.py")
        return
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", None)
    conditioning_mode = infer_conditioning_mode(checkpoint)
    
    # Create model
    model = ConditionalFlowPlanner(
        num_blocks=getattr(config, "num_blocks", args.num_blocks) if config else args.num_blocks,
        sg_dim=getattr(config, "sg_dim", 2) if config else 2,
        position_embed_dim=getattr(config, "position_embed_dim", 128) if config else 128,
        map_embed_dim=getattr(config, "map_embed_dim", 256) if config else 256,
        cond_dim=getattr(config, "cond_dim", args.cond_dim) if config else args.cond_dim,
        hidden_dim=getattr(config, "hidden_dim", 128) if config else 128,
        s_max=getattr(config, "s_max", 2.0) if config else 2.0,
        conditioning_mode=conditioning_mode,
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"✅ Loaded model from epoch {checkpoint['epoch']}")
    print(f"   Conditioning mode: {conditioning_mode}")
    
    # Load dataset (Optimized)
    print("Loading train dataset (partial)...")
    dataset = FastRLNFDataset(split="train", noise_std=0.05, max_samples=args.num_examples)
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        print("❌ No samples loaded! Check dataset path.")
        return

    # Create output directory
    save_dir = PROJECT_ROOT / "result" / "visualization" / "forward_flow"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Visualize multiple examples
    print(f"\nGenerating {args.num_examples} forward visualizations...")
    
    # Use all loaded examples
    indices = np.arange(len(dataset))
    
    all_metrics = []
    for i, idx in enumerate(indices):
        save_path = save_dir / f"forward_steps_{i}_{timestamp}.png"
        metrics = visualize_forward_steps(
            model,
            dataset,
            device,
            example_idx=idx,
            save_path=save_path,
        )
        metrics["example_idx"] = int(idx)
        all_metrics.append(metrics)
    
    print(f"\n✅ Generated {len(dataset)} forward flow visualizations!")
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        metrics_path = save_dir / f"forward_metrics_{timestamp}.csv"
        df.to_csv(metrics_path, index=False)
        print(f"✅ Saved latent diagnostics to: {metrics_path}")
        print("Latent diagnostics mean:")
        print(
            df[["mean_x", "mean_y", "var_x", "var_y", "corr_xy", "r_mean", "r_std", "mardia_kurtosis"]]
            .mean()
            .to_string(float_format=lambda x: f"{x:.4f}")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Forward Flow (Data -> Noise)")
    
    parser.add_argument("--checkpoint", type=str, default="result/models/v1_best_model.pt",
                        help="Path to checkpoint")
    parser.add_argument("--num_examples", type=int, default=5,
                        help="Number of examples to visualize")
    parser.add_argument("--num_blocks", type=int, default=4,
                        help="Number of coupling blocks")
    parser.add_argument("--cond_dim", type=int, default=128,
                        help="Condition dimension")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    main(args)
