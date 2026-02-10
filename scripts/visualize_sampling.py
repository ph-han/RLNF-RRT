#!/usr/bin/env python
"""
Visualize sample distributions from trained Flow model.

Usage:
    uv run python scripts/visualize_sampling.py
    uv run python scripts/visualize_sampling.py --num_samples 500 --num_examples 4
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from rlnf_rrt.data_pipeline.dataset import RLNFDataset
from rlnf_rrt.models.conditional_flow_planner import ConditionalFlowPlanner


def to_plot_xy(points: np.ndarray) -> np.ndarray:
    """Convert dataset/image-style coords (y-down) to plot coords (y-up)."""
    out = points.copy()
    out[..., 1] = 1.0 - out[..., 1]
    return out


def infer_conditioning_mode(checkpoint):
    config = checkpoint.get("config", None)
    if config is not None and hasattr(config, "conditioning_mode"):
        return getattr(config, "conditioning_mode")
    state_dict = checkpoint.get("model_state_dict", {})
    has_film = any(".film1." in k or ".film2." in k for k in state_dict.keys())
    return "film" if has_film else "concat"


def sample_collision_stats(samples: np.ndarray, map_np: np.ndarray) -> dict:
    """
    Collision in dataset coords (x, y) where y increases downward in image row index.
    map_np: free=1, obstacle=0.
    """
    h, w = map_np.shape
    x = samples[:, 0]
    y = samples[:, 1]

    oob = (x < 0.0) | (x > 1.0) | (y < 0.0) | (y > 1.0)
    col = np.clip((x * w).astype(int), 0, w - 1)
    row = np.clip((y * h).astype(int), 0, h - 1)

    obstacle = map_np[row, col] <= 0.5
    collision = obstacle | oob
    free = (~collision)

    return {
        "collision_rate": float(collision.mean()),
        "obstacle_hit_rate": float(obstacle.mean()),
        "oob_rate": float(oob.mean()),
        "free_rate": float(free.mean()),
    }


def visualize_sample_grid(model, dataset, device, num_samples=500, num_examples=4, save_path=None):
    """Visualize samples in a grid layout."""
    model.eval()
    
    # Get random examples
    indices = np.random.choice(len(dataset), num_examples, replace=False)
    
    rows = 2
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.flatten()
    
    stats_all = []
    for idx, ax in zip(indices, axes):
        data = dataset[idx]
        
        map_img = data["map"].unsqueeze(0).to(device)
        start = data["start"].unsqueeze(0).to(device)
        goal = data["goal"].unsqueeze(0).to(device)
        gt_path = data["gt_path"].numpy()
        
        # Sample from model
        with torch.no_grad():
            samples = model.sample(map_img, start, goal, num_samples=num_samples)
        
        samples = samples[0].cpu().numpy()
        start_np = start[0].cpu().numpy()
        goal_np = goal[0].cpu().numpy()
        map_np = map_img[0, 0].cpu().numpy()
        gt_plot = to_plot_xy(gt_path)
        samples_plot = to_plot_xy(samples)
        start_plot = to_plot_xy(start_np)
        goal_plot = to_plot_xy(goal_np)
        stats = sample_collision_stats(samples, map_np)
        stats["example_idx"] = int(idx)
        stats_all.append(stats)
        
        # Plot
        ax.imshow(map_np, cmap='hot', origin='upper', extent=[0, 1, 0, 1], alpha=0.8)
        ax.scatter(gt_plot[:, 0], gt_plot[:, 1], c='green', s=8, alpha=0.5, label='GT Path', zorder=5)
        ax.scatter(samples_plot[:, 0], samples_plot[:, 1], c='blue', s=8, alpha=0.4, label=f'Samples (n={num_samples})', zorder=3)
        ax.add_patch(Circle(start_plot, 0.02, color='red', zorder=10, label='Start'))
        ax.add_patch(Circle(goal_plot, 0.02, color='lime', zorder=10, label='Goal'))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(
            f"Example {idx} | collision {stats['collision_rate']*100:.1f}%",
            fontsize=11,
            fontweight='bold',
        )
        
        if idx == indices[0]:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved sample grid to: {save_path}")
    plt.close()
    return stats_all


def visualize_sample_density(model, dataset, device, num_samples=1000, save_path=None):
    """Visualize sample density with heatmap."""
    model.eval()
    
    # Get one example
    idx = np.random.randint(len(dataset))
    data = dataset[idx]
    
    map_img = data["map"].unsqueeze(0).to(device)
    start = data["start"].unsqueeze(0).to(device)
    goal = data["goal"].unsqueeze(0).to(device)
    gt_path = data["gt_path"].numpy()
    
    # Sample from model
    with torch.no_grad():
        samples = model.sample(map_img, start, goal, num_samples=num_samples)
    
    samples = samples[0].cpu().numpy()
    start_np = start[0].cpu().numpy()
    goal_np = goal[0].cpu().numpy()
    map_np = map_img[0, 0].cpu().numpy()
    gt_plot = to_plot_xy(gt_path)
    samples_plot = to_plot_xy(samples)
    start_plot = to_plot_xy(start_np)
    goal_plot = to_plot_xy(goal_np)
    stats = sample_collision_stats(samples, map_np)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Scatter plot
    ax = axes[0]
    ax.imshow(map_np, cmap='gray_r', origin='upper', extent=[0, 1, 0, 1], alpha=0.5)
    ax.scatter(samples_plot[:, 0], samples_plot[:, 1], c='blue', s=10, alpha=0.5, label=f'Samples (n={num_samples})', zorder=3)
    ax.scatter(gt_plot[:, 0], gt_plot[:, 1], c='green', s=10, alpha=0.9, label='GT Path', zorder=5)
    ax.add_patch(Circle(start_plot, 0.02, color='red', zorder=10, label='Start'))
    ax.add_patch(Circle(goal_plot, 0.02, color='lime', zorder=10, label='Goal'))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title('Sample Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.text(
        0.02,
        0.02,
        f"collision: {stats['collision_rate']*100:.1f}%\n"
        f"obstacle: {stats['obstacle_hit_rate']*100:.1f}%\n"
        f"oob: {stats['oob_rate']*100:.1f}%",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )
    
    # Right: Density heatmap
    ax = axes[1]
    ax.imshow(map_np, cmap='gray', origin='upper', extent=[0, 1, 0, 1], alpha=0.3)
    
    # 2D histogram
    h, xedges, yedges = np.histogram2d(samples_plot[:, 0], samples_plot[:, 1], bins=50, range=[[0, 1], [0, 1]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(h.T, origin='lower', extent=extent, cmap='hot', alpha=0.7, zorder=1)
    
    ax.scatter(gt_plot[:, 0], gt_plot[:, 1], c='green', s=10, alpha=0.5, label='GT Path', zorder=5)
    ax.add_patch(Circle(start_plot, 0.02, color='red', zorder=10))
    ax.add_patch(Circle(goal_plot, 0.02, color='lime', zorder=10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title('Sample Density Heatmap', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Sample Count')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved density heatmap to: {save_path}")
    plt.close()
    return stats


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
    
    # Try to load config from checkpoint
    config = checkpoint.get('config', None)
    if config:
        print("Found config in checkpoint. Overriding model parameters.")
        num_blocks = getattr(config, 'num_blocks', args.num_blocks)
        position_embed_dim = getattr(config, 'position_embed_dim', 128)
        cond_dim = getattr(config, 'cond_dim', args.cond_dim)
        map_embed_dim = getattr(config, 'map_embed_dim', 256)
        hidden_dim = getattr(config, 'hidden_dim', 128)
        s_max = getattr(config, 's_max', 2.0)
        sg_dim = getattr(config, 'sg_dim', 2)
    else:
        num_blocks = args.num_blocks
        position_embed_dim = 128
        cond_dim = args.cond_dim
        map_embed_dim = 256
        hidden_dim = 128
        s_max = 2.0
        sg_dim = 2

    conditioning_mode = infer_conditioning_mode(checkpoint)

    model = ConditionalFlowPlanner(
        num_blocks=num_blocks,
        sg_dim=sg_dim,
        position_embed_dim=position_embed_dim,
        map_embed_dim=map_embed_dim,
        cond_dim=cond_dim,
        hidden_dim=hidden_dim,
        s_max=s_max,
        conditioning_mode=conditioning_mode,
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"✅ Loaded model from epoch {checkpoint['epoch']}")
    val_loss = checkpoint.get('val_loss', checkpoint.get('best_val_loss', 'N/A'))
    print(f"   Val Loss: {val_loss}")
    print(f"   Conditioning mode: {conditioning_mode}")
    
    # Load dataset
    print("Loading validation dataset...")
    dataset = RLNFDataset(split="valid")
    print(f"Dataset size: {len(dataset)}")
    
    # Create output directory
    save_dir = PROJECT_ROOT / "result" / "visualization" / "sampling"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Visualize
    print(f"\nGenerating visualizations with {args.num_samples} samples...")
    
    grid_path = save_dir / f"samples_grid_{timestamp}.png"
    grid_stats = visualize_sample_grid(
        model,
        dataset,
        device,
        num_samples=args.num_samples,
        num_examples=args.num_examples,
        save_path=grid_path,
    )
    
    density_path = save_dir / f"sample_density_{timestamp}.png"
    density_stats = visualize_sample_density(
        model,
        dataset,
        device,
        num_samples=args.num_samples,
        save_path=density_path,
    )
    if grid_stats:
        collision_rates = [s["collision_rate"] for s in grid_stats]
        obstacle_rates = [s["obstacle_hit_rate"] for s in grid_stats]
        oob_rates = [s["oob_rate"] for s in grid_stats]
        print("Grid collision stats:")
        print(f"  mean collision: {np.mean(collision_rates)*100:.2f}%")
        print(f"  mean obstacle:  {np.mean(obstacle_rates)*100:.2f}%")
        print(f"  mean oob:       {np.mean(oob_rates)*100:.2f}%")
    print("Density example collision stats:")
    print(f"  collision: {density_stats['collision_rate']*100:.2f}%")
    print(f"  obstacle:  {density_stats['obstacle_hit_rate']*100:.2f}%")
    print(f"  oob:       {density_stats['oob_rate']*100:.2f}%")
    
    print("\n✅ Sampling visualization complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Flow Model Sampling")
    
    parser.add_argument("--checkpoint", type=str, default="result/checkpoints/best_model.pt",
                        help="Path to checkpoint")
    parser.add_argument("--num_samples", type=int, default=512,
                        help="Number of samples to generate")
    parser.add_argument("--num_examples", type=int, default=50,
                        help="Number of examples in grid")
    parser.add_argument("--num_blocks", type=int, default=4,
                        help="Number of coupling blocks")
    parser.add_argument("--cond_dim", type=int, default=128,
                        help="Condition dimension")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    main(args)
