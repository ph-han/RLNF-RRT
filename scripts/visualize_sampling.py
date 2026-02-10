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


def visualize_sample_grid(model, dataset, device, num_samples=500, num_examples=4, save_path=None):
    """Visualize samples in a grid layout."""
    model.eval()
    
    # Get random examples
    indices = np.random.choice(len(dataset), num_examples, replace=False)
    
    rows = 2
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.flatten()
    
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
        
        # Plot
        im = ax.imshow(map_np, cmap='hot', extent=[0, 1, 0, 1], alpha=0.8)
        ax.scatter(gt_path[:, 0], gt_path[:, 1], c='green', s=8, alpha=0.5, label='GT Path', zorder=5)
        ax.scatter(samples[:, 0], samples[:, 1], c='blue', s=8, alpha=0.4, label=f'Samples (n={num_samples})', zorder=3)
        ax.add_patch(Circle(start_np, 0.02, color='red', zorder=10, label='Start'))
        ax.add_patch(Circle(goal_np, 0.02, color='lime', zorder=10, label='Goal'))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(f'Example {idx}', fontsize=11, fontweight='bold')
        
        if idx == indices[0]:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved sample grid to: {save_path}")
    plt.close()


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
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Scatter plot
    ax = axes[0]
    ax.imshow(map_np, cmap='gray_r', extent=[0, 1, 0, 1], alpha=0.5)
    ax.scatter(samples[:, 0], samples[:, 1], c='blue', s=10, alpha=0.5, label=f'Samples (n={num_samples})', zorder=3)
    ax.scatter(gt_path[:, 0], gt_path[:, 1], c='green', s=10, alpha=0.9, label='GT Path', zorder=5)
    ax.add_patch(Circle(start_np, 0.02, color='red', zorder=10, label='Start'))
    ax.add_patch(Circle(goal_np, 0.02, color='lime', zorder=10, label='Goal'))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title('Sample Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    
    # Right: Density heatmap
    ax = axes[1]
    ax.imshow(map_np, cmap='gray', origin='lower', extent=[0, 1, 0, 1], alpha=0.3)
    
    # 2D histogram
    h, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1], bins=50, range=[[0, 1], [0, 1]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(h.T, origin='lower', extent=extent, cmap='hot', alpha=0.7, zorder=1)
    
    ax.scatter(gt_path[:, 0], gt_path[:, 1], c='green', s=10, alpha=0.5, label='GT Path', zorder=5)
    ax.add_patch(Circle(start_np, 0.02, color='red', zorder=10))
    ax.add_patch(Circle(goal_np, 0.02, color='lime', zorder=10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title('Sample Density Heatmap', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Sample Count')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved density heatmap to: {save_path}")
    plt.close()


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
    else:
        num_blocks = args.num_blocks
        position_embed_dim = 128
        cond_dim = args.cond_dim
        map_embed_dim = 256

    model = ConditionalFlowPlanner(
        num_blocks=num_blocks,
        sg_dim=2,
        position_embed_dim=position_embed_dim,
        map_embed_dim=map_embed_dim,
        cond_dim=cond_dim,
        hidden_dim=128
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"✅ Loaded model from epoch {checkpoint['epoch']}")
    val_loss = checkpoint.get('val_loss', checkpoint.get('best_val_loss', 'N/A'))
    print(f"   Val Loss: {val_loss}")
    
    # Load dataset
    print("Loading validation dataset...")
    dataset = RLNFDataset(split="test")
    print(f"Dataset size: {len(dataset)}")
    
    # Create output directory
    save_dir = PROJECT_ROOT / "result" / "visualization" / "sampling"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Visualize
    print(f"\nGenerating visualizations with {args.num_samples} samples...")
    
    grid_path = save_dir / f"samples_grid_{timestamp}.png"
    visualize_sample_grid(model, dataset, device, num_samples=args.num_samples, 
                          num_examples=args.num_examples, save_path=grid_path)
    
    density_path = save_dir / f"sample_density_{timestamp}.png"
    visualize_sample_density(model, dataset, device, num_samples=args.num_samples, 
                             save_path=density_path)
    
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
