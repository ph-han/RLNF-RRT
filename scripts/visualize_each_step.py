#!/usr/bin/env python
"""
Visualize step-by-step transformations through coupling blocks.

Shows how samples transform from base distribution z0 → z1 → z2 → ... → x_final

Usage:
    uv run python scripts/visualize_each_step.py
    uv run python scripts/visualize_each_step.py --num_examples 3 --num_samples 300
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


def infer_conditioning_mode(checkpoint):
    config = checkpoint.get("config", None)
    if config is not None and hasattr(config, "conditioning_mode"):
        return getattr(config, "conditioning_mode")
    state_dict = checkpoint.get("model_state_dict", {})
    has_film = any(".film1." in k or ".film2." in k for k in state_dict.keys())
    return "film" if has_film else "concat"


def visualize_flow_steps(model, dataset, device, num_samples=300, example_idx=None, save_path=None):
    """Visualize intermediate transformations through each coupling block."""
    model.eval()
    
    # Get example
    if example_idx is None:
        example_idx = np.random.randint(len(dataset))
    
    data = dataset[example_idx]
    
    map_img = data["map"].unsqueeze(0).to(device)
    start = data["start"].unsqueeze(0).to(device)
    goal = data["goal"].unsqueeze(0).to(device)
    gt_path = data["gt_path"].numpy()
    
    # Get intermediate transformations
    with torch.no_grad():
        intermediates = model.sample_with_intermediates(map_img, start, goal, num_samples=num_samples)
    
    # Convert to numpy
    intermediates = [z[0].cpu().numpy() for z in intermediates]  # Each: (num_samples, 2)
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
    else:
        axes = axes.flatten() if rows > 1 else axes
    
    for i, (ax, z) in enumerate(zip(axes[:num_steps], intermediates)):
        # Plot map
        ax.imshow(map_np, cmap='gray', origin='lower', extent=[0, 1, 0, 1], alpha=0.5)
        
        # Plot ground truth on last step only
        if i == num_steps - 1:
            ax.scatter(gt_path[:, 0], gt_path[:, 1], c='g', s=10, alpha=0.9, label='GT Path', zorder=5)
        
        # Plot samples
        color = 'red' if i == 0 else ('blue' if i == num_steps - 1 else 'orange')
        alpha = 0.3 if i == 0 else (0.5 if i == num_steps - 1 else 0.4)
        label = f'z{i}' if i == 0 else (f'x (final)' if i == num_steps - 1 else f'z{i}')
        
        ax.scatter(z[:, 0], z[:, 1], c=color, s=10, alpha=alpha, label=label, zorder=3)
        
        # Plot start/goal
        ax.add_patch(Circle(start_np, 0.02, color='red', zorder=10))
        ax.add_patch(Circle(goal_np, 0.02, color='lime', zorder=10))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        
        if i == 0:
            title = f'Step {i}: Base Distribution (z0)'
        elif i == num_steps - 1:
            title = f'Step {i}: Final Output (x)'
        else:
            title = f'Step {i}: After Block {i}'
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
    
    # Hide unused subplots
    for ax in axes[num_steps:]:
        ax.axis('off')
    
    fig.suptitle(f'Flow Transformation Steps (Example {example_idx})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved flow steps to: {save_path}")
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
    print(f"   Number of coupling blocks: {len(model.flow_model)}")
    print(f"   Conditioning mode: {conditioning_mode}")
    
    # Load dataset
    print("Loading validation dataset...")
    dataset = RLNFDataset(split="valid")
    print(f"Dataset size: {len(dataset)}")
    
    # Create output directory
    save_dir = PROJECT_ROOT / "result" / "visualization" / "each_step"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Visualize multiple examples
    print(f"\nGenerating {args.num_examples} visualizations with {args.num_samples} samples each...")
    
    for i in range(args.num_examples):
        save_path = save_dir / f"flow_steps_{i}_{timestamp}.png"
        visualize_flow_steps(model, dataset, device, 
                            num_samples=args.num_samples,
                            example_idx=None,  # Random
                            save_path=save_path)
    
    print(f"\n✅ Generated {args.num_examples} flow step visualizations!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Flow Step-by-Step Transformations")
    
    parser.add_argument("--checkpoint", type=str, default="result/models/v1_best_model.pt",
                        help="Path to checkpoint")
    parser.add_argument("--num_samples", type=int, default=300,
                        help="Number of samples to generate per example")
    parser.add_argument("--num_examples", type=int, default=50,
                        help="Number of examples to visualize")
    parser.add_argument("--num_blocks", type=int, default=4,
                        help="Number of coupling blocks")
    parser.add_argument("--cond_dim", type=int, default=128,
                        help="Condition dimension")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    main(args)
