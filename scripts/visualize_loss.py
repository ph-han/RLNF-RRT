#!/usr/bin/env python
"""
Visualize training and validation loss curves.

Usage:
    uv run python scripts/visualize_loss.py
    uv run python scripts/visualize_loss.py --checkpoint result/checkpoints/best_model.pt
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import matplotlib.pyplot as plt
import json


def load_loss_history(checkpoint_path):
    """Load loss history from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Check if checkpoint has loss history
    if "loss_history" in checkpoint:
        return checkpoint["loss_history"]
    
    # Otherwise, return current epoch info only
    return {
        "train_loss": [checkpoint.get("train_loss", 0)],
        "val_loss": [checkpoint.get("val_loss", 0)],
        "epochs": [checkpoint.get("epoch", 1)]
    }


def plot_loss_curves(loss_history, save_path):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = loss_history.get("epochs", list(range(1, len(loss_history["train_loss"]) + 1)))
    train_loss = loss_history["train_loss"]
    val_loss = loss_history["val_loss"]
    
    ax.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    
    # Mark best validation loss
    if val_loss:
        best_val_idx = val_loss.index(min(val_loss))
        best_val_epoch = epochs[best_val_idx]
        best_val_loss = val_loss[best_val_idx]
        ax.scatter([best_val_epoch], [best_val_loss], color='red', s=100, zorder=10, 
                   marker='*', label=f'Best Val Loss: {best_val_loss:.4f}')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (NLL)', fontsize=12)
    ax.set_title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved loss curve to: {save_path}")
    plt.close()


def main(args):
    # Load checkpoint
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("   Train a model first using: uv run python scripts/train_flow.py")
        return
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    loss_history = load_loss_history(checkpoint_path)
    
    # Create output directory
    save_dir = PROJECT_ROOT / "result" / "visualization" / "loss"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_dir / f"loss_curve_{timestamp}.png"
    
    # Plot
    print("Generating loss curve...")
    plot_loss_curves(loss_history, save_path)
    
    # Print summary
    print("\n📊 Loss Summary:")
    print(f"   Final Train Loss: {loss_history['train_loss'][-1]:.4f}")
    print(f"   Final Val Loss: {loss_history['val_loss'][-1]:.4f}")
    if loss_history['val_loss']:
        print(f"   Best Val Loss: {min(loss_history['val_loss']):.4f}")
    
    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Loss Curves")
    
    parser.add_argument("--checkpoint", type=str, default="result/checkpoints/best_model.pt",
                        help="Path to checkpoint")
    
    args = parser.parse_args()
    main(args)
