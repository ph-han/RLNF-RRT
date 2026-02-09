#!/usr/bin/env python
"""
Simplified training script for RLNF-RRT Flow Model.
Quick experimentation and testing.

Usage:
    uv run python scripts/train_flow.py
    uv run python scripts/train_flow.py --epochs 10 --batch_size 8
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rlnf_rrt.data_pipeline.dataset import RLNFDataset
from rlnf_rrt.models.conditional_flow_planner import ConditionalFlowPlanner
from rlnf_rrt.training.loss import FlowNLLLoss, compute_bits_per_dim


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch_idx, batch in enumerate(pbar):
        map_img = batch["map"].to(device)
        start = batch["start"].to(device)
        goal = batch["goal"].to(device)
        gt_path = batch["gt_path"].to(device)
        
        # Forward
        z, log_det = model(gt_path, map_img, start, goal)
        loss = criterion(z, log_det)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Logging
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        # Bits per dim
        num_dims = gt_path.shape[1] * gt_path.shape[2]
        bpd = compute_bits_per_dim(torch.tensor(avg_loss), num_dims).item()
        
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "bpd": f"{bpd:.2f}"})
    
    return total_loss / len(train_loader)


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    
    for batch in tqdm(val_loader, desc="Validation"):
        map_img = batch["map"].to(device)
        start = batch["start"].to(device)
        goal = batch["goal"].to(device)
        gt_path = batch["gt_path"].to(device)
        
        z, log_det = model(gt_path, map_img, start, goal)
        loss = criterion(z, log_det)
        
        total_loss += loss.item()
    
    return total_loss / len(val_loader)


def main(args):
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset
    print("Loading datasets...")
    train_dataset = RLNFDataset(split="train")
    val_dataset = RLNFDataset(split="valid")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Model
    model = ConditionalFlowPlanner(
        num_blocks=args.num_blocks,
        sg_dim=2,
        position_embed_dim=128,
        map_embed_dim=256,
        cond_dim=args.cond_dim,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Loss & Optimizer
    criterion = FlowNLLLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Training loop
    best_val_loss = float("inf")
    result_dir = PROJECT_ROOT / "result" / "checkpoints"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }
            torch.save(checkpoint, result_dir / "best_model.pt")
            print(f"  ✅ Saved best model (val_loss: {val_loss:.4f})")
    
    print("=" * 60)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {result_dir / 'best_model.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RLNF-RRT Flow Model")
    
    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    
    # Model
    parser.add_argument("--num_blocks", type=int, default=4, help="Number of coupling blocks")
    parser.add_argument("--cond_dim", type=int, default=128, help="Condition dimension")
    
    # System
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--num_workers", type=int, default=6, help="Number of workers")
    
    args = parser.parse_args()
    main(args)
