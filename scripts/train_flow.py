#!/usr/bin/env python
"""
Simplified training script for RLNF-RRT Flow Model.
Using centralized TrainConfig for hyperparameter management.

Usage:
    uv run python scripts/train_flow.py --version v4
"""
import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rlnf_rrt.data_pipeline.dataset import RLNFDataset
from rlnf_rrt.models.conditional_flow_planner import ConditionalFlowPlanner
from rlnf_rrt.training.loss import FlowNLLLoss, compute_bits_per_dim
from rlnf_rrt.training.config import TrainConfig


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, grad_clip):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
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


def main():
    # 1. Parse Version Only
    parser = argparse.ArgumentParser(description="Train RLNF-RRT Flow Model")
    parser.add_argument("--version", type=str, required=True, help="Experiment version (e.g. v4)")
    args = parser.parse_args()
    
    # 2. Load Config from src/rlnf_rrt/training/config.py
    config = TrainConfig()
    print(f"🚀 Starting Training Version: {args.version}")
    print(f"📌 Hyperparameters loaded from TrainConfig:\n{config}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 3. Dataset
    print("Loading datasets...")
    train_dataset = RLNFDataset(split="train", noise_std=config.noise_std)
    val_dataset = RLNFDataset(split="valid", noise_std=0.0)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # 4. Model
    model = ConditionalFlowPlanner(
        num_blocks=config.num_blocks,
        sg_dim=config.sg_dim,
        map_embed_dim=config.map_embed_dim,
        hidden_dim=config.hidden_dim,
        s_max=config.s_max,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # 5. Loss & Optimizer
    criterion = FlowNLLLoss()
    
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=config.betas
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.T_max,
        eta_min=config.min_lr
    )
    
    # 6. Training Loop Setup
    result_dir = PROJECT_ROOT / "result"
    checkpoint_dir = result_dir / "checkpoints"
    best_model_path = result_dir / "models"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path.mkdir(parents=True, exist_ok=True)
    
    loss_history = {
        "train_loss": [],
        "val_loss": [],
        "epochs": [],
    }
    
    best_val_loss = float("inf")
    
    print(f"\nStarting training for {config.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(config.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, config.grad_clip)
        
        # Validate
        if (epoch + 1) % config.val_interval == 0:
            val_loss = validate(model, val_loader, criterion, device)
            
            # Update Scheduler
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Logging
            loss_history["train_loss"].append(train_loss)
            loss_history["val_loss"].append(val_loss)
            loss_history["epochs"].append(epoch + 1)
            
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")
            
            # Checkpoint (Periodic)
            if (epoch + 1) % config.log_interval == 0:
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "loss_history": loss_history,
                }
                torch.save(checkpoint, checkpoint_dir / f"{args.version}_model_{epoch+1}.pt")
                print(f"  ✅ Saved checkpoint: {args.version}_model_{epoch+1}.pt")
            
            # Best Model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "loss_history": loss_history,
                    "best_val_loss": best_val_loss
                }
                torch.save(checkpoint, best_model_path / f"{args.version}_best_model.pt")
                print(f"  🏆 Saved BEST model (Val Loss: {val_loss:.4f})")
        else:
            # Just training log if skipping validation
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f}")
            scheduler.step()

    print("=" * 60)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
