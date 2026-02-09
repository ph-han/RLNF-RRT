"""
Main training script for RLNF-RRT Normalizing Flow model.

Usage:
    uv run python -m rlnf_rrt.training.train
    uv run python -m rlnf_rrt.training.train --epochs 50 --batch_size 16
"""
import argparse
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rlnf_rrt.data_pipeline.dataset import RLNFDataset
from rlnf_rrt.models.conditional_flow_planner import ConditionalFlowPlanner
from rlnf_rrt.training.config import TrainConfig
from rlnf_rrt.training.loss import FlowNLLLoss, compute_bits_per_dim


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr_scheduler(optimizer: torch.optim.Optimizer, config: TrainConfig, num_training_steps: int):
    """Create learning rate scheduler."""
    warmup_steps = config.warmup_epochs * (num_training_steps // config.epochs)
    
    if config.scheduler == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.min_lr)
    elif config.scheduler == "step":
        from torch.optim.lr_scheduler import StepLR
        return StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: FlowNLLLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: TrainConfig,
    writer: SummaryWriter | None = None,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        map_img = batch["map"].to(device)
        start = batch["start"].to(device)
        goal = batch["goal"].to(device)
        gt_path = batch["gt_path"].to(device)
        
        # Forward pass
        z, log_det = model(gt_path, map_img, start, goal)
        loss = criterion(z, log_det)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        optimizer.step()
        
        # Logging
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        # Compute bits per dim for interpretability
        num_dims = gt_path.shape[1] * gt_path.shape[2]  # T * D
        bpd = compute_bits_per_dim(torch.tensor(avg_loss), num_dims).item()
        
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "bpd": f"{bpd:.2f}"})
        
        # Tensorboard logging
        if writer is not None and batch_idx % config.log_interval == 0:
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/bpd", bpd, global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
    
    return total_loss / num_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: FlowNLLLoss,
    device: torch.device,
    epoch: int,
    config: TrainConfig,
) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Valid]")
    for batch in pbar:
        map_img = batch["map"].to(device)
        start = batch["start"].to(device)
        goal = batch["goal"].to(device)
        gt_path = batch["gt_path"].to(device)
        
        z, log_det = model(gt_path, map_img, start, goal)
        loss = criterion(z, log_det)
        
        total_loss += loss.item()
    
    return total_loss / len(val_loader)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    config: TrainConfig,
    is_best: bool = False,
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": config,
    }
    
    # Save latest checkpoint
    latest_path = config.checkpoint_dir / "latest.pt"
    torch.save(checkpoint, latest_path)
    
    # Save periodic checkpoint
    if (epoch + 1) % config.save_interval == 0:
        epoch_path = config.checkpoint_dir / f"epoch_{epoch+1:03d}.pt"
        torch.save(checkpoint, epoch_path)
    
    # Save best checkpoint
    if is_best:
        best_path = config.checkpoint_dir / "best.pt"
        torch.save(checkpoint, best_path)


def main(config: TrainConfig | None = None) -> None:
    """Main training function."""
    if config is None:
        config = TrainConfig()
    
    # Setup
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Tensorboard writer
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=config.log_dir / run_name)
    
    # Dataset & DataLoader
    print("Loading datasets...")
    train_dataset = RLNFDataset(split="train")
    val_dataset = RLNFDataset(split="valid")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Model
    model = ConditionalFlowPlanner(
        num_blocks=config.num_blocks,
        sg_dim=config.sg_dim,
        position_embed_dim=config.position_embed_dim,
        map_embed_dim=config.map_embed_dim,
        cond_dim=config.cond_dim,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Loss, Optimizer, Scheduler
    criterion = FlowNLLLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    scheduler = get_lr_scheduler(optimizer, config, len(train_loader) * config.epochs)
    
    # Training loop
    best_val_loss = float("inf")
    
    print(f"\nStarting training for {config.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(config.epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config, writer
        )
        
        # Validate
        if (epoch + 1) % config.val_interval == 0:
            val_loss = validate(model, val_loader, criterion, device, epoch, config)
            
            # Log validation
            writer.add_scalar("val/loss", val_loss, epoch)
            
            # Check if best
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            # Save checkpoint
            save_checkpoint(model, optimizer, epoch, val_loss, config, is_best)
            
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Best: {best_val_loss:.4f} | Time: {elapsed:.1f}s")
        else:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Time: {elapsed:.1f}s")
        
        # Step scheduler
        scheduler.step()
    
    print("=" * 60)
    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {config.checkpoint_dir}")
    
    writer.close()


def parse_args() -> TrainConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RLNF-RRT Flow Model")
    
    # Override config defaults via command line
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--cond_dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    return TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        num_blocks=args.num_blocks,
        cond_dim=args.cond_dim,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    config = parse_args()
    main(config)
