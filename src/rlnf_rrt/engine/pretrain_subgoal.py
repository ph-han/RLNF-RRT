from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Beta
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from rlnf_rrt.data.dataset import RLNFDataset
from rlnf_rrt.models.subgoal_policy import SubGoalPolicy
from rlnf_rrt.utils.config import load_toml, resolve_project_path
from rlnf_rrt.utils.seed import seed_everything
from rlnf_rrt.utils.utils import get_device


def _midpoint(gt_path: torch.Tensor) -> torch.Tensor:
    """gt_path (B, N, 2) → 중앙점 (B, 2)"""
    N = gt_path.size(1)
    return gt_path[:, N // 2, :]


def _run_epoch(
    model: SubGoalPolicy,
    loader: DataLoader,
    optimizer: Adam | None,
    grad_clip_norm: float,
    device: torch.device,
    log_interval: int,
) -> tuple[float, float]:
    """한 epoch 실행. (avg_nll_loss, avg_l2_dist) 반환."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_l2 = 0.0
    total_count = 0

    pbar = tqdm(loader, desc="train" if is_train else "val", leave=False)
    for i, batch in enumerate(pbar, start=1):
        cond_image = batch["cond_image"].to(device, non_blocking=True)
        start = batch["start"].to(device, non_blocking=True)
        goal = batch["goal"].to(device, non_blocking=True)
        gt_path = batch["gt_path"].to(device, non_blocking=True)

        target = _midpoint(gt_path)  # (B, 2)
        target_clamped = target.clamp(1e-4, 1.0 - 1e-4)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            alpha, beta_param = model(cond_image, start, goal)  # (B, 2) each
            dist = Beta(alpha.clamp(min=0.1), beta_param.clamp(min=0.1))
            loss = -dist.log_prob(target_clamped).mean()

            if is_train:
                loss.backward()
                if grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        with torch.no_grad():
            pred_mean = alpha / (alpha + beta_param)  # Beta distribution mean
            l2 = torch.norm(pred_mean - target, dim=-1).mean().item()

        B = target.size(0)
        total_loss += float(loss.item()) * B
        total_l2 += l2 * B
        total_count += B

        if i % log_interval == 0:
            pbar.set_postfix(nll=f"{loss.item():.4f}", l2=f"{l2:.4f}")

    n = max(total_count, 1)
    return total_loss / n, total_l2 / n


def sl_train(config_path: str | Path = "configs/sl/default.toml") -> None:
    cfg = load_toml(config_path)

    seed = int(cfg["seed"]["value"])
    data_cfg = cfg["data"]
    policy_cfg = cfg["policy"]
    train_cfg = cfg["train"]
    output_cfg = cfg["output"]

    seed_everything(seed)
    device = get_device()

    pin_memory = device.type == "cuda"
    data_root = resolve_project_path(data_cfg.get("data_root", "data"))
    num_points = int(data_cfg["num_points"])
    clearance = int(data_cfg["clearance"])
    step_size = int(data_cfg["step_size"])
    batch_size = int(data_cfg["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 4))

    train_ds = RLNFDataset(
        split="train",
        data_root=data_root,
        num_points=num_points,
        clearance=clearance,
        step_size=step_size,
    )
    val_ds = RLNFDataset(
        split="val",
        data_root=data_root,
        num_points=num_points,
        clearance=clearance,
        step_size=step_size,
    )
    if len(train_ds) == 0:
        raise RuntimeError("Train dataset is empty.")
    if len(val_ds) == 0:
        raise RuntimeError("Val dataset is empty.")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = SubGoalPolicy(
        latent_dim=int(policy_cfg.get("latent_dim", 128)),
        hidden_dim=int(policy_cfg.get("hidden_dim", 128)),
        backbone=str(policy_cfg.get("backbone", "resnet34")),
        num_subgoals=1,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=float(train_cfg["lr"]))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = str(output_cfg.get("run_name", "")).strip() or f"sl_{stamp}"
    output_dir = resolve_project_path(output_cfg.get("checkpoint_root", "outputs/checkpoints")) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "train_config.snapshot.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    epochs = int(train_cfg["epochs"])
    val_interval = int(train_cfg.get("val_interval", 1))
    log_interval = int(train_cfg.get("log_interval", 50))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))

    print(f"device={device}  train={len(train_ds)}  val={len(val_ds)}")
    print(f"target=midpoint(N//2)  checkpoints={output_dir}")

    best_val_nll = float("inf")

    for epoch in range(1, epochs + 1):
        train_nll, train_l2 = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            grad_clip_norm=grad_clip_norm,
            device=device,
            log_interval=log_interval,
        )

        log_str = f"[epoch {epoch:03d}/{epochs:03d}] train_nll={train_nll:.4f}  train_l2={train_l2:.4f}"

        if epoch % val_interval == 0:
            with torch.no_grad():
                val_nll, val_l2 = _run_epoch(
                    model=model,
                    loader=val_loader,
                    optimizer=None,
                    grad_clip_norm=0.0,
                    device=device,
                    log_interval=log_interval,
                )
            log_str += f"  val_nll={val_nll:.4f}  val_l2={val_l2:.4f}"

            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_nll": train_nll,
                "val_nll": val_nll,
                "config": cfg,
            }
            torch.save(ckpt, output_dir / "last.pt")
            if val_nll < best_val_nll:
                best_val_nll = val_nll
                torch.save(ckpt, output_dir / "best.pt")
                log_str += "  ← best"

        print(log_str)

    print(f"done. best_val_nll={best_val_nll:.4f}")
