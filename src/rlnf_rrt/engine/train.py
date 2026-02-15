from __future__ import annotations

import math
import json
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from rlnf_rrt.data.dataset import RLNFDataset
from rlnf_rrt.models.flow import Flow
from rlnf_rrt.utils.config import load_toml, resolve_project_path
from rlnf_rrt.utils.seed import seed_everything
from rlnf_rrt.utils.utils import get_device
from rlnf_rrt.utils.loss import _nll_loss


def _run_epoch(
    model: Flow,
    loader: DataLoader,
    optimizer: AdamW | None,
    grad_clip_norm: float,
    device: torch.device,
    log_interval: int,
    max_steps: int | None = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0

    pbar = tqdm(loader, desc="train" if is_train else "val", leave=False)
    for i, batch in enumerate(pbar, start=1):
        if max_steps is not None and i > max_steps:
            break

        map_img = batch["map"].to(device, non_blocking=True)
        start = batch["start"].to(device, non_blocking=True)
        goal = batch["goal"].to(device, non_blocking=True)
        gt_path = batch["gt_path"].to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            z, log_det = model(map_img, start, goal, gt_path)
            loss = _nll_loss(z, log_det)

            if is_train:
                loss.backward()
                if grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        batch_size = gt_path.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

        if i % log_interval == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_count, 1)


def train(config_path: str | Path = "configs/train/default.toml") -> None:
    cfg = load_toml(config_path)

    seed = int(cfg["seed"]["value"])
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    output_cfg = cfg["output"]

    seed_everything(seed)
    device = get_device()

    pin_memory = bool(train_cfg.get("pin_memory", True)) and device.type == "cuda"
    data_root = resolve_project_path(data_cfg.get("data_root", "data"))

    train_ds = RLNFDataset(
        split="train",
        data_root=data_root,
        noise_std=float(data_cfg.get("noise_std", 0.0)),
        num_points=int(data_cfg["num_points"]),
        clearance=int(data_cfg["clearance"]),
        step_size=int(data_cfg["step_size"]),
    )
    val_ds = RLNFDataset(
        split="val",
        data_root=data_root,
        noise_std=0.0,
        num_points=int(data_cfg["num_points"]),
        clearance=int(data_cfg["clearance"]),
        step_size=int(data_cfg["step_size"]),
    )
    if len(train_ds) == 0:
        raise RuntimeError("Train dataset is empty. Check config[data] filters.")
    if len(val_ds) == 0:
        raise RuntimeError("Val dataset is empty. Check config[data] filters.")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = Flow(
        num_blocks=int(model_cfg["num_blocks"]),
        latent_dim=int(model_cfg["latent_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        s_max=float(model_cfg["s_max"]),
        channels=tuple(int(x) for x in model_cfg["channels"]),
        norm=str(model_cfg["norm"]),
    ).to(device)
    
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = str(output_cfg.get("run_name", "")).strip() or f"flow_{stamp}"
    output_root = resolve_project_path(output_cfg.get("checkpoint_root", "outputs/checkpoints"))
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "train_config.snapshot.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    best_val = float("inf")
    epochs = int(train_cfg["epochs"])
    log_interval = int(train_cfg.get("log_interval", 50))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
    max_steps_per_epoch = train_cfg.get("max_steps_per_epoch")
    max_val_steps = train_cfg.get("max_val_steps")
    save_last_every_epoch = bool(output_cfg.get("save_last_every_epoch", True))

    print(f"device={device}  train={len(train_ds)}  val={len(val_ds)}")
    print(f"checkpoints={output_dir}")

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            grad_clip_norm=grad_clip_norm,
            device=device,
            log_interval=log_interval,
            max_steps=(int(max_steps_per_epoch) if max_steps_per_epoch is not None else None),
        )

        with torch.no_grad():
            val_loss = _run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                grad_clip_norm=0.0,
                device=device,
                log_interval=log_interval,
                max_steps=(int(max_val_steps) if max_val_steps is not None else None),
            )

        print(f"[epoch {epoch:03d}/{epochs:03d}] train_nll={train_loss:.6f} val_nll={val_loss:.6f}")

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": cfg,
        }
        if save_last_every_epoch:
            torch.save(ckpt, output_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, output_dir / "best.pt")

    print(f"training done. best_val_nll={best_val:.6f}")
