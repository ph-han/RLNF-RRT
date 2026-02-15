from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rlnf_rrt.data.dataset import RLNFDataset
from rlnf_rrt.models.flow import Flow
from rlnf_rrt.utils.config import load_toml, resolve_project_path
from rlnf_rrt.utils.utils import get_device


def _nll_loss(z: torch.Tensor, log_det: torch.Tensor) -> torch.Tensor:
    log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=z.device, dtype=z.dtype))
    log_pz = -0.5 * (z.pow(2) + log_2pi).sum(dim=(1, 2))
    log_px = log_pz + log_det
    return -log_px.mean()


def _build_model_from_ckpt(ckpt: dict, device: torch.device) -> Flow:
    cfg = ckpt.get("config")
    if cfg is None or "model" not in cfg:
        raise RuntimeError("Checkpoint missing model config. Re-train with current train.py.")

    m = cfg["model"]
    model = Flow(
        num_blocks=int(m["num_blocks"]),
        latent_dim=int(m["latent_dim"]),
        hidden_dim=int(m["hidden_dim"]),
        s_max=float(m["s_max"]),
        channels=tuple(int(x) for x in m["channels"]),
        norm=str(m["norm"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def evaluate(config_path: str | Path = "configs/eval/default.toml") -> None:
    cfg = load_toml(config_path)
    data_cfg = cfg["data"]
    eval_cfg = cfg["eval"]

    ckpt_path = resolve_project_path(eval_cfg["checkpoint"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = get_device()
    pin_memory = bool(eval_cfg.get("pin_memory", True)) and device.type == "cuda"

    ckpt = torch.load(ckpt_path, map_location=device)
    model = _build_model_from_ckpt(ckpt, device)

    split = str(data_cfg.get("split", "test"))
    ds = RLNFDataset(
        split=split,
        data_root=resolve_project_path(data_cfg.get("data_root", "data")),
        noise_std=0.0,
        num_points=int(data_cfg["num_points"]),
        clearance=int(data_cfg["clearance"]),
        step_size=int(data_cfg["step_size"]),
    )
    if len(ds) == 0:
        raise RuntimeError(f"{split} dataset is empty. Check config[data] filters.")

    loader = DataLoader(
        ds,
        batch_size=int(eval_cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(eval_cfg.get("num_workers", 4)),
        pin_memory=pin_memory,
        drop_last=False,
    )

    total_loss = 0.0
    total_count = 0

    save_predictions = bool(eval_cfg.get("save_predictions", True))
    num_samples = int(eval_cfg.get("num_samples_per_condition", 1))
    pred_root = resolve_project_path(eval_cfg.get("prediction_dir", "outputs/predictions"))
    pred_dir: Path | None = None
    if save_predictions:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pred_dir = pred_root / f"{ckpt_path.stem}_{split}_{stamp}"
        pred_dir.mkdir(parents=True, exist_ok=True)

    dataset_offset = 0
    pbar = tqdm(loader, desc=f"eval:{split}", leave=False)
    with torch.no_grad():
        for batch in pbar:
            map_img = batch["map"].to(device, non_blocking=True)
            start = batch["start"].to(device, non_blocking=True)
            goal = batch["goal"].to(device, non_blocking=True)
            gt_path = batch["gt_path"].to(device, non_blocking=True)

            z_gt, log_det = model(map_img, start, goal, gt_path)
            loss = _nll_loss(z_gt, log_det)

            bsz = gt_path.size(0)
            total_loss += float(loss.item()) * bsz
            total_count += bsz
            pbar.set_postfix(nll=f"{loss.item():.4f}")

            if not save_predictions or pred_dir is None:
                dataset_offset += bsz
                continue

            T = gt_path.size(1)
            for s in range(num_samples):
                z = torch.randn((bsz, T, 2), device=device, dtype=gt_path.dtype)
                pred_path, _ = model.inverse(map_img, start, goal, z)
                pred_path = pred_path.clamp(0.0, 1.0).cpu().numpy().astype(np.float32)

                for i in range(bsz):
                    ds_idx = dataset_offset + i
                    np.save(pred_dir / f"pred_{ds_idx:07d}_s{s:02d}.npy", pred_path[i])
            dataset_offset += bsz

    mean_nll = total_loss / max(total_count, 1)
    print(f"device={device}  split={split}  samples={len(ds)}")
    print(f"checkpoint={ckpt_path}")
    print(f"mean_nll={mean_nll:.6f}")
    if pred_dir is not None:
        manifest = {
            "checkpoint": str(ckpt_path),
            "split": split,
            "dataset_size": len(ds),
            "num_samples_per_condition": num_samples,
            "config": cfg,
        }
        with open(pred_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"saved_predictions={pred_dir}")
