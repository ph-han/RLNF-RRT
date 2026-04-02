from __future__ import annotations

import torch

from rlnf_rrt.models.flow import Flow
from rlnf_rrt.models.subgoal_policy import SubGoalPolicy


def build_flow_from_ckpt(ckpt: dict, device: torch.device, flow_cfg: dict) -> Flow:
    """Build Flow model from checkpoint dict and freeze all params."""
    cfg = ckpt.get("config")
    if cfg is None or "model" not in cfg:
        raise RuntimeError("Checkpoint missing model config.")

    m = cfg["model"]
    model = Flow(
        num_blocks=int(m["num_blocks"]),
        latent_dim=int(m["latent_dim"]),
        hidden_dim=int(m["hidden_dim"]),
        s_max=float(m["s_max"]),
        backbone=str(flow_cfg.get("backbone", "resnet34")),
        is_pe=bool(flow_cfg.get("is_pe", False)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_subgoal_model(ckpt_path: str, device: torch.device) -> SubGoalPolicy:
    """Load SL-pretrained SubGoalPolicy as KL anchor, freeze all params."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    # pretrain_subgoal.py saves config as the full TOML dict
    policy_cfg = cfg.get("policy", cfg.get("model", {}))

    model = SubGoalPolicy(
        latent_dim=int(policy_cfg.get("latent_dim", 128)),
        hidden_dim=int(policy_cfg.get("hidden_dim", 128)),
        backbone=str(policy_cfg.get("backbone", "resnet34")),
        num_subgoals=1,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
