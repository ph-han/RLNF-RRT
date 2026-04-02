from __future__ import annotations

import torch
import torch.distributions as D
from rlnf_rrt.utils.types import RolloutModels, RolloutConfig, DynamicRolloutParams


# ---------------------------------------------------------------------------
# Reward helpers
# ---------------------------------------------------------------------------

def compute_subgoal_clearance_reward(
    midpoint: torch.Tensor,
    sdf: torch.Tensor,
    clearance_thresh: float,
) -> float:
    """Evaluate clearance of a single midpoint (2,) against the SDF channel."""
    H, W = sdf.shape
    px = int(torch.clamp(torch.round(midpoint[0] * (W - 1)), 0, W - 1).item())
    py = int(torch.clamp(torch.round(midpoint[1] * (H - 1)), 0, H - 1).item())
    sdf_val = sdf[py, px].item()

    # sdf is normalized by max(H, W), convert thresh to same scale
    norm = float(max(H, W))
    thresh_norm = clearance_thresh / norm

    if sdf_val < 0:
        return -2.0  # obstacle
    if sdf_val < thresh_norm:
        # linear interpolation from -0.5 (sdf=0) to 0.0 (sdf=thresh)
        return -0.5 * (1.0 - sdf_val / thresh_norm)
    return 0.3  # good clearance


def compute_flow_coverage_reward(
    seg_start: torch.Tensor,
    seg_goal: torch.Tensor,
    cond_image: torch.Tensor,
    flow_model: torch.nn.Module,
    device: torch.device,
    num_samples: int,
    pts_per_seg: int,
    clearance_thresh: float,
) -> float:
    """Sample paths from flow model and compute free-space coverage ratio.

    Returns:
        reward in [0.0, 1.0] — fraction of sampled path points with sufficient clearance.
    """
    sdf = cond_image[2]  # (H, W)
    H, W = sdf.shape
    norm = float(max(H, W))
    thresh_norm = clearance_thresh / norm

    # Prepare batched inputs for flow inverse
    cond_batch = cond_image.unsqueeze(0).expand(num_samples, -1, -1, -1)  # (N, 3, H, W)
    start_batch = seg_start.unsqueeze(0).expand(num_samples, -1)  # (N, 2)
    goal_batch = seg_goal.unsqueeze(0).expand(num_samples, -1)  # (N, 2)

    z = torch.randn(num_samples, pts_per_seg, 2, device=device)
    with torch.no_grad():
        pred_paths, _ = flow_model.inverse(cond_batch, start_batch, goal_batch, z)
    pred_paths = pred_paths.clamp(0.0, 1.0)  # (N, T, 2)

    # Evaluate clearance for each point
    px = (pred_paths[..., 0] * (W - 1)).round().long().clamp(0, W - 1)
    py = (pred_paths[..., 1] * (H - 1)).round().long().clamp(0, H - 1)
    sdf_vals = sdf[py, px]  # (N, T)

    good = (sdf_vals >= thresh_norm).float()
    reward = good.mean().item()
    return reward


def compute_segment_reward(
    seg_start: torch.Tensor,
    seg_goal: torch.Tensor,
    cond_image: torch.Tensor,
    flow_model: torch.nn.Module,
    device: torch.device,
    num_samples: int,
    pts_per_seg: int,
    clearance_thresh: float,
) -> dict:
    """Compute segment-level reward from flow coverage."""
    flow_cov = compute_flow_coverage_reward(
        seg_start, seg_goal, cond_image, flow_model, device,
        num_samples, pts_per_seg, clearance_thresh,
    )
    return {"reward": flow_cov, "flow_coverage": flow_cov}


# Alias kept for import compatibility
compute_rrt_segment_reward = compute_segment_reward


# ---------------------------------------------------------------------------
# Recursive rollout (training)
# ---------------------------------------------------------------------------

def _recursive_train(
    seg_start: torch.Tensor,
    seg_goal: torch.Tensor,
    depth: int,
    cond_image: torch.Tensor,
    models: RolloutModels,
    cfg: RolloutConfig,
    dyn: DynamicRolloutParams,
    device: torch.device,
    agent_map_feat: torch.Tensor | None,
    anchor_map_feat: torch.Tensor | None,
    decisions: list[dict],
) -> dict:
    """Recursively split a segment and collect rewards + log-probs."""
    agent = models.split_agent
    clearance_thresh = float(cfg.rrt_cfg.get("clearance", 2))

    # Forward pass
    split_prob, alpha, beta, agent_map_feat = agent(
        cond_image, (seg_start.unsqueeze(0), seg_goal.unsqueeze(0)),
        map_feat_cache=agent_map_feat,
    )
    split_prob = split_prob.squeeze(0)  # scalar
    alpha = alpha.squeeze(0)  # (2,)
    beta_param = beta.squeeze(0)  # (2,)

    # Beta distribution (always compute entropy)
    beta_dist = D.Beta(alpha, beta_param)
    mid_entropy = beta_dist.entropy().sum()

    # KL with anchor model
    kl = torch.tensor(0.0, device=device)
    if models.kl_anchor_model is not None:
        with torch.no_grad():
            if anchor_map_feat is None:
                anchor_alpha, anchor_beta = models.kl_anchor_model(
                    cond_image, seg_start, seg_goal,
                )
            else:
                anchor_alpha, anchor_beta = models.kl_anchor_model(
                    cond_image, seg_start, seg_goal,
                    map_feat_cache=anchor_map_feat,
                )
        anchor_dist = D.Beta(anchor_alpha, anchor_beta)
        kl = D.kl_divergence(beta_dist, anchor_dist).sum()

    # Split decision
    force_no_split = depth >= cfg.max_depth
    if force_no_split:
        do_split = False
        split_lp = torch.tensor(0.0, device=device)
        split_ent = torch.tensor(0.0, device=device)
    else:
        split_dist = D.Bernoulli(probs=split_prob)
        split_ent = split_dist.entropy()

        # Exploration: epsilon-greedy for split
        if dyn.force_split_eps > 0 and torch.rand(1).item() < dyn.force_split_eps:
            do_split = True
        else:
            do_split = bool(split_dist.sample().item())

        split_lp = split_dist.log_prob(torch.tensor(float(do_split), device=device))

    if do_split:
        # Sample midpoint from Beta
        raw_mid = beta_dist.rsample()  # (2,) reparameterized
        mid_lp = beta_dist.log_prob(raw_mid).sum()

        # Interpolate midpoint between start and goal
        midpoint = seg_start + raw_mid * (seg_goal - seg_start)
        midpoint = midpoint.clamp(0.0, 1.0)

        # Clearance reward for the midpoint
        sdf = cond_image[2]
        clr_reward = compute_subgoal_clearance_reward(
            midpoint, sdf, clearance_thresh,
        )

        # Recurse left and right
        left = _recursive_train(
            seg_start, midpoint, depth + 1,
            cond_image, models, cfg, dyn, device,
            agent_map_feat, anchor_map_feat, decisions,
        )
        right = _recursive_train(
            midpoint.detach(), seg_goal, depth + 1,
            cond_image, models, cfg, dyn, device,
            agent_map_feat, anchor_map_feat, decisions,
        )

        total_reward = clr_reward + left["reward"] + right["reward"]

        decisions.append({
            "split_lp": split_lp,
            "mid_lp": mid_lp,
            "split_ent": split_ent,
            "mid_ent": mid_entropy,
            "kl": kl,
        })

        return {"reward": total_reward}
    else:
        # Leaf: evaluate segment via flow coverage
        seg_result = compute_segment_reward(
            seg_start, seg_goal, cond_image, models.flow_model,
            device, cfg.num_samples, cfg.pts_per_seg,
            clearance_thresh,
        )

        decisions.append({
            "split_lp": split_lp,
            "mid_lp": torch.tensor(0.0, device=device),
            "split_ent": split_ent,
            "mid_ent": mid_entropy,
            "kl": kl,
        })

        return {"reward": seg_result["reward"]}


def rollout_subgoal_policy(
    sample: dict,
    models: RolloutModels,
    cfg: RolloutConfig,
    dyn: DynamicRolloutParams,
    device: torch.device,
) -> dict:
    """Entry point for a single training rollout."""
    cond_image = sample["cond_image"].to(device)  # (3, H, W)
    start = sample["start"].to(device)  # (2,)
    goal = sample["goal"].to(device)  # (2,)

    decisions: list[dict] = []

    result = _recursive_train(
        start, goal, depth=0,
        cond_image=cond_image,
        models=models, cfg=cfg, dyn=dyn, device=device,
        agent_map_feat=None, anchor_map_feat=None,
        decisions=decisions,
    )

    # Aggregate
    total_log_prob = sum(d["split_lp"] + d["mid_lp"] for d in decisions)
    total_split_entropy = sum(d["split_ent"] for d in decisions)
    total_mid_entropy = sum(d["mid_ent"] for d in decisions)
    total_kl = sum(d["kl"] for d in decisions)
    num_splits = sum(1 for d in decisions if d["mid_lp"].item() != 0.0)
    num_leaves = len(decisions) - num_splits

    return {
        "total_reward": result["reward"],
        "total_log_prob": total_log_prob,
        "total_split_entropy": total_split_entropy,
        "total_mid_entropy": total_mid_entropy,
        "total_kl": total_kl,
        "num_splits": num_splits,
        "num_leaves": num_leaves,
    }


# ---------------------------------------------------------------------------
# Recursive rollout (validation — greedy, no grad)
# ---------------------------------------------------------------------------

def _recursive_greedy(
    seg_start: torch.Tensor,
    seg_goal: torch.Tensor,
    depth: int,
    cond_image: torch.Tensor,
    models: RolloutModels,
    cfg: RolloutConfig,
    device: torch.device,
    agent_map_feat: torch.Tensor | None,
    goals: list[torch.Tensor],
) -> dict:
    agent = models.split_agent
    clearance_thresh = float(cfg.rrt_cfg.get("clearance", 2))

    split_prob, alpha, beta_param, agent_map_feat = agent(
        cond_image, (seg_start.unsqueeze(0), seg_goal.unsqueeze(0)),
        map_feat_cache=agent_map_feat,
    )
    split_prob = split_prob.squeeze(0)
    alpha = alpha.squeeze(0)
    beta_param = beta_param.squeeze(0)

    do_split = (split_prob.item() > 0.5) and (depth < cfg.max_depth)

    if do_split:
        beta_dist = D.Beta(alpha, beta_param)
        raw_mid = beta_dist.mean  # greedy = mean
        midpoint = seg_start + raw_mid * (seg_goal - seg_start)
        midpoint = midpoint.clamp(0.0, 1.0)

        goals.append(midpoint)

        sdf = cond_image[2]
        clr_reward = compute_subgoal_clearance_reward(
            midpoint, sdf, clearance_thresh,
        )

        left = _recursive_greedy(
            seg_start, midpoint, depth + 1,
            cond_image, models, cfg, device, agent_map_feat, goals,
        )
        right = _recursive_greedy(
            midpoint, seg_goal, depth + 1,
            cond_image, models, cfg, device, agent_map_feat, goals,
        )

        return {
            "reward": clr_reward + left["reward"] + right["reward"],
            "flow_coverage": (left["flow_coverage"] + right["flow_coverage"]) / 2,
            "num_splits": 1 + left["num_splits"] + right["num_splits"],
            "num_leaves": left["num_leaves"] + right["num_leaves"],
        }
    else:
        seg_result = compute_segment_reward(
            seg_start, seg_goal, cond_image, models.flow_model,
            device, cfg.num_samples, cfg.pts_per_seg,
            clearance_thresh,
        )
        return {
            "reward": seg_result["reward"],
            "flow_coverage": seg_result["flow_coverage"],
            "num_splits": 0,
            "num_leaves": 1,
        }


@torch.no_grad()
def rollout_subgoal_policy_validation(
    sample: dict,
    models: RolloutModels,
    cfg: RolloutConfig,
    device: torch.device,
) -> dict:
    """Greedy validation rollout (no grad)."""
    cond_image = sample["cond_image"].to(device)
    start = sample["start"].to(device)
    goal = sample["goal"].to(device)

    goals: list[torch.Tensor] = []

    result = _recursive_greedy(
        start, goal, depth=0,
        cond_image=cond_image,
        models=models, cfg=cfg, device=device,
        agent_map_feat=None, goals=goals,
    )

    return {
        "total_reward": result["reward"],
        "flow_coverage": result["flow_coverage"],
        "num_splits": result["num_splits"],
        "num_leaves": result["num_leaves"],
        "all_goals": goals,
    }
