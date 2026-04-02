from __future__ import annotations

import json
import random
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from rlnf_rrt.models.recursive_subgoal_policy import RecursiveSubgoalPolicy, RecursiveSubgoalPolicyConfig

from rlnf_rrt.data.difficulty import precompute_difficulty
from rlnf_rrt.data.dataset import DifficultyBatchSampler, RLNFDataset

from rlnf_rrt.engine.rl_utils import build_flow_from_ckpt, load_subgoal_model
from rlnf_rrt.engine.rl_episode import rollout_subgoal_policy, rollout_subgoal_policy_validation, compute_rrt_segment_reward

from rlnf_rrt.utils.config import load_toml, resolve_project_path
from rlnf_rrt.utils.seed import seed_everything
from rlnf_rrt.utils.utils import get_device
from rlnf_rrt.utils.types import RolloutModels, RolloutConfig, DynamicRolloutParams


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def recurive_subgoal_train(config_path: str | Path = "configs/split/default.toml") -> None:
    cfg = load_toml(config_path)

    seed = int(cfg["seed"]["value"])
    data_cfg = cfg["data"]
    pretrain_flow_cfg = cfg["pretrain-flow"]
    pretrain_sg_cfg = cfg["pretrain-subgoal"]
    agent_cfg = cfg["recursive_subgoal_policy"]
    rl_cfg = cfg["rl"]
    output_cfg = cfg["output"]
    curriculum_cfg = cfg.get("curriculum", {})

    seed_everything(seed)
    device = get_device()

    # --- Dataset ---
    data_root = resolve_project_path(data_cfg.get("data_root", "data"))
    ds = RLNFDataset(
        split=str(data_cfg.get("split", "train")),
        data_root=data_root,
        num_points=int(data_cfg["num_points"]),
        clearance=int(data_cfg["clearance"]),
        step_size=int(data_cfg["step_size"]),
    )
    if len(ds) == 0:
        raise RuntimeError("Dataset is empty.")

    # --- Validation dataset ---
    val_ds = RLNFDataset(
        split="val",
        data_root=data_root,
        num_points=int(data_cfg["num_points"]),
        clearance=int(data_cfg["clearance"]),
        step_size=int(data_cfg["step_size"]),
    )

    # --- Flow model (frozen) ---
    flow_ckpt_path = resolve_project_path(pretrain_flow_cfg["checkpoint"])
    flow_ckpt = torch.load(flow_ckpt_path, map_location=device, weights_only=False)
    flow_model = build_flow_from_ckpt(flow_ckpt, device, pretrain_flow_cfg)

    # --- SubGoal model (SL pretrained) ---
    subgoal_model_ckpt_path = resolve_project_path(pretrain_sg_cfg["checkpoint"])
    subgoal_model = load_subgoal_model(subgoal_model_ckpt_path, device)

    # --- Get agent configs ---
    max_depth = int(agent_cfg.get("max_depth", 4))
    use_complexity_feats = bool(agent_cfg.get("use_complexity_feats", True))

    cfg_obj = RecursiveSubgoalPolicyConfig(
        use_complexity_feats=use_complexity_feats
    )
    
    # --- Optimizer + LR Scheduler ---
    lr = float(rl_cfg.get("lr", 3e-4))
    lr_min = float(rl_cfg.get("lr_min", 3e-5))
    total_episodes = int(rl_cfg.get("total_episodes", 50000))
    subgoal_lr_scale = float(rl_cfg.get("midpoint_lr_scale", 0.1))

    # --- Agent ---
    pretrained_map_encoder = flow_model.cond_encoder.map_encoder
    agent = RecursiveSubgoalPolicy(cfg=cfg_obj, pretrained_encoder=pretrained_map_encoder).to(device)
    rollout_models = RolloutModels(agent, flow_model, kl_anchor_model=subgoal_model)

    # Param groups: subgoal heads에 낮은 LR
    subgoal_params = list(agent.mu_head.parameters()) + list(agent.conc_head.parameters())
    subgoal_param_ids = {id(p) for p in subgoal_params}
    other_params = [p for p in agent.parameters() if p.requires_grad and id(p) not in subgoal_param_ids]
    optimizer = Adam([
        {"params": other_params, "lr": lr},
        {"params": subgoal_params, "lr": lr * subgoal_lr_scale},
    ])

    # --- RL hyperparams ---
    entropy_coef = float(rl_cfg.get("entropy_coef", 0.05))
    entropy_coef_start = float(rl_cfg.get("entropy_coef_start", entropy_coef))
    entropy_warmup_ratio = float(rl_cfg.get("entropy_warmup_ratio", 0.15))
    split_explore_bonus = float(rl_cfg.get("split_explore_bonus", 0.0))
    split_explore_warmup = float(rl_cfg.get("split_explore_warmup", 0.2))
    grad_clip_norm = float(rl_cfg.get("grad_clip_norm", 1.0))
    log_interval = int(rl_cfg.get("log_interval", 50))
    num_flow_samples = int(rl_cfg.get("num_flow_samples", 4))
    pts_per_seg = int(rl_cfg.get("pts_per_seg", 32))
    img_size = int(rl_cfg.get("img_size", 224))

    # Unified model RL hyperparams
    midpoint_loss_coef = float(rl_cfg.get("midpoint_loss_coef", 0.1))
    entropy_coef_mid = float(rl_cfg.get("entropy_coef_mid", 0.001))
    sl_reg_coef_start = float(rl_cfg.get("sl_reg_coef_start", 1.0))
    sl_reg_coef_end = float(rl_cfg.get("sl_reg_coef_end", 0.0))
    clearance_penalty_coef = float(rl_cfg.get("clearance_penalty_coef", 0.3))

    
    batch_size = int(rl_cfg.get("batch_size", 8))  # mini-batch episodes
    num_updates = total_episodes // batch_size
    scheduler = CosineAnnealingLR(optimizer, T_max=num_updates, eta_min=lr_min)
    total_budget = int(rl_cfg.get("total_budget", 1500))
    min_iter = int(rl_cfg.get("min_iter", 200))
    force_split_eps = float(rl_cfg.get("force_split_eps", 0.0))
    complexity_weight = float(rl_cfg.get("complexity_weight", 0.0))
    budget_scale = float(rl_cfg.get("budget_scale", 0.5))

    # --- RRT* setting ---
    rrt_cfg = {**rl_cfg, "clearance": data_cfg["clearance"], "step_size": data_cfg["step_size"]}

    split_cfg = RolloutConfig(
        max_depth, num_flow_samples, pts_per_seg, rrt_cfg, total_budget, min_iter, budget_scale, clearance_penalty_coef, img_size
    )

    # --- Validation indices (고정) ---
    val_interval = int(rl_cfg.get("val_interval", 200))
    val_samples = int(rl_cfg.get("val_samples", 100))
    val_indices = list(range(val_samples))
    print(f"val_indices fixed: {len(val_indices)} samples from val set")

    # --- Output (must be before curriculum for cache path) ---
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = str(output_cfg.get("run_name", "")).strip() or f"split_{stamp}"
    output_dir = resolve_project_path(output_cfg.get("checkpoint_root", "outputs/checkpoints")) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "split_config.snapshot.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # --- Curriculum Calculate ---
    curriculum_enabled = bool(curriculum_cfg.get("enabled", False))
    difficulties: np.ndarray | None = None
    warmup_ratio = float(curriculum_cfg.get("warmup_ratio", 0.1))
    transition_ratio = float(curriculum_cfg.get("transition_ratio", 0.3))
    easy_thresh = float(curriculum_cfg.get("easy_thresh", 0.3))
    hard_thresh = float(curriculum_cfg.get("hard_thresh", 0.7))

    if curriculum_enabled:
        diff_cache_path = output_dir / "difficulties.npy"
        if diff_cache_path.exists():
            difficulties = np.load(diff_cache_path)
            print(f"[curriculum] loaded cached difficulties from {diff_cache_path}")
        else:
            difficulties = precompute_difficulty(ds)
            np.save(diff_cache_path, difficulties)
            print(f"[curriculum] saved difficulties to {diff_cache_path}")

    # --- Batch Sampler ---
    if curriculum_enabled and difficulties is not None:
        batch_sampler = DifficultyBatchSampler(
            difficulties=difficulties,
            batch_size=batch_size,
            total_batches=num_updates,
            easy_thresh=easy_thresh,
            hard_thresh=hard_thresh,
            warmup_ratio=warmup_ratio,
            transition_ratio=transition_ratio,
        )
        print(
            f"[batch_sampler] stratified: easy(<{easy_thresh})={len(batch_sampler.easy_ids)}  "
            f"med=[{easy_thresh},{hard_thresh})={len(batch_sampler.med_ids)}  "
            f"hard(>={hard_thresh})={len(batch_sampler.hard_ids)}"
        )
    else:
        batch_sampler = None

    print(f"device={device}  dataset={len(ds)}  run={run_name}")
    print(f"flow={flow_ckpt_path}  midpoint={subgoal_model_ckpt_path}")
    print(f"mode={'recursive_subgoal'}  max_depth={max_depth}  entropy_coef={entropy_coef}")
    print(f"subgoal_loss_coef={midpoint_loss_coef}  sl_reg={sl_reg_coef_start}→{sl_reg_coef_end}  clr_penalty={clearance_penalty_coef}")
    print(f"reward=episode-level  total_budget={total_budget}  min_iter={min_iter}")
    print(f"batch_size={batch_size}  val_interval={val_interval}  lr={lr}→{lr_min}(cosine)")
    print(f"checkpoints={output_dir}")
    ## --- setting done ---

    # --- Training loop ---
    best_avg_reward = float("-inf")
    best_avg_flow_cov = float("-inf")
    reward_history: list[float] = []
    relative_reward_history: list[float] = []
    loss_history: list[float] = []
    ep_global = 0

    batch_iter = iter(batch_sampler) if batch_sampler is not None else None

    for update_idx in tqdm(range(num_updates), desc="RL updates"):
        progress = update_idx / max(num_updates, 1)

        # --- Schedule coefficients ---
        # entropy warmup
        if progress < entropy_warmup_ratio:
            ent_coef = entropy_coef_start + (entropy_coef - entropy_coef_start) * (progress / entropy_warmup_ratio)
        else:
            ent_coef = entropy_coef

        # SL reg decay (linear)
        sl_reg = sl_reg_coef_start + (sl_reg_coef_end - sl_reg_coef_start) * progress

        # force_split_eps decay over first half
        cur_force_split_eps = force_split_eps * max(1.0 - 2.0 * progress, 0.0)

        # split_explore_factor decay after warmup
        if progress < split_explore_warmup:
            cur_split_explore = split_explore_bonus
        else:
            cur_split_explore = split_explore_bonus * max(1.0 - (progress - split_explore_warmup) / (1.0 - split_explore_warmup), 0.0)

        dyn = DynamicRolloutParams(
            baseline_reward=0.0,
            baseline_is_goal=False,
            force_split_eps=cur_force_split_eps,
            complexity_weight=complexity_weight,
            split_explore_factor=cur_split_explore,
        )

        # --- Get batch indices ---
        if batch_iter is not None:
            try:
                batch_indices = next(batch_iter)
            except StopIteration:
                batch_iter = iter(batch_sampler)
                batch_indices = next(batch_iter)
        else:
            batch_indices = random.sample(range(len(ds)), min(batch_size, len(ds)))

        # --- Rollout batch ---
        batch_rewards = []
        batch_log_probs = []
        batch_split_ent = []
        batch_mid_ent = []
        batch_kl = []

        agent.train()
        for idx in batch_indices:
            sample = ds[idx]
            result = rollout_subgoal_policy(sample, rollout_models, split_cfg, dyn, device)
            batch_rewards.append(result["total_reward"])
            batch_log_probs.append(result["total_log_prob"])
            batch_split_ent.append(result["total_split_entropy"])
            batch_mid_ent.append(result["total_mid_entropy"])
            batch_kl.append(result["total_kl"])
            ep_global += 1

        # --- REINFORCE loss ---
        rewards_t = torch.tensor(batch_rewards, device=device, dtype=torch.float32)
        advantages = rewards_t - rewards_t.mean()

        log_probs = torch.stack(batch_log_probs)
        split_ents = torch.stack(batch_split_ent)
        mid_ents = torch.stack(batch_mid_ent)
        kls = torch.stack(batch_kl)

        reinforce_loss = -(log_probs * advantages.detach()).mean()
        entropy_loss = -(ent_coef * split_ents.mean() + entropy_coef_mid * mid_ents.mean())
        kl_loss = sl_reg * kls.mean()
        total_loss = reinforce_loss + entropy_loss + kl_loss

        # --- Optimize ---
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), grad_clip_norm)
        optimizer.step()
        scheduler.step()

        # --- Logging ---
        avg_reward = rewards_t.mean().item()
        reward_history.append(avg_reward)
        loss_history.append(total_loss.item())

        if (update_idx + 1) % log_interval == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"[{update_idx+1}/{num_updates}] ep={ep_global}  "
                f"R={avg_reward:.4f}  loss={total_loss.item():.4f}  "
                f"ent_s={split_ents.mean().item():.4f}  ent_m={mid_ents.mean().item():.4f}  "
                f"kl={kls.mean().item():.4f}  sl_reg={sl_reg:.4f}  "
                f"lr={lr_now:.2e}  eps={cur_force_split_eps:.3f}"
            )

        # --- Validation ---
        if (update_idx + 1) % val_interval == 0:
            agent.eval()
            val_rewards = []
            val_flow_covs = []
            val_splits = []

            for vi in val_indices:
                if vi >= len(val_ds):
                    break
                val_sample = val_ds[vi]
                val_result = rollout_subgoal_policy_validation(
                    val_sample, rollout_models, split_cfg, device,
                )
                val_rewards.append(val_result["total_reward"])
                val_flow_covs.append(val_result["flow_coverage"])
                val_splits.append(val_result["num_splits"])

            avg_val_reward = np.mean(val_rewards)
            avg_val_flow_cov = np.mean(val_flow_covs)
            avg_val_splits = np.mean(val_splits)

            print(
                f"  [VAL] reward={avg_val_reward:.4f}  flow_cov={avg_val_flow_cov:.4f}  "
                f"avg_splits={avg_val_splits:.2f}"
            )

            # Save checkpoints
            ckpt = {
                "update_idx": update_idx,
                "ep_global": ep_global,
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
                "val_reward": avg_val_reward,
                "val_flow_coverage": avg_val_flow_cov,
            }
            torch.save(ckpt, output_dir / "latest.pt")

            if avg_val_reward > best_avg_reward:
                best_avg_reward = avg_val_reward
                torch.save(ckpt, output_dir / "best.pt")
                print(f"  -> new best reward: {best_avg_reward:.4f}")

            if avg_val_flow_cov > best_avg_flow_cov:
                best_avg_flow_cov = avg_val_flow_cov
                torch.save(ckpt, output_dir / "best_flow_cov.pt")
                print(f"  -> new best flow_cov: {best_avg_flow_cov:.4f}")

            agent.train()

        # --- Training curve plot (periodic) ---
        if (update_idx + 1) % (val_interval * 5) == 0:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].plot(reward_history)
                axes[0].set_title("Reward per update")
                axes[0].set_xlabel("Update")
                axes[1].plot(loss_history)
                axes[1].set_title("Loss per update")
                axes[1].set_xlabel("Update")
                plt.tight_layout()
                fig.savefig(output_dir / "training_curve.png", dpi=120)
                plt.close(fig)
            except Exception:
                pass

    print(f"Training done. best_reward={best_avg_reward:.4f}  best_flow_cov={best_avg_flow_cov:.4f}")
    print(f"Checkpoints: {output_dir}")
