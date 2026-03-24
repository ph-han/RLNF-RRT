from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Beta, Categorical
from torch.optim import Adam
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import BoundedContinuous, Unbounded, Composite
import torch.nn.functional as F

from rlnf_rrt.data.dataset import RLNFDataset
from rlnf_rrt.models.flow import Flow
from rlnf_rrt.models.subgoal_policy import AutoregressiveSubGoalPolicyCount
from rlnf_rrt.utils.config import load_toml, resolve_project_path
from rlnf_rrt.utils.seed import seed_everything
from rlnf_rrt.utils.utils import get_device


# ---------------------------------------------------------------------------
# Reward helpers  (rl_train_count.py에서 그대로 재사용)
# ---------------------------------------------------------------------------

def _make_seg_cond(
    cond_base: torch.Tensor,  # (3, H, W)
    seg_start: torch.Tensor,  # (2,)
    seg_goal: torch.Tensor,   # (2,)
) -> torch.Tensor:            # (1, 3, H, W)
    H, W = cond_base.shape[-2:]
    cond = cond_base.clone()
    cond[1] = 0.0
    sy = int(round(float(seg_start[1].item()) * (H - 1)))
    sx = int(round(float(seg_start[0].item()) * (W - 1)))
    gy = int(round(float(seg_goal[1].item()) * (H - 1)))
    gx = int(round(float(seg_goal[0].item()) * (W - 1)))
    sy, sx = max(0, min(H - 1, sy)), max(0, min(W - 1, sx))
    gy, gx = max(0, min(H - 1, gy)), max(0, min(W - 1, gx))
    cond[1, sy, sx] = 1.0
    cond[1, gy, gx] = -1.0
    return cond.unsqueeze(0)


def _make_gt_score_map(
    gt_path: torch.Tensor,
    H: int,
    W: int,
    obstacle_mask: np.ndarray = None,
    decay_rate: float = 0.01,
) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    pts = gt_path.cpu().numpy()
    px = np.clip(np.round(pts[:, 0] * (W - 1)).astype(int), 0, W - 1)
    py = np.clip(np.round(pts[:, 1] * (H - 1)).astype(int), 0, H - 1)
    binary_map = np.ones((H, W), dtype=bool)
    binary_map[py, px] = False
    dist_to_path = distance_transform_edt(binary_map)
    score_map = 5.0 - (dist_to_path * decay_rate)
    score_map = np.maximum(score_map, -20.0)
    if obstacle_mask is not None:
        score_map[obstacle_mask > 0] = -25.0
    return score_map.astype(np.float32)


@torch.no_grad()
def _seg_gt_score(
    flow_model: Flow,
    cond_image: torch.Tensor,
    seg_start: torch.Tensor,
    seg_goal: torch.Tensor,
    gt_score_map: np.ndarray,
    num_samples: int,
    pts_per_seg: int,
    device: torch.device,
) -> float:
    seg_cond = _make_seg_cond(cond_image, seg_start, seg_goal)
    H, W = gt_score_map.shape
    b_cond = seg_cond.expand(num_samples, -1, -1, -1)
    b_s = seg_start.unsqueeze(0).expand(num_samples, -1)
    b_g = seg_goal.unsqueeze(0).expand(num_samples, -1)
    z = torch.randn(num_samples, pts_per_seg, 2, device=device, dtype=cond_image.dtype)
    pred, _ = flow_model.inverse(b_cond, b_s, b_g, z)
    path = pred.clamp(0.0, 1.0)
    px = (path[..., 0] * (W - 1)).long().clamp(0, W - 1).cpu().numpy()
    py = (path[..., 1] * (H - 1)).long().clamp(0, H - 1).cpu().numpy()
    scores = gt_score_map[py, px]
    return float(scores.sum()) / max(num_samples * pts_per_seg, 1)


@torch.no_grad()
def _compute_reward(
    flow_model: Flow,
    cond_image: torch.Tensor,   # (3, H, W)
    start: torch.Tensor,        # (2,)
    sub_goals: torch.Tensor,    # (K, 2), K=0 가능
    goal: torch.Tensor,         # (2,)
    gt_path: torch.Tensor,
    device: torch.device,
    num_samples: int,
    pts_per_seg: int,
    w_path: float,
    w_gt: float,
    w_efficiency: float,
    max_subgoals: int,
    gt_score_decay_rate: float = 0.01,
) -> float:
    H, W = cond_image.shape[-2:]
    K = len(sub_goals)

    obstacle_mask = (cond_image[0] <= 0.5).cpu().numpy().astype(np.uint8)
    gt_score_map = _make_gt_score_map(
        gt_path, H, W, obstacle_mask=obstacle_mask, decay_rate=gt_score_decay_rate
    )

    waypoints = [start] + [sub_goals[i] for i in range(K)] + [goal]

    if K > 0:
        on_the_gt = float(np.mean([
            gt_score_map[
                max(0, min(H - 1, int(round(float(sg[1]) * (H - 1))))),
                max(0, min(W - 1, int(round(float(sg[0]) * (W - 1)))))
            ]
            for sg in sub_goals
        ]))
    else:
        on_the_gt = 0.0

    seg_scores = [
        _seg_gt_score(flow_model, cond_image, waypoints[i], waypoints[i + 1],
                      gt_score_map, num_samples, pts_per_seg, device)
        for i in range(len(waypoints) - 1)
    ]
    path_reward = float(np.mean(seg_scores))

    efficiency_bonus = (max_subgoals - K) * w_efficiency

    return w_path * path_reward + w_gt * on_the_gt + efficiency_bonus


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SubGoalEnv(EnvBase):
    """1-step MDP for AR variable-count sub-goal selection."""

    def __init__(
        self,
        dataset: RLNFDataset,
        flow_model: Flow,
        device: torch.device,
        num_samples: int,
        pts_per_seg: int,
        img_size: int,
        max_subgoals: int,
        w_path: float = 1.0,
        w_gt: float = 1.5,
        w_efficiency: float = 0.0,
        gt_score_decay_rate: float = 1.0,
        **kwargs,
    ):
        super().__init__(device=device, batch_size=[], **kwargs)
        self.dataset = dataset
        self.flow_model = flow_model
        self.num_samples = num_samples
        self.pts_per_seg = pts_per_seg
        self.img_size = img_size
        self.max_subgoals = max_subgoals
        self.w_path = w_path
        self.w_gt = w_gt
        self.w_efficiency = w_efficiency
        self.gt_score_decay_rate = gt_score_decay_rate
        self._current_gt_path: torch.Tensor | None = None
        self._make_specs()

    def _make_specs(self) -> None:
        H = W = self.img_size
        self.observation_spec = Composite(
            cond_image=BoundedContinuous(low=-1.0, high=1.0, shape=(3, H, W), device=self.device),
            start=BoundedContinuous(low=0.0, high=1.0, shape=(2,), device=self.device),
            goal=BoundedContinuous(low=0.0, high=1.0, shape=(2,), device=self.device),
            shape=(),
            device=self.device,
        )
        # action: flat (2*max_subgoals,) zero-padded, k: actual count
        self.action_spec = BoundedContinuous(
            low=0.0, high=1.0, shape=(2 * self.max_subgoals,), device=self.device
        )
        self.reward_spec = Unbounded(shape=(1,), device=self.device)
        self.done_spec = Composite(
            done=Unbounded(shape=(1,), dtype=torch.bool, device=self.device),
            terminated=Unbounded(shape=(1,), dtype=torch.bool, device=self.device),
            shape=(),
            device=self.device,
        )

    def _reset(self, tensordict=None) -> TensorDict:
        idx = random.randint(0, len(self.dataset) - 1)
        sample = self.dataset[idx]
        cond_img = sample["cond_image"]
        H, W = cond_img.shape[-2:]
        if H != self.img_size or W != self.img_size:
            cond_img = F.interpolate(
                cond_img.unsqueeze(0),
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        self._current_gt_path = sample["gt_path"]
        return TensorDict(
            {
                "cond_image": cond_img.to(self.device),
                "start": sample["start"].to(self.device),
                "goal": sample["goal"].to(self.device),
            },
            batch_size=[],
            device=self.device,
        )

    def _step(self, tensordict: TensorDict) -> TensorDict:
        k = int(tensordict["k"].item())
        action = tensordict["action"]                           # (2*max_subgoals,)
        sub_goals = action.view(self.max_subgoals, 2)[:k]      # (k, 2)

        reward_val = _compute_reward(
            flow_model=self.flow_model,
            cond_image=tensordict["cond_image"],
            start=tensordict["start"],
            sub_goals=sub_goals,
            goal=tensordict["goal"],
            gt_path=self._current_gt_path,
            device=self.device,
            num_samples=self.num_samples,
            pts_per_seg=self.pts_per_seg,
            w_path=self.w_path,
            w_gt=self.w_gt,
            w_efficiency=self.w_efficiency,
            max_subgoals=self.max_subgoals,
            gt_score_decay_rate=self.gt_score_decay_rate,
        )
        return TensorDict(
            {
                "reward": torch.tensor([reward_val], dtype=torch.float32, device=self.device),
                "done": torch.tensor([True], dtype=torch.bool, device=self.device),
                "terminated": torch.tensor([True], dtype=torch.bool, device=self.device),
            },
            batch_size=[],
            device=self.device,
        )

    def _set_seed(self, seed: int | None) -> None:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Policy 빌드
# ---------------------------------------------------------------------------

def _build_actor(
    policy_cfg: dict,
    device: torch.device,
    pretrained_encoder: nn.Module,
) -> AutoregressiveSubGoalPolicyCount:
    model = AutoregressiveSubGoalPolicyCount(
        max_subgoals=int(policy_cfg.get("max_subgoals", 4)),
        latent_dim=int(policy_cfg.get("latent_dim", 128)),
        hidden_dim=int(policy_cfg.get("hidden_dim", 128)),
        backbone=str(policy_cfg.get("backbone", "resnet34")),
        pretrained_encoder=pretrained_encoder,
    )
    print(f"[actor] AutoregressiveSubGoalPolicyCount  max_subgoals={model.max_subgoals}  frozen encoder")
    return model.to(device)


def _build_flow_from_ckpt(ckpt: dict, device: torch.device, flow_cfg: dict) -> Flow:
    m = ckpt["config"]["model"]
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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def rl_train(config_path: str | Path = "configs/rl/reinforce_ar.toml") -> None:
    cfg = load_toml(config_path)

    seed = int(cfg["seed"]["value"])
    data_cfg   = cfg["data"]
    flow_cfg   = cfg["flow"]
    policy_cfg = cfg["policy"]
    rl_cfg     = cfg["rl"]
    output_cfg = cfg["output"]

    seed_everything(seed)
    device = get_device()

    # --- 데이터셋 ---
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

    # --- Flow 모델 ---
    ckpt_path = resolve_project_path(flow_cfg["checkpoint"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Flow checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    flow_model = _build_flow_from_ckpt(ckpt, device, flow_cfg)

    # --- 파라미터 ---
    max_subgoals       = int(policy_cfg.get("max_subgoals", 4))
    img_size           = int(rl_cfg.get("img_size", 224))
    grad_clip_norm     = float(rl_cfg.get("grad_clip_norm", 0.5))
    sg_entropy_coef    = float(rl_cfg.get("sg_entropy_coef", 0.1))
    count_entropy_coef = float(rl_cfg.get("count_entropy_coef", 0.5))
    baseline_ema       = float(rl_cfg.get("baseline_ema", 0.05))
    steps_per_update   = int(rl_cfg.get("steps_per_update", 64))
    total_steps        = int(rl_cfg.get("total_steps", 40000))
    log_interval       = int(rl_cfg.get("log_interval", 1))
    w_efficiency       = float(rl_cfg.get("reward_w_efficiency", 0.0))
    k_warmup_steps     = int(rl_cfg.get("k_warmup_steps", 0))  # 0 = 비활성

    # --- 환경 ---
    env = SubGoalEnv(
        dataset=ds,
        flow_model=flow_model,
        device=device,
        num_samples=int(rl_cfg.get("num_flow_samples", 4)),
        pts_per_seg=int(rl_cfg.get("pts_per_seg", 32)),
        img_size=img_size,
        max_subgoals=max_subgoals,
        w_path=float(rl_cfg.get("reward_w_path", 1.0)),
        w_gt=float(rl_cfg.get("reward_w_gt", 1.5)),
        w_efficiency=w_efficiency,
        gt_score_decay_rate=float(rl_cfg.get("gt_score_decay_rate", 1.0)),
    )
    env.set_seed(seed)

    # --- Policy ---
    flow_encoder = flow_model.cond_encoder.map_encoder
    policy = _build_actor(policy_cfg, device, pretrained_encoder=flow_encoder)
    optimizer = Adam(policy.parameters(), lr=float(rl_cfg.get("lr", 1e-4)))

    # --- 출력 디렉토리 ---
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = str(output_cfg.get("run_name", "")).strip() or f"rl_ar_{stamp}"
    output_root = resolve_project_path(output_cfg.get("checkpoint_root", "outputs/checkpoints"))
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "rl_config.snapshot.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"device={device}  dataset={len(ds)}  run={run_name}")
    print(f"max_subgoals={max_subgoals}  w_efficiency={w_efficiency}")
    print(f"count_entropy_coef={count_entropy_coef}  sg_entropy_coef={sg_entropy_coef}")
    print(f"k_warmup_steps={k_warmup_steps}  (0=비활성)")

    # --- REINFORCE + EMA baseline ---
    best_avg_reward = float("-inf")
    reward_history: list[float] = []
    loss_history:   list[float] = []
    baseline = 0.0

    force_k = None
    for step in range(1, total_steps + 1):
        # ── Rollout ─────────────────────────────────────────────────────
        buf_cond:   list[torch.Tensor] = []
        buf_start:  list[torch.Tensor] = []
        buf_goal:   list[torch.Tensor] = []
        buf_action: list[torch.Tensor] = []  # (max_subgoals, 2) padded
        buf_k:      list[torch.Tensor] = []  # scalar int
        buf_reward: list[torch.Tensor] = []
        total_k = 0

        
        if k_warmup_steps > 0:
            phase   = step // k_warmup_steps   # 0,1,2,...,max_subgoals
            force_k = (max_subgoals - phase) if phase < max_subgoals else None

        policy.eval()
        for _ in range(steps_per_update):
            obs = env.reset()
            cond_img = obs["cond_image"]
            start    = obs["start"]
            goal     = obs["goal"]

            with torch.no_grad():
                # forward: (count_logits, K, sampled_sgs_padded)
                _, K, sampled_sgs = policy(cond_img, start, goal, force_k=force_k)
                # sampled_sgs: (max_subgoals, 2) zero-padded
                # K: scalar tensor (int)

            k_val = int(K.item())
            total_k += k_val
            k_tensor = torch.tensor(k_val, dtype=torch.long, device=device)

            step_td = env.step(TensorDict(
                {
                    "cond_image": cond_img,
                    "start": start,
                    "goal": goal,
                    "action": sampled_sgs.view(-1),  # flat (2*max_subgoals,)
                    "k": k_tensor,
                },
                batch_size=[],
                device=device,
            ))

            buf_cond.append(cond_img)
            buf_start.append(start)
            buf_goal.append(goal)
            buf_action.append(sampled_sgs)    # (max_subgoals, 2)
            buf_k.append(k_tensor)
            buf_reward.append(step_td["next", "reward"].squeeze())

        B = steps_per_update
        b_cond   = torch.stack(buf_cond)    # (B, 3, H, W)
        b_start  = torch.stack(buf_start)   # (B, 2)
        b_goal   = torch.stack(buf_goal)    # (B, 2)
        b_action = torch.stack(buf_action)  # (B, max_subgoals, 2)
        b_k      = torch.stack(buf_k)       # (B,) int64
        b_reward = torch.stack(buf_reward)  # (B,)

        mean_reward = b_reward.mean().item()
        avg_k       = total_k / B

        # ── EMA baseline & advantage ─────────────────────────────────
        baseline = (1.0 - baseline_ema) * baseline + baseline_ema * mean_reward
        advantage = b_reward - baseline
        if advantage.std() > 1e-6:
            advantage = (advantage) / (advantage.std() + 1e-8)

        # ── REINFORCE 업데이트 (teacher forcing) ─────────────────────
        policy.train()
        count_logits, alpha, beta_p = policy.forward_ar_teacher(
            b_cond, b_start, b_goal,
            b_action.detach(),   # (B, max_subgoals, 2) — rollout 버퍼
            b_k,
        )
        # count_logits: (B, max_subgoals+1)
        # alpha, beta_p: (B, max_subgoals, 2)

        # Mask: position i는 i < K_b 일 때만 유효
        step_idx = torch.arange(max_subgoals, device=device).unsqueeze(0)  # (1, max_K)
        mask = (step_idx < b_k.unsqueeze(1)).float()                        # (B, max_K)

        # Count log_prob
        count_dist = Categorical(logits=count_logits)
        log_prob_k = count_dist.log_prob(b_k)  # (B,)

        # AR position log_prob (masked)
        log_prob_sg = (
            Beta(alpha, beta_p)
            .log_prob(b_action.clamp(0.001, 0.999))   # (B, max_K, 2)
            .sum(-1)                                   # → (B, max_K)
            * mask                                     # zero out padded
        ).sum(-1)                                      # → (B,)

        # Entropy (Sub-goal 엔트로피도 길이 편향 방지를 위해 평균으로 수정 추천!)
        count_entropy = count_dist.entropy().mean()
        
        # 수정 전: sg_entropy = (Beta(alpha, beta_p).entropy().sum(-1) * mask).sum(-1).mean()
        # 수정 후 (권장): 유효한 K 개수만큼만 나누어서 스텝당 평균 엔트로피 계산
        sg_entropy_sum = (Beta(alpha, beta_p).entropy().sum(-1) * mask).sum(-1)
        sg_entropy = (sg_entropy_sum / b_k.clamp(min=1).float()).mean()

        if force_k is not None:
            # Warmup 기간: K를 강제로 고정했으므로 Count Head 쪽은 Loss에 반영 안 함
            log_prob = log_prob_sg
            loss = -(log_prob * advantage.detach()).mean() \
                   - sg_entropy_coef * sg_entropy
        else:
            # Warmup 종료: 정상적으로 K 예측값과 Sub-goal 예측값 모두 학습
            log_prob = log_prob_k + log_prob_sg
            loss = -(log_prob * advantage.detach()).mean() \
                   - count_entropy_coef * count_entropy \
                   - sg_entropy_coef * sg_entropy

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        optimizer.step()

        reward_history.append(mean_reward)
        loss_history.append(loss.item())

        if step % log_interval == 0:
            avg_r = float(np.mean(reward_history[-log_interval:]))
            avg_l = float(np.mean(loss_history[-log_interval:]))
            print(
                f"[step {step:5d}/{total_steps}] "
                f"reward={avg_r:.4f}  loss={avg_l:.4f}  baseline={baseline:.4f}  "
                f"avg_k={avg_k:.2f}  "
                f"cnt_ent={count_entropy.item():.3f}  sg_ent={sg_entropy.item():.3f}"
            )

        ckpt_data = {
            "step": step,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "mean_reward": mean_reward,
            "baseline": baseline,
            "config": cfg,
        }
        torch.save(ckpt_data, output_dir / "last.pt")

        if mean_reward > best_avg_reward:
            print(f"Update best.pt! step={step}  reward={mean_reward:.4f}  avg_k={avg_k:.2f}")
            best_avg_reward = mean_reward
            torch.save(ckpt_data, output_dir / "best.pt")

    print(f"training done. best_avg_reward={best_avg_reward:.4f}")
    print(f"checkpoints={output_dir}")

    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(reward_history)
        ax1.axhline(y=0, color="r", linestyle="--", alpha=0.5)
        ax1.set_title("Reward per step")
        ax1.set_xlabel("Step")
        ax2.plot(loss_history)
        ax2.set_title("Loss per step")
        ax2.set_xlabel("Step")
        plt.tight_layout()
        fig.savefig(output_dir / "training_curve.png", dpi=120)
        plt.close(fig)
        print(f"training_curve={output_dir / 'training_curve.png'}")
    except Exception:
        pass
