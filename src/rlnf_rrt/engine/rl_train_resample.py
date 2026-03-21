from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Beta
from torch.optim import Adam
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import BoundedContinuous, Unbounded, Composite
import torch.nn.functional as F

from rlnf_rrt.data.dataset import RLNFDataset
from rlnf_rrt.models.flow import Flow
from rlnf_rrt.models.subgoal_policy import SubGoalPolicy
from rlnf_rrt.utils.config import load_toml, resolve_project_path
from rlnf_rrt.utils.seed import seed_everything
from rlnf_rrt.utils.utils import get_device


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def _make_seg_cond(
    cond_base: torch.Tensor,  # (3, H, W)
    seg_start: torch.Tensor,  # (2,)
    seg_goal: torch.Tensor,   # (2,)
) -> torch.Tensor:            # (1, 3, H, W)
    """channel 1을 segment start(+1) / goal(-1)로 교체한 conditioning image 반환."""
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
    return cond.unsqueeze(0)  # (1, 3, H, W)


def _make_gt_score_map(
    gt_path: torch.Tensor,   # (N, 2) normalized [0, 1]
    H: int,
    W: int,
    obstacle_mask: np.ndarray = None,  # (H, W) binary mask (1: obstacle, 0: free)
    decay_rate: float = 0.01
) -> np.ndarray:
    """
    GT 경로 기반 Score Map 생성
    - 경로 위: 5.0
    - 경로에서 멀어질수록: 5.0 - (거리 * decay_rate)
    - 장애물 위: -10.0
    """
    from scipy.ndimage import distance_transform_edt

    pts = gt_path.cpu().numpy()
    px = np.clip(np.round(pts[:, 0] * (W - 1)).astype(int), 0, W - 1)
    py = np.clip(np.round(pts[:, 1] * (H - 1)).astype(int), 0, H - 1)

    binary_map = np.ones((H, W), dtype=bool)
    binary_map[py, px] = False

    dist_to_path = distance_transform_edt(binary_map)
    score_map = 5.0 - (dist_to_path * decay_rate)

    if obstacle_mask is not None:
        score_map[obstacle_mask > 0] = -10.0

    return score_map.astype(np.float32)


@torch.no_grad()
def _seg_gt_score(
    flow_model: Flow,
    cond_image: torch.Tensor,   # (3, H, W)
    seg_start: torch.Tensor,    # (2,)
    seg_goal: torch.Tensor,     # (2,)
    gt_score_map: np.ndarray,   # (H, W) float
    num_samples: int,
    pts_per_seg: int,
    device: torch.device,
) -> float:
    """flow 샘플 경로 점들의 gt_score_map 평균 점수 (배치 단위 연산 최적화)."""
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
    total_score = float(scores.sum())
    total_pts = num_samples * pts_per_seg

    return total_score / max(total_pts, 1)


@torch.no_grad()
def _compute_reward(
    flow_model: Flow,
    cond_image: torch.Tensor,   # (3, H, W)
    start: torch.Tensor,        # (2,)
    sub_goal: torch.Tensor,     # (2,)
    goal: torch.Tensor,         # (2,)
    gt_path: torch.Tensor,      # (N, 2) normalized [0, 1]
    device: torch.device,
    num_samples: int,
    pts_per_seg: int,
    w_path: float,
    w_gt: float,
    gt_score_decay_rate: float = 0.01,
) -> float:
    H, W = cond_image.shape[-2:]

    obstacle_mask = (cond_image[0] <= 0.5).cpu().numpy().astype(np.uint8)
    gt_score_map = _make_gt_score_map(
        gt_path, H, W, obstacle_mask=obstacle_mask, decay_rate=gt_score_decay_rate
    )

    sg_xi = int(round(float(sub_goal[0].item()) * (W - 1)))
    sg_yi = int(round(float(sub_goal[1].item()) * (H - 1)))
    sg_xi = max(0, min(W - 1, sg_xi))
    sg_yi = max(0, min(H - 1, sg_yi))

    on_the_gt = gt_score_map[sg_yi, sg_xi]

    seg1 = _seg_gt_score(flow_model, cond_image, start, sub_goal, gt_score_map, num_samples, pts_per_seg, device)
    seg2 = _seg_gt_score(flow_model, cond_image, sub_goal, goal, gt_score_map, num_samples, pts_per_seg, device)
    path_reward = (seg1 + seg2) / 2.0

    return w_path * path_reward + w_gt * on_the_gt


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SubGoalEnv(EnvBase):
    """1-step MDP for sub-goal selection."""

    def __init__(
        self,
        dataset: RLNFDataset,
        flow_model: Flow,
        device: torch.device,
        num_samples: int,
        pts_per_seg: int,
        img_size: int,
        w_path: float = 1.0,
        w_gt: float = 0.5,
        gt_score_decay_rate: float = 0.01,
        **kwargs,
    ):
        super().__init__(device=device, batch_size=[], **kwargs)
        self.dataset = dataset
        self.flow_model = flow_model
        self.num_samples = num_samples
        self.pts_per_seg = pts_per_seg
        self.img_size = img_size
        self.w_path = w_path
        self.w_gt = w_gt
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
        self.action_spec = BoundedContinuous(low=0.0, high=1.0, shape=(2,), device=self.device)
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
            import torch.nn.functional as F
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
        reward_val = _compute_reward(
            flow_model=self.flow_model,
            cond_image=tensordict["cond_image"],
            start=tensordict["start"],
            sub_goal=tensordict["action"],
            goal=tensordict["goal"],
            gt_path=self._current_gt_path,
            device=self.device,
            num_samples=self.num_samples,
            pts_per_seg=self.pts_per_seg,
            w_path=self.w_path,
            w_gt=self.w_gt,
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
) -> SubGoalPolicy:
    model = SubGoalPolicy(
        latent_dim=int(policy_cfg.get("latent_dim", 64)),
        hidden_dim=int(policy_cfg.get("hidden_dim", 128)),
        backbone=str(policy_cfg.get("backbone", "resnet34")),
        num_subgoals=1,
        pretrained_encoder=pretrained_encoder,
    )
    print("[actor] using pretrained flow encoder (frozen)")
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

def rl_train(config_path: str | Path = "configs/rl/reinforce_resample.toml") -> None:
    cfg = load_toml(config_path)

    seed = int(cfg["seed"]["value"])
    data_cfg = cfg["data"]
    flow_cfg = cfg["flow"]
    policy_cfg = cfg["policy"]
    rl_cfg = cfg["rl"]
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
        raise RuntimeError("Dataset is empty. Check config[data] filters.")

    # --- Flow 모델 (reward 계산용, 학습 안 함) ---
    ckpt_path = resolve_project_path(flow_cfg["checkpoint"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Flow checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    flow_model = _build_flow_from_ckpt(ckpt, device, flow_cfg)

    # --- 환경 ---
    img_size = int(rl_cfg.get("img_size", 224))
    env = SubGoalEnv(
        dataset=ds,
        flow_model=flow_model,
        device=device,
        num_samples=int(rl_cfg.get("num_flow_samples", 8)),
        pts_per_seg=int(rl_cfg.get("pts_per_seg", 32)),
        img_size=img_size,
        w_path=float(rl_cfg.get("reward_w_path", 1.0)),
        w_gt=float(rl_cfg.get("reward_w_gt", 0.5)),
        gt_score_decay_rate=float(rl_cfg.get("gt_score_decay_rate", 0.01)),
    )
    env.set_seed(seed)

    # --- Policy ---
    grad_clip_norm        = float(rl_cfg.get("grad_clip_norm", 0.5))
    entropy_coef          = float(rl_cfg.get("entropy_coef", 0.01))
    baseline_ema          = float(rl_cfg.get("baseline_ema", 0.05))
    steps_per_update      = int(rl_cfg.get("steps_per_update", 32))
    total_steps           = int(rl_cfg.get("total_steps", 10000))
    log_interval          = int(rl_cfg.get("log_interval", 10))
    max_resamples         = int(rl_cfg.get("obstacle_max_resamples", 5))
    resample_penalty      = float(rl_cfg.get("obstacle_resample_penalty", 2.0))

    flow_encoder = flow_model.cond_encoder.map_encoder
    policy = _build_actor(policy_cfg, device, pretrained_encoder=flow_encoder)

    optimizer = Adam(policy.parameters(), lr=float(rl_cfg.get("lr", 1e-4)))

    # --- 출력 디렉토리 ---
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = str(output_cfg.get("run_name", "")).strip() or f"rl_resample_{stamp}"
    output_root = resolve_project_path(output_cfg.get("checkpoint_root", "outputs/checkpoints"))
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "rl_config.snapshot.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"device={device}  dataset={len(ds)}  run={run_name}")
    print(f"checkpoints={output_dir}")
    print(f"flow_checkpoint={ckpt_path}")
    print(f"obstacle_max_resamples={max_resamples}  obstacle_resample_penalty={resample_penalty}")

    # --- REINFORCE + EMA baseline 학습 루프 ---
    best_avg_reward = float("-inf")
    reward_history: list[float] = []
    loss_history: list[float] = []
    baseline = 0.0

    for step in range(1, total_steps + 1):
        # ── Rollout 수집 ──────────────────────────────────────────────
        buf_cond:   list[torch.Tensor] = []
        buf_start:  list[torch.Tensor] = []
        buf_goal:   list[torch.Tensor] = []
        buf_action: list[torch.Tensor] = []
        buf_reward: list[torch.Tensor] = []
        total_resamples = 0

        policy.eval()
        for _ in range(steps_per_update):
            obs = env.reset()
            cond_img = obs["cond_image"]
            start    = obs["start"]
            goal     = obs["goal"]

            H_img, W_img = cond_img.shape[-2:]
            obstacle_mask_np = (cond_img[0] <= 0.5).cpu().numpy()  # True = obstacle

            resample_count = 0
            with torch.no_grad():
                alpha, beta_param = policy(cond_img, start, goal)
                dist = Beta(alpha.clamp(min=0.1), beta_param.clamp(min=0.1))
                action = dist.sample().clamp(0.001, 0.999)

                for _ in range(max_resamples):
                    sg_xi = int(round(float(action[0].item()) * (W_img - 1)))
                    sg_yi = int(round(float(action[1].item()) * (H_img - 1)))
                    sg_xi = max(0, min(W_img - 1, sg_xi))
                    sg_yi = max(0, min(H_img - 1, sg_yi))
                    if not obstacle_mask_np[sg_yi, sg_xi]:
                        break
                    resample_count += 1
                    action = dist.sample().clamp(0.001, 0.999)

            total_resamples += resample_count

            step_td = env.step(TensorDict(
                {"cond_image": cond_img, "start": start, "goal": goal, "action": action},
                batch_size=[], device=device,
            ))

            raw_reward = step_td["next", "reward"].squeeze()
            adjusted_reward = raw_reward - resample_count * resample_penalty

            buf_cond.append(cond_img)
            buf_start.append(start)
            buf_goal.append(goal)
            buf_action.append(action)
            buf_reward.append(adjusted_reward)

        b_cond   = torch.stack(buf_cond)    # (B, 3, H, W)
        b_start  = torch.stack(buf_start)   # (B, 2)
        b_goal   = torch.stack(buf_goal)    # (B, 2)
        b_action = torch.stack(buf_action)  # (B, 2)
        b_reward = torch.stack(buf_reward)  # (B,)

        mean_reward = b_reward.mean().item()
        avg_resamples = total_resamples / steps_per_update

        # ── EMA baseline & advantage ───────────────────────────────────
        baseline = (1.0 - baseline_ema) * baseline + baseline_ema * mean_reward
        advantage = b_reward - baseline
        if advantage.std() > 1e-6:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # ── REINFORCE 업데이트 ────────────────────────────────────────
        policy.train()
        raw_alpha, raw_beta = policy(b_cond, b_start, b_goal)

        alpha = F.softplus(raw_alpha) + 0.001
        beta_param = F.softplus(raw_beta) + 0.001

        dist = Beta(alpha, beta_param)
        log_prob = dist.log_prob(b_action.clamp(0.001, 0.999)).sum(-1)  # (B,)
        entropy  = dist.entropy().sum(-1).mean()                         # scalar

        loss = -(log_prob * advantage.detach()).mean() - entropy_coef * entropy

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
                f"avg_resamples={avg_resamples:.2f}"
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
            print(f"Update best.pt! step={step}  reward={mean_reward:.4f}")
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
