from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import EnvBase
from torchrl.data import BoundedContinuous, Unbounded, Composite
from torchrl.modules import ProbabilisticActor, TanhNormal

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
    gt_path: torch.Tensor,  # (N, 2) normalized [0, 1]
    H: int,
    W: int,
    sigma: float = 8.0,
) -> np.ndarray:            # (H, W) float32, values in [0, 1]
    """
    GT 경로(A* 기반)를 Gaussian blur해서 score map 생성.
    sub_goal이 GT 경로 근처에 있을수록 높은 점수를 받는다.
    """
    from scipy.ndimage import gaussian_filter
    score = np.zeros((H, W), dtype=np.float32)
    pts = gt_path.cpu().numpy()
    px = np.clip(np.round(pts[:, 0] * (W - 1)).astype(int), 0, W - 1)
    py = np.clip(np.round(pts[:, 1] * (H - 1)).astype(int), 0, H - 1)
    score[py, px] = 1.0
    score = gaussian_filter(score, sigma=sigma)
    max_val = score.max()
    if max_val > 1e-8:
        score /= max_val
    return score


@torch.no_grad()
def _compute_reward(
    flow_model: Flow,
    cond_image: torch.Tensor,   # (3, H, W)  on device
    start: torch.Tensor,        # (2,)
    sub_goal: torch.Tensor,     # (2,)
    goal: torch.Tensor,         # (2,)
    gt_path: torch.Tensor,      # (N, 2) normalized [0, 1]
    device: torch.device,
    num_samples: int,
    pts_per_seg: int,
    sg_obstacle_penalty: float,
    w_path: float,
    w_sdf: float,
    w_gt: float,
    gt_score_sigma: float,
) -> float:
    """
    sub_goal 위치의 품질을 세 가지 신호로 평가.

    1. path reward  : flow 모델로 생성한 segment 경로의 충돌없는 비율  ∈ [0, 1]
    2. SDF reward   : cond_image[2] (SDF 채널)에서 sub_goal의 clearance  ∈ [0, 1]
    3. GT score     : GT(A*) 경로를 Gaussian blur한 score map에서의 점수  ∈ [0, 1]

    sub_goal이 장애물이면 즉시 -sg_obstacle_penalty 반환.
    """
    H, W = cond_image.shape[-2:]
    binary_map = cond_image[0] > 0.5  # free=True, (H, W)

    sg_xi = int(round(float(sub_goal[0].item()) * (W - 1)))
    sg_yi = int(round(float(sub_goal[1].item()) * (H - 1)))
    sg_xi = max(0, min(W - 1, sg_xi))
    sg_yi = max(0, min(H - 1, sg_yi))
    if not binary_map[sg_yi, sg_xi].item():
        return -sg_obstacle_penalty

    # 1. Path reward: segment별 충돌없는 경로 비율
    def _seg_free_rate(seg_s: torch.Tensor, seg_g: torch.Tensor) -> float:
        seg_cond = _make_seg_cond(cond_image, seg_s, seg_g).to(device)
        s_t = seg_s.unsqueeze(0).to(device)
        g_t = seg_g.unsqueeze(0).to(device)
        free_count = 0
        for _ in range(num_samples):
            z = torch.randn(1, pts_per_seg, 2, device=device, dtype=seg_cond.dtype)
            pred, _ = flow_model.inverse(seg_cond, s_t, g_t, z)
            path = pred.squeeze(0).clamp(0.0, 1.0)
            px = (path[:, 0] * (W - 1)).long().clamp(0, W - 1)
            py = (path[:, 1] * (H - 1)).long().clamp(0, H - 1)
            if binary_map[py, px].all():
                free_count += 1
        return free_count / num_samples

    path_reward = 0.5 * (_seg_free_rate(start, sub_goal) + _seg_free_rate(sub_goal, goal))

    # 2. SDF reward: sub_goal 위치의 signed distance field 값 → [0, 1]로 정규화
    sdf_val = float(cond_image[2, sg_yi, sg_xi].item())  # [-1, 1]
    sdf_reward = (sdf_val + 1.0) * 0.5                   # → [0, 1]

    # 3. GT score: A* GT 경로 기반 Gaussian score map에서 sub_goal 점수
    gt_score_map = _make_gt_score_map(gt_path, H, W, sigma=gt_score_sigma)
    gt_score = float(gt_score_map[sg_yi, sg_xi])

    return w_path * path_reward + w_sdf * sdf_reward + w_gt * gt_score


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SubGoalEnv(EnvBase):
    """
    1-step MDP for sub-goal selection.
    reset: 데이터셋에서 랜덤 샘플 로드.
    step:  policy가 선택한 sub_goal로 reward 계산.
    """

    def __init__(
        self,
        dataset: RLNFDataset,
        flow_model: Flow,
        device: torch.device,
        num_samples: int,
        pts_per_seg: int,
        img_size: int,
        sg_obstacle_penalty: float,
        w_path: float = 1.0,
        w_sdf: float = 0.3,
        w_gt: float = 0.5,
        gt_score_sigma: float = 8.0,
        **kwargs,
    ):
        super().__init__(device=device, batch_size=[], **kwargs)
        self.dataset = dataset
        self.flow_model = flow_model
        self.num_samples = num_samples
        self.pts_per_seg = pts_per_seg
        self.img_size = img_size
        self.sg_obstacle_penalty = sg_obstacle_penalty
        self.w_path = w_path
        self.w_sdf = w_sdf
        self.w_gt = w_gt
        self.gt_score_sigma = gt_score_sigma
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

        # gt_path는 reward 계산에만 사용 (policy 입력 아님)
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
            sg_obstacle_penalty=self.sg_obstacle_penalty,
            w_path=self.w_path,
            w_sdf=self.w_sdf,
            w_gt=self.w_gt,
            gt_score_sigma=self.gt_score_sigma,
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
) -> ProbabilisticActor:
    policy_net = SubGoalPolicy(
        latent_dim=int(policy_cfg.get("latent_dim", 64)),
        hidden_dim=int(policy_cfg.get("hidden_dim", 128)),
        backbone=str(policy_cfg.get("backbone", "resnet34")),
        num_subgoals=1,
    )
    policy_module = TensorDictModule(
        policy_net,
        in_keys=["cond_image", "start", "goal"],
        out_keys=["loc", "scale"],
    )
    actor = ProbabilisticActor(
        module=policy_module,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        distribution_kwargs={"low": 0.0, "high": 1.0},
        return_log_prob=True,
    ).to(device)
    return actor


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

def rl_train(config_path: str | Path = "configs/rl/default.toml") -> None:
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
        sg_obstacle_penalty=float(rl_cfg.get("sg_obstacle_penalty", 2.0)),
        w_path=float(rl_cfg.get("reward_w_path", 1.0)),
        w_sdf=float(rl_cfg.get("reward_w_sdf", 0.3)),
        w_gt=float(rl_cfg.get("reward_w_gt", 0.5)),
        gt_score_sigma=float(rl_cfg.get("gt_score_sigma", 8.0)),
    )
    env.set_seed(seed)

    # --- Policy ---
    actor = _build_actor(policy_cfg, device)
    optimizer = Adam(actor.parameters(), lr=float(rl_cfg.get("lr", 3e-4)))
    grad_clip_norm = float(rl_cfg.get("grad_clip_norm", 1.0))
    entropy_coef = float(rl_cfg.get("entropy_coef", 0.01))
    steps_per_update = int(rl_cfg.get("steps_per_update", 16))
    total_steps = int(rl_cfg.get("total_steps", 1000))
    log_interval = int(rl_cfg.get("log_interval", 10))

    # --- 출력 디렉토리 ---
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = str(output_cfg.get("run_name", "")).strip() or f"rl_{stamp}"
    output_root = resolve_project_path(output_cfg.get("checkpoint_root", "outputs/checkpoints"))
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "rl_config.snapshot.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"device={device}  dataset={len(ds)}  run={run_name}")
    print(f"checkpoints={output_dir}")
    print(f"flow_checkpoint={ckpt_path}")

    # --- REINFORCE 학습 루프 ---
    baseline = 0.0
    alpha_baseline = 0.05
    best_avg_reward = float("-inf")
    reward_history: list[float] = []
    loss_history: list[float] = []

    for step in range(1, total_steps + 1):
        log_probs_list: list[torch.Tensor] = []
        rewards_list: list[torch.Tensor] = []

        for _ in range(steps_per_update):
            obs = env.reset()
            actor_out = actor(obs.clone())
            step_td = env.step(actor_out)

            log_probs_list.append(actor_out["action_log_prob"])
            rewards_list.append(step_td["next", "reward"].squeeze())

        rewards = torch.stack(rewards_list)      # (B,)
        log_probs = torch.stack(log_probs_list)  # (B,)

        mean_reward = rewards.mean().item()
        baseline = (1.0 - alpha_baseline) * baseline + alpha_baseline * mean_reward

        advantage = (rewards - baseline).detach()
        if advantage.std() > 1e-6:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        entropy = -log_probs.mean()
        loss = -(log_probs * advantage).mean() - entropy_coef * entropy

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(actor.parameters(), grad_clip_norm)
        optimizer.step()

        reward_history.append(mean_reward)
        loss_history.append(loss.item())

        if step % log_interval == 0:
            avg_r = float(np.mean(reward_history[-log_interval:]))
            avg_l = float(np.mean(loss_history[-log_interval:]))
            print(
                f"[step {step:5d}/{total_steps}] "
                f"reward={avg_r:.4f}  loss={avg_l:.4f}  baseline={baseline:.4f}"
            )

        ckpt_data = {
            "step": step,
            "model_state_dict": actor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "mean_reward": mean_reward,
            "config": cfg,
        }
        torch.save(ckpt_data, output_dir / "last.pt")
        if mean_reward > best_avg_reward:
            best_avg_reward = mean_reward
            torch.save(ckpt_data, output_dir / "best.pt")

    print(f"training done. best_avg_reward={best_avg_reward:.4f}")
    print(f"checkpoints={output_dir}")

    # 학습 곡선 저장
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
