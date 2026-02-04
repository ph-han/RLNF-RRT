# visualize_flow_states.py
"""
Example:
  uv run scripts/visualize_flow_states.py --ckpt ../result/models/planner_flows_v7_best_loss.pth \
    --split test --idx 126 --num_samples 1000 --env_latent_dim 256 --hidden_dim 128 --inverse
"""

import argparse
import math
from typing import Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from rlnf_rrt.models.CustomPlannerFlows import CustomPlannerFlows

try:
    from rlnf_rrt.data_pipeline.custom_dataset import RLNFDataset
except Exception:
    RLNFDataset = None


# 기본 마스크 (모델 생성 시 사용한 것과 동일해야 함)
DEFAULT_MASKS = [[1.0, 0.0], [0.0, 1.0]] * 4


def _strip_module_prefix(state_dict: dict) -> dict:
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def load_model(args, device: torch.device) -> CustomPlannerFlows:
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state_dict = _strip_module_prefix(state_dict)

    masks = ckpt.get("masks", DEFAULT_MASKS) if isinstance(ckpt, dict) else DEFAULT_MASKS

    model = CustomPlannerFlows(masks, args.hidden_dim, args.env_latent_dim, state_dim=2).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def get_data_sample(args, device: torch.device):
    if RLNFDataset is None:
        raise ImportError("RLNFDataset을 찾을 수 없습니다.")

    dataset = RLNFDataset(dataset_root_path=args.dataset_root, split=args.split)
    sample = dataset[args.idx]

    return {
        "map": sample["map"].unsqueeze(0).to(device),
        "start": sample["start"].unsqueeze(0).to(device),
        "goal": sample["goal"].unsqueeze(0).to(device),
        "gt": sample["gt"].to(device) if "gt" in sample else None
    }


def _get_prior_params_if_available(model: CustomPlannerFlows, cond_rep: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    CustomPlannerFlows에 conditional prior가 있으면 (mu, log_sigma)를 반환.
    없으면 (None, None).
    """
    if hasattr(model, "_prior_params"):
        mu, log_sigma = model._prior_params(cond_rep)
        return mu, log_sigma
    return None, None


def _sample_conditional_prior(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    eps = torch.randn_like(mu)
    return mu + torch.exp(log_sigma) * eps


def collect_flow_steps(model, sample, num_samples, is_inverse=False, whiten_eps=False):
    """
    레이어별 변환 과정을 수집.
    - is_inverse=False: z -> x (forward)
    - is_inverse=True : x(gt) -> z (inverse)
        - conditional prior가 있으면 "Target Gaussian"은 N(mu(cond), sigma(cond))로 샘플링
        - whiten_eps=True면 최종 z를 eps=(z-mu)/sigma로 변환하여 표준 N(0,I)와 비교 가능
    """
    device = sample["map"].device
    with torch.no_grad():
        condition = model._get_condition(sample["map"], sample["start"], sample["goal"])  # (1, C)

        if not is_inverse:
            # Forward: prior -> data
            # conditional prior가 있으면 그 prior에서 z 샘플, 없으면 N(0,I)
            cond_rep = condition.expand(num_samples, -1)

            mu, log_sigma = _get_prior_params_if_available(model, cond_rep)
            if mu is not None:
                curr = _sample_conditional_prior(mu, log_sigma)
                labels = ["Step 0 (Prior z ~ N(mu, sigma))"]
            else:
                curr = torch.randn(num_samples, 2, device=device)
                labels = ["Step 0 (z ~ N(0, I))"]

            states = [curr.cpu().numpy()]
            for layer in model.flow.layers:
                curr, _ = layer(curr, cond_rep)
                states.append(curr.cpu().numpy())
                labels.append(f"Step {len(states)-1}")

            # axis는 data 공간(-1~1)일 가능성이 높으니 고정 옵션 유지
            axis_mode = "data"
            return states, labels, axis_mode

        else:
            # Inverse: data(gt) -> latent z
            curr = sample["gt"]
            if curr is None:
                raise ValueError("sample['gt']가 없습니다. --inverse는 GT가 필요합니다.")
            if curr.shape[0] > num_samples:
                curr = curr[:num_samples]

            states = [curr.cpu().numpy()]
            labels = ["Inv Step 0 (GT)"]

            cond_rep = condition.expand(curr.size(0), -1)

            # step-wise inverse
            for layer in reversed(model.flow.layers):
                curr, _ = layer.inverse(curr, cond_rep)
                states.append(curr.cpu().numpy())
                labels.append(f"Inv Step {len(states)-1}")

            z = curr  # final latent

            mu, log_sigma = _get_prior_params_if_available(model, cond_rep)

            # (1) Final z plot
            states.append(z.cpu().numpy())
            labels.append("Final Inv Step (z)")

            # (2) Target prior sample plot (conditional prior가 있으면 그걸로!)
            if mu is not None:
                z_tgt = _sample_conditional_prior(mu, log_sigma)
                states.append(z_tgt.cpu().numpy())
                labels.append("Target Prior z ~ N(mu(cond), sigma(cond))")
            else:
                states.append(torch.randn_like(z).cpu().numpy())
                labels.append("Target Gaussian z ~ N(0, I)")

            # (3) Optional: whitened eps plot (가장 강력한 디버깅)
            # eps = (z - mu)/sigma 가 표준 N(0,I)처럼 보여야 정상
            if whiten_eps and (mu is not None):
                eps = (z - mu) * torch.exp(-log_sigma)
                eps_tgt = torch.randn_like(eps)
                states.append(eps.cpu().numpy())
                labels.append("Whitened eps = (z-mu)/sigma")
                states.append(eps_tgt.cpu().numpy())
                labels.append("Target eps ~ N(0, I)")

                axis_mode = "eps"  # latent/eps는 [-1,1]로 자르면 안 됨
            else:
                axis_mode = "latent"  # latent도 [-1,1]로 자르면 안 됨

            return states, labels, axis_mode


def plot_and_save(states, labels, save_path, axis_mode="data", max_panels=300):
    """
    axis_mode:
      - "data": [-1,1] 고정 (맵 좌표 같은 데이터 공간)
      - "latent"/"eps": 자동 범위(데이터 min/max 기반)로 보여주기
    """
    num_steps = len(states)
    indices = np.linspace(0, num_steps - 1, min(num_steps, max_panels), dtype=int)

    cols = min(4, len(indices))
    rows = math.ceil(len(indices) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    # axis 범위 결정
    if axis_mode == "data":
        xlim = (-1.0, 1.0)
        ylim = (-1.0, 1.0)
    else:
        # latent/eps는 자동 범위 (너가 지금 -1~1 고정해서 “선처럼 보이는” 착시가 생길 수 있음)
        all_pts = np.concatenate([states[i] for i in indices], axis=0)
        xmin, ymin = np.min(all_pts, axis=0)
        xmax, ymax = np.max(all_pts, axis=0)
        # 여백
        padx = 0.05 * (xmax - xmin + 1e-6)
        pady = 0.05 * (ymax - ymin + 1e-6)
        xlim = (xmin - padx, xmax + padx)
        ylim = (ymin - pady, ymax + pady)

    for i, idx in enumerate(indices):
        data = states[idx]
        axes[i].scatter(data[:, 0], data[:, 1], s=2, alpha=0.5, color='royalblue')
        axes[i].set_title(labels[idx])
        axes[i].set_xlim(*xlim)
        axes[i].set_ylim(*ylim)
        axes[i].grid(True, linestyle='--', alpha=0.6)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved visualization to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--inverse", action="store_true", help="GT에서 latent으로 가는 과정 시각화")
    parser.add_argument("--whiten_eps", action="store_true", help="conditional prior가 있을 때 eps=(z-mu)/sigma도 같이 시각화")

    # 모델 복원을 위한 인자
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--env_latent_dim", type=int, default=64)
    parser.add_argument("--dataset_root", type=str, default="data")

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = load_model(args, device)
    sample = get_data_sample(args, device)

    states, labels, axis_mode = collect_flow_steps(
        model, sample, args.num_samples, is_inverse=args.inverse, whiten_eps=args.whiten_eps
    )

    mode = "inv" if args.inverse else "fwd"
    suffix = "_eps" if (args.inverse and args.whiten_eps) else ""
    save_path = f"flow_{mode}{suffix}_idx{args.idx}.png"
    plot_and_save(states, labels, save_path, axis_mode=axis_mode)


if __name__ == "__main__":
    main()
