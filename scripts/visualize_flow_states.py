"""
Example:
  python scripts/visualize_flow_states.py --ckpt outputs/model.pt --idx 0 --num_samples 5000 --save_gif --device cuda
"""

import argparse
import math
import os
from typing import List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation

from rlnf_rrt.models.CustomPlannerFlows import CustomPlannerFlows
from rlnf_rrt.data_pipeline.utils import get_device

try:
    from rlnf_rrt.data_pipeline.custom_dataset import RLNFDataset
except Exception:
    RLNFDataset = None

# 기본 마스크 (모델 생성 시 사용한 것과 동일해야 함)
DEFAULT_MASKS = [[1.0, 0.0], [0.0, 1.0]] * 12

def _strip_module_prefix(state_dict: dict) -> dict:
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}

def load_model(args, device: torch.device) -> CustomPlannerFlows:
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state_dict = _strip_module_prefix(state_dict)

    # 마스크 및 하이퍼파라미터 추론 (생략 가능하면 args에서 가져옴)
    masks = ckpt.get("masks", DEFAULT_MASKS) if isinstance(ckpt, dict) else DEFAULT_MASKS
    
    # 모델 로드 (state_dim=2 고정)
    model = CustomPlannerFlows(masks, args.hidden_dim, args.env_latent_dim, state_dim=2).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def get_data_sample(args, device: torch.device):
    """데이터셋에서 특정 인덱스의 샘플(Map, Start, Goal, GT)을 가져옴"""
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

def collect_flow_steps(model, sample, num_samples, is_inverse=False):
    """레이어별 변환 과정을 수집"""
    device = sample["map"].device
    with torch.no_grad():
        condition = model._get_condition(sample["map"], sample["start"], sample["goal"])
        
        if not is_inverse:
            # Forward: Gaussian -> Data (-3~3)
            curr = torch.randn(num_samples, 2, device=device)
            states = [curr.cpu().numpy()]
            cond_rep = condition.expand(num_samples, -1)
            for layer in model.flow.layers:
                curr, _ = layer(curr, cond_rep)
                states.append(curr.cpu().numpy())
            labels = [f"Step {i} (Gaussian)" if i==0 else f"Step {i}" for i in range(len(states))]
        else:
            # Inverse: Data (-3~3) -> Gaussian
            curr = sample["gt"]
            if curr.shape[0] > num_samples:
                curr = curr[:num_samples]
            states = [curr.cpu().numpy()]
            cond_rep = condition.expand(curr.size(0), -1)
            for layer in reversed(model.flow.layers):
                curr, _ = layer.inverse(curr, cond_rep)
                states.append(curr.cpu().numpy())
            states.append(torch.randn_like(curr).cpu().numpy()) # 타겟 비교용
            labels = [f"Inv Step {i} (GT)" if i==0 else f"Inv Step {i}" for i in range(len(states)-1)] + ["Target Gaussian"]
            
    return states, labels

def plot_and_save(states, labels, save_path, max_panels=8):
    """수집된 상태들을 서브플롯으로 시각화"""
    num_steps = len(states)
    indices = np.linspace(0, num_steps - 1, min(num_steps, max_panels), dtype=int)
    
    cols = min(4, len(indices))
    rows = math.ceil(len(indices) / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, idx in enumerate(indices):
        data = states[idx]
        axes[i].scatter(data[:, 0], data[:, 1], s=2, alpha=0.5, color='royalblue')
        axes[i].set_title(labels[idx])
        # 데이터 범위가 -3~3이므로 축 고정
        axes[i].set_xlim(-4, 4)
        axes[i].set_ylim(-4, 4)
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
    parser.add_argument("--inverse", action="store_true", help="GT에서 Gaussian으로 가는 과정 시각화")
    parser.add_argument("--save_gif", action="store_true")
    
    # 모델 복원을 위한 인자 (체크포인트에 정보가 없을 경우 대비)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--env_latent_dim", type=int, default=64)
    parser.add_argument("--dataset_root", type=str, default="data")
    
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. 모델 및 데이터 로드
    model = load_model(args, device)
    sample = get_data_sample(args, device)

    # 2. 상태 수집
    states, labels = collect_flow_steps(model, sample, args.num_samples, is_inverse=args.inverse)

    # 3. 결과 저장
    mode = "inv" if args.inverse else "fwd"
    save_path = f"outputs/flow_{mode}_idx{args.idx}.png"
    plot_and_save(states, labels, save_path)

    # 4. GIF 저장 (선택 사항)
    if args.save_gif:
        from matplotlib import animation
        fig, ax = plt.subplots(figsize=(6, 6))
        def update(i):
            ax.clear()
            ax.scatter(states[i][:, 0], states[i][:, 1], s=2, color='crimson')
            ax.set_title(labels[i])
            ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        
        anim = animation.FuncAnimation(fig, update, frames=len(states), interval=200)
        anim.save(f"outputs/flow_{mode}_idx{args.idx}.gif", writer='pillow')
        print("Saved GIF.")

if __name__ == "__main__":
    main()