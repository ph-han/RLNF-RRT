import torch
import numpy as np
import matplotlib.pyplot as plt
from rlnf_rrt.data_pipeline.dataset import RLNFDataset
from PIL import Image

def debug_coordinates():
    dataset = RLNFDataset(split='test')
    idx = 182 # 아까 장애물을 뚫고 지나갔던 예제 번호
    data = dataset[idx]
    
    # 1. 원본 지도 (dataset.py에서 np.flipud 제거된 상태)
    map_np = data["map"][0].numpy() # (224, 224)
    gt_path = data["gt_path"].numpy() # (N, 2)
    start = data["start"].numpy()
    goal = data["goal"].numpy()

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    orientations = [
        ("Original (origin='upper')", map_np, 'upper'),
        ("Original (origin='lower')", map_np, 'lower'),
        ("Flip Vertical (origin='upper')", np.flipud(map_np), 'upper'),
        ("Flip Vertical (origin='lower')", np.flipud(map_np), 'lower'),
    ]

    for i, (title, img, origin) in enumerate(orientations):
        ax = axes[i//2, i%2]
        ax.imshow(img, cmap='gray_r', origin=origin, extent=[0, 1, 0, 1], alpha=0.6)
        ax.scatter(gt_path[:, 0], gt_path[:, 1], c='green', s=5, label='GT')
        ax.scatter(start[0], start[1], c='red', s=100, marker='o', label='Start')
        ax.scatter(goal[0], goal[1], c='lime', s=100, marker='x', label='Goal')
        ax.set_title(title)
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.savefig("coordinate_debug.png")
    print("✅ Saved coordinate_debug.png. Please check which one is correct!")

if __name__ == "__main__":
    debug_coordinates()
