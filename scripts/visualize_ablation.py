import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from rlnf_rrt.models.conditional_flow_planner import ConditionalFlowPlanner
from rlnf_rrt.data_pipeline.dataset import RLNFDataset
from rlnf_rrt.training.config import TrainConfig
from pathlib import Path

def visualize_ablation(checkpoint_path, num_samples=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing Ablation on {device}...")

    # Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Model Setup
    model = ConditionalFlowPlanner(
        num_blocks=4,
        position_embed_dim=32,
        map_embed_dim=256,
        cond_dim=128,
        hidden_dim=128
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load Example
    dataset = RLNFDataset(split='test')
    idx = 182 
    data = dataset[idx]
    
    real_map = data["map"].unsqueeze(0).to(device)
    start = data["start"].unsqueeze(0).to(device)
    goal = data["goal"].unsqueeze(0).to(device)
    gt_path = data["gt_path"].numpy()

    zero_map = torch.zeros_like(real_map)

    with torch.no_grad():
        samples_normal = model.sample(real_map, start, goal, num_samples=num_samples).cpu().numpy()[0]
        samples_zero = model.sample(zero_map, start, goal, num_samples=num_samples).cpu().numpy()[0]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # NORMAL
    axes[0].imshow(data["map"][0].numpy(), cmap='gray_r', extent=[0, 1, 0, 1], alpha=0.5)
    axes[0].scatter(samples_normal[:, 0], samples_normal[:, 1], c='blue', s=8, alpha=0.4, label='Generated')
    axes[0].scatter(gt_path[:, 0], gt_path[:, 1], c='green', s=8, alpha=0.8, label='GT')
    axes[0].add_patch(Circle(data["start"].numpy(), 0.02, color='red'))
    axes[0].add_patch(Circle(data["goal"].numpy(), 0.02, color='lime'))
    axes[0].set_title("NORMAL (Reverted Map)")

    # ZERO
    axes[1].imshow(data["map"][0].numpy(), cmap='gray_r', extent=[0, 1, 0, 1], alpha=0.5)
    axes[1].scatter(samples_zero[:, 0], samples_zero[:, 1], c='orange', s=8, alpha=0.4, label='Zero Map')
    axes[1].scatter(gt_path[:, 0], gt_path[:, 1], c='green', s=8, alpha=0.8, label='GT')
    axes[1].add_patch(Circle(data["start"].numpy(), 0.02, color='red'))
    axes[1].add_patch(Circle(data["goal"].numpy(), 0.02, color='lime'))
    axes[1].set_title("ABLATION (Zero Map)")

    plt.savefig("result/visualization/ablation_reverted.png")
    print("✅ Saved result/visualization/ablation_reverted.png")

if __name__ == "__main__":
    visualize_ablation("result/models/v6_2_best_model.pt")
