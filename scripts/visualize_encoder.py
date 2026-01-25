import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import sys
from pathlib import Path

# Add src to sys.path to ensure modules can be imported
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from rlnf_rrt.data_pipeline.custom_dataset import RLNFDataset
from rlnf_rrt.data_pipeline.utils import get_device
import rlnf_rrt.models.CustomPlannerFlows

# --- Define Legacy Model to match Checkpoint ---
class LegacyMapEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # Matches checkpoint architecture inferred from error logs
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 256), # Was 1024 in current code, 256 in checkpoint
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        return self.conv_block(x)

# Monkey Patch
print("Patching MapEncoder to match legacy checkpoint...")
rlnf_rrt.models.CustomPlannerFlows.MapEncoder = LegacyMapEncoder

from rlnf_rrt.models.CustomPlannerFlows import CustomPlannerFlows

def load_latest_model(model_dir, model, device):
    """Finds and loads the latest .pth model checkpoint."""
    model_files = glob.glob(os.path.join(model_dir, "*.pth"))
    if not model_files:
        print(f"No model checkpoints found in {model_dir}. Using random weights.")
        return model
    
    # Sort by modification time
    latest_model = max(model_files, key=os.path.getmtime)
    print(f"Loading model from: {latest_model}")
    
    state_dict = torch.load(latest_model, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def visualize_encoder(model_dir="../result/models"):
    device = get_device()
    print(f"Using device: {device}")

    # Model parameters (matching checkpoint)
    masks = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],         
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ]
    hidden_dim = 128
    env_latent_dim = 128 # Changed from 256 to 128 to match checkpoint
    
    # Initialize Model
    model = CustomPlannerFlows(masks, hidden_dim, env_latent_dim).to(device)
    
    # Load Weights
    script_dir = Path(__file__).parent
    abs_model_dir = script_dir / model_dir
    model = load_latest_model(str(abs_model_dir), model, device)
    
    # Load Data
    print("Loading dataset...")
    # Use validation set to see unseen examples
    dataset = RLNFDataset(split="train") 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Get one sample
    try:
        sample = next(iter(dataloader))
    except StopIteration:
        print("Dataset is empty.")
        return

    map_tensor = sample['map'].to(device)
    
    # Forward pass through encoder only
    with torch.no_grad():
        latent_vector = model.encoder(map_tensor)
    
    # Convert to numpy for plotting
    map_img = map_tensor.cpu().numpy()[0, 0] # (B, C, H, W) -> (H, W)
    latent_vec_np = latent_vector.cpu().numpy()[0] # (B, Dim) -> (Dim,)
    
    start = sample['start'].cpu().numpy()[0] * 224.0
    goal = sample['goal'].cpu().numpy()[0] * 224.0
    gt_path = sample['gt'].cpu().numpy()[0] * 224.0

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Input Map
    axes[0].imshow(map_img, cmap='gray')
    axes[0].plot(gt_path[:, 0], gt_path[:, 1], 'b--', label='GT Path', alpha=0.6)
    axes[0].scatter(start[0], start[1], c='green', label='Start', s=100, edgecolors='black') 
    axes[0].scatter(goal[0], goal[1], c='red', label='Goal', s=100, edgecolors='black')
    axes[0].set_title("Input Map with Start/Goal")
    axes[0].legend()
    
    # Plot 2: Latent Vector
    # Reshape for heatmap if it's a square number (128 is not square, 16x8?)
    # 128 = 16 * 8.
    axes[1].bar(range(len(latent_vec_np)), latent_vec_np)
    axes[1].set_title(f"Latent Vector ({env_latent_dim} dim)")
    axes[1].set_xlabel("Dimension Index")
    axes[1].set_ylabel("Value")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = script_dir / "../result/encoder_visualization.png"
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    visualize_encoder()
