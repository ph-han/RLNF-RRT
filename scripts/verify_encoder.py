import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from rlnf_rrt.data_pipeline.dataset import RLNFDataset
from rlnf_rrt.models.condition_encoder import ConditionEncoder
from rlnf_rrt.models.conditional_flow_planner import ConditionalFlowPlanner

def verify_encoder():
    # 1. Load Dataset
    dataset = RLNFDataset(split="train", noise_std=0.0)
    sample = dataset[0]  # Get first sample
    
    map_img = sample["map"].unsqueeze(0)  # (1, 1, 64, 64)
    start = sample["start"].unsqueeze(0)
    goal = sample["goal"].unsqueeze(0)

    print(f"Map shape: {map_img.shape}")


    full_model = ConditionalFlowPlanner(
        num_blocks=4,
        map_embed_dim=256,
        cond_dim=128,
        hidden_dim=128
    )
    
    checkpoint_path = "result/checkpoints/v3_model_50.pt"
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    full_model.load_state_dict(checkpoint['model_state_dict'])

    model = full_model.condition_encoder
    model.eval()

    # 3. Hook Feature Maps (Intermediate Outputs)
    feature_maps = []
    def hook_fn(module, input, output):
        feature_maps.append(output.detach())

    # Register hook on the first Conv layer of SimpleMapEncoder
    # Adjust layer index based on your SimpleMapEncoder structure
    # Typically features[0] is the first Conv2d+ReLU
    model.map_encoder.features[0].register_forward_hook(hook_fn)      # First layer
    model.map_encoder.features[4].register_forward_hook(hook_fn)      # Second layer (after pool)
    
    # 4. Forward Pass
    with torch.no_grad():
        output = model(map_img, start, goal)

    print("Forward pass successful!")
    print(f"Condition Vector Output shape: {output.shape}")

    # 5. Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original Map
    axes[0].imshow(map_img[0, 0], cmap='gray')
    axes[0].set_title("Input Map (Occupancy)")
    axes[0].scatter(start[0, 0]*64, start[0, 1]*64, c='g', label='Start')
    axes[0].scatter(goal[0, 0]*64, goal[0, 1]*64, c='r', label='Goal')
    axes[0].legend()

    # Feature Map 1 (Early Layer) - Should detect edges/obstacles
    if len(feature_maps) > 0:
        fmap1 = feature_maps[0][0].mean(dim=0) # Average over channels
        axes[1].imshow(fmap1, cmap='viridis')
        axes[1].set_title("Layer 1 Features (Edge Detection)")

    # Feature Map 2 (Deeper Layer) - Should show abstract regions
    if len(feature_maps) > 1:
        fmap2 = feature_maps[1][0].mean(dim=0)
        axes[2].imshow(fmap2, cmap='viridis')
        axes[2].set_title("Layer 2 Features (Abstract)")

    plt.tight_layout()
    plt.savefig("result/encoder_features.png")
    print("Saved visualization to 'result/encoder_features.png'")

if __name__ == "__main__":
    verify_encoder()
