import torch
import numpy as np
from rlnf_rrt.models.conditional_flow_planner import ConditionalFlowPlanner
from rlnf_rrt.training.config import TrainConfig
import os

def check_v6_1_sensitivity(model_path):
    print(f"\n--- Analyzing {os.path.basename(model_path)} (v6.1 32D Bottleneck) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Instantiate model with 32D position embedding
    model = ConditionalFlowPlanner(
        num_blocks=4,
        position_embed_dim=32,
        map_embed_dim=256,
        cond_dim=128,
        hidden_dim=128
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    encoder = model.condition_encoder
    encoder.eval()

    # Create dummy inputs
    map_img = torch.randn(1, 1, 224, 224).to(device)
    start = torch.randn(1, 2).to(device)
    goal = torch.randn(1, 2).to(device)

    with torch.no_grad():
        corig = encoder(map_img, start, goal)
        
        # 1. Map Sensitivity (Check if changing map changes the embedding)
        map_noise = map_img + torch.randn_like(map_img) * 0.1
        c_map = encoder(map_noise, start, goal)
        map_sim = torch.nn.functional.cosine_similarity(corig, c_map).item()
        
        # 2. Start/Goal Sensitivity
        s_noise = start + torch.randn_like(start) * 0.1
        c_sg = encoder(map_img, s_noise, goal)
        sg_sim = torch.nn.functional.cosine_similarity(corig, c_sg).item()

    print(f"Val Loss (Best): {checkpoint.get('val_loss', 'N/A')}")
    print(f"Map Cosine Similarity (Lower is better): {map_sim:.4f}")
    print(f"S/G Cosine Similarity (Lower is better): {sg_sim:.4f}")
    
    print("-" * 50)
    print("V5 Comparison: Map Sim 0.9745 | SG Sim 0.89/0.82")
    print("-" * 50)

if __name__ == "__main__":
    check_v6_1_sensitivity("result/models/v6_1_best_model.pt")
