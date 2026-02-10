import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from rlnf_rrt.data_pipeline.dataset import RLNFDataset
from rlnf_rrt.models.condition_encoder import ConditionEncoder
from rlnf_rrt.models.conditional_flow_planner import ConditionalFlowPlanner

def verify_condition_sensitivity():
    # 1. Load Dataset
    dataset = RLNFDataset(split="train", noise_std=0.0)
    
    # Get a base sample
    idx = 0
    sample = dataset[idx]
    map_base = sample["map"].unsqueeze(0)
    start_base = sample["start"].unsqueeze(0)
    goal_base = sample["goal"].unsqueeze(0)
    
    # Re-create model structure (TrajFlowPlanner as used in v3 training)
    # Check args in train_flow.py used for v3:
    # default args: num_blocks=4, map_embed_dim=256, cond_dim=128, hidden_dim=128
    full_model = ConditionalFlowPlanner(
        num_blocks=4, 
        map_embed_dim=256,
        cond_dim=128,
        hidden_dim=128
    )
    
    checkpoint_path = "result/models/v4_best_model.pt"
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    full_model.load_state_dict(checkpoint['model_state_dict'])
    
    model = full_model.condition_encoder
    model.eval() # Use eval mode for trained model! 

    print("-" * 50)
    print("Testing Condition Encoder Sensitivity")
    print("-" * 50)

    # Hook to check intermediate map_embed
    map_embeddings = []
    def hook_fn(module, input, output):
        map_embeddings.append(output)
    model.map_encoder.register_forward_hook(hook_fn)

    with torch.no_grad():
        # Debugging Inputs
        print(f"Map Base Stats: min={map_base.min().item():.4f}, max={map_base.max().item():.4f}, mean={map_base.mean().item():.4f}")
        print(f"Start Base: {start_base}")
        print(f"Goal Base: {goal_base}")
        
        # Case A: Reference (Base)
        cond_base = model(map_base, start_base, goal_base)
        print(f"\nMap Embed Base (First 5): {map_embeddings[-1][0, :5].detach()}") # Last captured embed
        print(f"Cond Base Vector (First 10): {cond_base[0, :10]}")
        print(f"Cond Base Stats: min={cond_base.min().item():.4f}, max={cond_base.max().item():.4f}, std={cond_base.std().item():.4f}")

        # Case B: Change Start Position (Slightly)
        start_mod = start_base + 0.5  # Move start point Significantly
        cond_start_mod = model(map_base, start_mod, goal_base)
        
        # Case C: Change Goal Position (Slightly)
        goal_mod = goal_base + 0.5  # Move goal point Significantly
        cond_goal_mod = model(map_base, start_base, goal_mod)

        # Case D: Totally Different Map (Next Sample)
        sample2 = None
        for i in range(1, 100):
            temp_sample = dataset[idx + i]
            # Check if map is different (compare means or exact tensors)
            if not torch.allclose(temp_sample["map"], map_base):
                sample2 = temp_sample
                print(f"\nFound different map at index {idx + i}")
                break
        
        if sample2 is None:
            # Fallback
            map_diff = map_base.clone()
            map_diff = 1.0 - map_diff # Invert map if no diff map found
            print("\n[WARNING] Could not find a different map! Inverting existing map for test.")
        else:
            map_diff = sample2["map"].unsqueeze(0)

        print(f"\nMap Diff Stats: min={map_diff.min().item():.4f}, max={map_diff.max().item():.4f}")
        cond_map_diff = model(map_diff, start_base, goal_base)
        print(f"Cond Map Diff Vector (First 10): {cond_map_diff[0, :10]}")

        # Compute Cosine Similarities
        sim_start = F.cosine_similarity(cond_base, cond_start_mod).item()
        sim_goal = F.cosine_similarity(cond_base, cond_goal_mod).item()
        sim_map = F.cosine_similarity(cond_base, cond_map_diff).item()

        print(f"Similarity when changing START: {sim_start:.4f} (Should be < 1.0)")
        print(f"Similarity when changing GOAL:  {sim_goal:.4f} (Should be < 1.0)")
        print(f"Similarity when changing MAP:   {sim_map:.4f}   (Should be << 1.0)")

        # Analysis
        if sim_start > 0.99 or sim_goal > 0.99:
            print("\n[WARNING] The encoder is ignoring Start/Goal positions! (Similarity too high)")
            print("Check if Start/Goal inputs are properly connected in the fusion layer.")
        else:
            print("\n[SUCCESS] The encoder is sensitive to Start/Goal changes.")

        if sim_map > 0.9:
            print("\n[WARNING] The encoder is ignoring Map features! (Similarity too high)")
        else:
            print("\n[SUCCESS] The encoder is sensitive to Map changes.")

if __name__ == "__main__":
    verify_condition_sensitivity()
