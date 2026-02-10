#!/usr/bin/env python
import torch
import matplotlib.pyplot as plt

from rlnf_rrt.data_pipeline.dataset import RLNFDataset
from rlnf_rrt.models.conditional_flow_planner import ConditionalFlowPlanner


def verify_encoder():
    dataset = RLNFDataset(split="train", noise_std=0.0)
    sample = dataset[0]

    map_img = sample["map"].unsqueeze(0)
    start = sample["start"].unsqueeze(0)
    goal = sample["goal"].unsqueeze(0)

    model = ConditionalFlowPlanner(
        num_blocks=4,
        sg_dim=2,
        map_embed_dim=256,
        hidden_dim=128,
    )

    feature_maps = {}

    def save_hook(name):
        def _hook(_module, _inputs, output):
            feature_maps[name] = output.detach()
        return _hook

    model.condition_encoder.map_encoder.layer1.register_forward_hook(save_hook("layer1"))
    model.condition_encoder.map_encoder.layer4.register_forward_hook(save_hook("layer4"))

    with torch.no_grad():
        map_feat, sg_feat = model.condition_encoder(map_img, start, goal)

    print("Forward pass successful")
    print(f"map_feat shape: {map_feat.shape}")
    print(f"sg_feat shape: {sg_feat.shape}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(map_img[0, 0], cmap="gray")
    axes[0].scatter(start[0, 0] * 64, start[0, 1] * 64, c="g", label="Start")
    axes[0].scatter(goal[0, 0] * 64, goal[0, 1] * 64, c="r", label="Goal")
    axes[0].set_title("Input Map")
    axes[0].legend()

    if "layer1" in feature_maps:
        axes[1].imshow(feature_maps["layer1"][0].mean(dim=0), cmap="viridis")
        axes[1].set_title("ResNet layer1")

    if "layer4" in feature_maps:
        axes[2].imshow(feature_maps["layer4"][0].mean(dim=0), cmap="viridis")
        axes[2].set_title("ResNet layer4")

    plt.tight_layout()
    plt.savefig("result/encoder_features.png")
    print("Saved visualization to result/encoder_features.png")


if __name__ == "__main__":
    verify_encoder()
