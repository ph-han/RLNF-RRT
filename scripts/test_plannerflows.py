import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from rlnf_rrt.data_pipeline.utils import get_device
from rlnf_rrt.models.PlannerFlows import PlannerFlows
from rlnf_rrt.data_pipeline.custom_dataset import RLNFDataset



def visualize_samples(model, dataset, device, num_samples=1000):
    model.eval() # 추론 모드 설정
    
    
    i = 0
    for batch in tqdm(dataset):
        plt.cla()
        map_img = batch['map'].unsqueeze(0).to(device)    # [1, 1, 224, 224]
        start = batch['start'].unsqueeze(0).to(device)    # [1, 2]
        goal = batch['goal'].unsqueeze(0).to(device)      # [1, 2]

        with torch.no_grad():
            q_samples = model.forward(map_img, start, goal, num_samples=num_samples)
        
        map_np = map_img.squeeze().cpu().numpy()
        q_samples_np = q_samples.squeeze().cpu().numpy() * 224.0
        start_np = start.squeeze().cpu().numpy() * 224.0
        goal_np = goal.squeeze().cpu().numpy() * 224.0

        plt.figure(figsize=(8, 8))
        plt.imshow(map_np, cmap='gray', origin='lower') # 맵 표시
        plt.scatter(q_samples_np[:, 0], q_samples_np[:, 1], 
                    color='blue', s=2, alpha=0.3, label='Generated Samples')
        
        plt.scatter(start_np[0], start_np[1], color='green', s=100, marker='o', label='Start')
        plt.scatter(goal_np[0], goal_np[1], color='red', s=100, marker='X', label='Goal')
        
        plt.title(f"PlannerFlows Sampling Distribution (Epoch Test)")
        plt.legend()
        plt.savefig(f"../result/visualization/res_{i}.png")
        plt.close()
        i+=1

if __name__ == "__main__":
    masks = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],         
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ]

    device = get_device()
    hidden_dim = 128
    env_latent_dim = 128
    num_epochs = 10

    model = PlannerFlows(masks, hidden_dim, env_latent_dim).to(device)
    state = torch.load("../result/models/planner_flows_v1_ep300.pth")
    model.load_state_dict(state)
    dataset = RLNFDataset(split="test")
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False)

    visualize_samples(model, dataset, device, num_samples=1000)

    
