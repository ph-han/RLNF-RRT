import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from rlnf_rrt.data_pipeline.utils import get_device
from rlnf_rrt.models.CustomPlannerFlows2 import CustomPlannerFlows
from rlnf_rrt.data_pipeline.custom_dataset2 import RLNFDataset



def visualize_samples(model, dataset, device, num_samples=1000):
    model.eval() # 추론 모드 설정
    
    
    i = 0
    loss_mean = []
    for batch in tqdm(dataset):
        plt.cla()
        
        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            # start_event.record()
            map_img = batch['map'].unsqueeze(0).to(device)    # [1, 1, 224, 224]
            start = batch['start'].unsqueeze(0).to(device)    # [1, 2]
            goal = batch['goal'].unsqueeze(0).to(device)      # [1, 2]
            q_samples, ll = model.forward(map_img, start, goal, num_samples=num_samples)
            loss_mean.append(ll.mean().cpu())
            
        
        map_np = map_img.squeeze().cpu().numpy()
        q_samples_np = q_samples.squeeze().cpu().numpy() * 224.0
        start_np = start.squeeze().cpu().numpy() * 224.0
        goal_np = goal.squeeze().cpu().numpy() * 224.0
        gt_np = batch['gt'].squeeze() * 224.0
        
        plt.figure(figsize=(8, 8))
        plt.imshow(map_np, cmap='gray', origin='lower') # 맵 표시
        plt.scatter(q_samples_np[:, 0], q_samples_np[:, 1], 
                    color='blue', s=2, alpha=0.3, label='Generated Samples')
        plt.scatter(gt_np[:, 0], gt_np[:, 1], color='orange', s=2, alpha=0.7, label='Ground Truth')
        
        plt.scatter(start_np[0], start_np[1], color='green', s=100, marker='o', label='Start')
        plt.scatter(goal_np[0], goal_np[1], color='red', s=100, marker='X', label='Goal')
        
        plt.title(f"PlannerFlows Sampling Distribution (Epoch Test)")
        plt.legend()
        plt.savefig(f"../result/visualization/res_{i}.png")
        plt.close()
        i+=1
    # plt.plot(gpu_time)
    # plt.show()

    print(f"loss: {np.mean(loss_mean):.5f}")

if __name__ == "__main__":
    masks = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],         
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],         
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],         
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],         
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ]

    device = get_device()
    print(f"Using device: {device}")
    hidden_dim = 128
    env_latent_dim = 256
    num_epochs = 10

    model = CustomPlannerFlows(masks, hidden_dim, env_latent_dim).to(device)
    state = torch.load("../result/models/planner_flows_v4_best_loss.pth", map_location="cpu")
    model.load_state_dict(state)
    dataset = RLNFDataset(split="valid")
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False)

    visualize_samples(model, dataset, device, num_samples=1000)

    
