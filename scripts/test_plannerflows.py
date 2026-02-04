import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from rlnf_rrt.data_pipeline.utils import get_device
from rlnf_rrt.models.CustomPlannerFlows import CustomPlannerFlows
from rlnf_rrt.data_pipeline.custom_dataset import RLNFDataset



def visualize_samples(model, dataloader, device, num_samples=1000):
    model.eval()
    
    i = 0
    loss_mean = []
    scale_factor = 3.0 # 데이터셋 설정과 동일해야 함
    
    for batch in tqdm(dataloader):
        with torch.no_grad():
            map_img = batch['map'].to(device)
            start = batch['start'].to(device)
            goal = batch['goal'].to(device)
            
            # Forward: 가우시안 노이즈로부터 -3 ~ 3 범위 샘플 생성
            q_samples, ll = model.forward(map_img, start, goal, num_samples=num_samples)
            loss_mean.append(ll.mean().cpu())
            
        map_np = map_img.squeeze().cpu().numpy()
        
        # --- [중요] 좌표 변환 함수: 모델 범위(-3~3) -> 픽셀 범위(0~223) ---
        def to_pixel(val_tensor):
            val_np = val_tensor.squeeze().cpu().numpy()
            # 1. [-3, 3] -> [-1, 1]
            norm_val = val_np / scale_factor
            # 2. [-1, 1] -> [0, 1]
            zero_one_val = (norm_val + 1.0) / 2.0
            # 3. [0, 1] -> [0, 223]
            return zero_one_val * 223.0

        q_samples_np = to_pixel(q_samples)
        start_np = to_pixel(start)
        goal_np = to_pixel(goal)
        gt_np = to_pixel(batch['gt']) # batch['gt']가 이미 -3~3인 경우
        # -----------------------------------------------------------

        plt.figure(figsize=(8, 8))
        plt.imshow(map_np, cmap='gray', origin='lower') # origin 확인 필요 (보통 lower)
        
        # 샘플 및 GT 시각화
        plt.scatter(q_samples_np[:, 0], q_samples_np[:, 1], 
                    color='blue', s=2, label='Generated Samples')
        plt.scatter(gt_np[:, 0], gt_np[:, 1], color='orange', s=5, alpha=0.5, label='Ground Truth')
        
        # Start/Goal 시각화
        plt.scatter(start_np[0], start_np[1], color='green', s=150, marker='o', edgecolors='white', label='Start')
        plt.scatter(goal_np[0], goal_np[1], color='red', s=150, marker='X', edgecolors='white', label='Goal')
        
        plt.title(f"PlannerFlows Sampling (Index {i})")
        plt.legend(loc='upper right')
        
        # 결과 저장 폴더 확인
        save_dir = os.path.join(os.path.dirname(__file__), "..", "result", "visualization")
        os.makedirs(save_dir, exist_ok=True)
            
        plt.savefig(f"{save_dir}/res_{i}.png")
        plt.close()
        i += 1

    print(f"Average Log-Likelihood: {np.mean(loss_mean):.5f}")

if __name__ == "__main__":
    masks = [[1.0, 0.0], [0.0, 1.0]] * 32

    device = get_device()
    print(f"Using device: {device}")
    hidden_dim = 128
    env_latent_dim = 256
    num_epochs = 10

    model = CustomPlannerFlows(masks, hidden_dim, env_latent_dim).to(device)
    state = torch.load("../result/models/planner_flows_v1_ep1090.pth")
    model.load_state_dict(state)
    dataset = RLNFDataset(split="test")
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=1)

    visualize_samples(model, dataloader, device, num_samples=1000)

    
