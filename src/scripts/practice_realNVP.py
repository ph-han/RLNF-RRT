import math
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles

from src.models.RealNVP import RealNVP


if __name__ == "__main__":
    masks = [[1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],         
            [0.0, 1.0],
            [1.0, 0.0],         
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0]]

    hidden_dim = 128

    realNVP = RealNVP(masks, hidden_dim)
    if torch.cuda.device_count():
        realNVP = realNVP.cuda()
    device = next(realNVP.parameters()).device

    optimizer = optim.Adam(realNVP.parameters(), lr = 0.001)
    num_steps = 5000

    for idx_step in range(num_steps):
        X, label = make_moons(n_samples = 512, noise = 0.05)
        X = torch.Tensor(X).to(device = device)

        z, logdet = realNVP.inverse(X)

        loss = torch.log(z.new_tensor([2*math.pi])) + torch.mean(torch.sum(0.5*z**2, -1) - logdet)
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()

        if (idx_step + 1) % 100 == 0:
            print(f"idx_steps: {idx_step:}, loss: {loss.item():.5f}")
            
    X, label = make_moons(n_samples = 1000, noise = 0.05)
    X = torch.Tensor(X).to(device = device)
    z, logdet_jacobian = realNVP.inverse(X)
    z = z.cpu().detach().numpy()

    X = X.cpu().detach().numpy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # --- 1행: Inference (Data X -> Latent Z) ---
    # 왼쪽: 원본 Moon 데이터셋
    axes[0, 0].scatter(X[label==0, 0], X[label==0, 1], s=5, alpha=0.5, label="Class 0")
    axes[0, 0].scatter(X[label==1, 0], X[label==1, 1], s=5, alpha=0.5, label="Class 1")
    axes[0, 0].set_title("X: Sampled from Moon dataset", fontsize=12)
    axes[0, 0].set_xlabel(r"$x_1$")
    axes[0, 0].set_ylabel(r"$x_2$")
    axes[0, 0].legend()

    # 오른쪽: 모델이 역변환(Inference)한 잠재 공간 Z [cite: 40]
    axes[0, 1].scatter(z[label==0, 0], z[label==0, 1], s=5, alpha=0.5, color='C0')
    axes[0, 1].scatter(z[label==1, 0], z[label==1, 1], s=5, alpha=0.5, color='C1')
    axes[0, 1].set_title("Z: Inferred from X (Gaussianized)", fontsize=12)
    axes[0, 1].set_xlabel(r"$z_1$")
    axes[0, 1].set_ylabel(r"$z_2$")
    axes[0, 1].set_xlim([-5, 5]) # Gaussian 분포 범위를 고려해 고정
    axes[0, 1].set_ylim([-5, 5])

    # --- 2행: Generation (Latent Z -> Data X) ---
    # 왼쪽: 표준 정규 분포에서 새로 샘플링한 Z [cite: 33]
    z_rand = torch.normal(0, 1, size=(1000, 2)).to(device)
    axes[1, 0].scatter(z_rand[:, 0].cpu(), z_rand[:, 1].cpu(), s=5, alpha=0.5, color='gray')
    axes[1, 0].set_title("Z: Sampled from Normal Distribution", fontsize=12)
    axes[1, 0].set_xlabel(r"$z_1$")
    axes[1, 0].set_ylabel(r"$z_2$")
    axes[1, 0].set_xlim([-5, 5])
    axes[1, 0].set_ylim([-5, 5])

    # 오른쪽: 모델이 생성(Forward)한 데이터 X [cite: 42]
    X_gen, _ = realNVP(z_rand)
    X_gen = X_gen.cpu().detach().numpy()
    axes[1, 1].scatter(X_gen[:, 0], X_gen[:, 1], s=5, alpha=0.6, color='purple')
    axes[1, 1].set_title("X: Generated from Z", fontsize=12)
    axes[1, 1].set_xlabel(r"$x_1$")
    axes[1, 1].set_ylabel(r"$x_2$")

    plt.tight_layout()
    plt.show()

