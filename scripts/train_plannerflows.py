import os
import torch
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from rlnf_rrt.training.train import train_cnf
from rlnf_rrt.training.eval import eval_cnf
from rlnf_rrt.models.CustomPlannerFlows import CustomPlannerFlows
from rlnf_rrt.data_pipeline.custom_dataset import RLNFDataset
from rlnf_rrt.data_pipeline.utils import get_device

def plot_loss_curve(train_losses, val_losses, save_dir="../result/images", filename="loss_curve_v5.png"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='blue')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='red', linestyle='--')
    
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    masks = [[1.0, 0.0], [0.0, 1.0]] * 4

    device = get_device()
    hidden_dim = 128
    env_latent_dim = 256
    num_epochs = 1500
    batch_size=128
    num_workers=6

    model = CustomPlannerFlows(masks, hidden_dim, env_latent_dim).to(device)

    optimizer = Adam(model.parameters(), lr=1e-4)
    # scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    train_dataset = RLNFDataset(split="train")
    valid_dataset = RLNFDataset(split="valid")

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                             shuffle=True, num_workers=num_workers,
                                             pin_memory=True, persistent_workers=True)
    
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                                   shuffle=False, num_workers=num_workers,
                                                   pin_memory=True, persistent_workers=True)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        avg_loss = train_cnf(model, train_dataloader, optimizer, device, epoch, num_epochs)
        train_losses.append(avg_loss)
        avg_val_loss = eval_cnf(model, valid_dataloader, device, epoch, num_epochs)
        val_losses.append(avg_val_loss)
        
        print(f"[epoch {epoch}] avg loss : {avg_loss}")
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"../result/models/planner_flows_v7_ep{epoch+1}.pth")
            print(f"Model saved at epoch {epoch+1}")
            plot_loss_curve(train_losses, val_losses, filename=f"loss_curve_v7_ep{epoch+1}.png")

        if best_val_loss > avg_val_loss:
            torch.save(model.state_dict(), f"../result/models/planner_flows_v7_best_loss.pth")

    plot_loss_curve(train_losses, val_losses, filename="loss_curve_v7_final.png")
    print("Training finished.")
