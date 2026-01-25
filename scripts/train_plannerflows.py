import torch
from torch.optim import Adam

from rlnf_rrt.training.train import train_cnf
from rlnf_rrt.models.CustomPlannerFlows import CustomPlannerFlows
from rlnf_rrt.data_pipeline.custom_dataset import RLNFDataset
from rlnf_rrt.data_pipeline.utils import get_device

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
    env_latent_dim = 256
    num_epochs = 300
    batch_size=256
    num_workers=4

    model = CustomPlannerFlows(masks, hidden_dim, env_latent_dim).to(device)

    optimizer = Adam(model.parameters(), lr=5e-5)

    dataset = RLNFDataset(split="train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    for epoch in range(num_epochs):
        avg_loss = train_cnf(model, dataloader, optimizer, device, epoch, num_epochs)
        
        print(f"[epoch {epoch}] avg loss : {avg_loss}")
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"../result/models/planner_flows_v1_ep{epoch+1}.pth")
            print(f"Model saved at epoch {epoch+1}")
