import torch
from tqdm import tqdm

def train_cnf(model, dataloader:torch.utils.data.DataLoader, optimizer:torch.optim.Optimizer, device:torch.device, epoch:int, num_epochs:int):
    model.train()
    loop = tqdm(
            dataloader,
            leave=False
        )
    
    running_loss = 0.0
    for step, batch in enumerate(loop):
        condition_map = batch['map'].to(device)
        condition_start = batch['start'].to(device)
        condition_goal = batch['goal'].to(device)
        gt = batch['gt'].to(device)


        optimizer.zero_grad()
        loss = model.inverse(gt, condition_map, condition_start, condition_goal)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_description(f"Epoch {epoch+1}/{num_epochs} | Step [{step+1}/{len(dataloader)}]")
        loop.set_postfix(loss=loss.item())

    return running_loss / len(dataloader)