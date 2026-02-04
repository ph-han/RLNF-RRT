import torch
from tqdm import tqdm

def eval_cnf(model, dataloader, device, epoch, num_epochs):
    model.eval()
    
    running_loss = 0.0
    
    with torch.no_grad():
        loop = tqdm(
            dataloader,
            leave=False,
            desc=f"Eval Epoch {epoch+1}/{num_epochs}"
        )
        
        for batch in loop:
            condition_map = batch['map'].to(device)
            condition_start = batch['start'].to(device)
            condition_goal = batch['goal'].to(device)
            gt = batch['gt'].to(device)
            loss = model.nll(gt, condition_map, condition_start, condition_goal)

            running_loss += loss.item()
            loop.set_postfix(val_loss=loss.item())

    return running_loss / len(dataloader)