import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset

class RLNFDataset(Dataset):
    def __init__(self, dataset_root_path:str="dataset", split:str="train"):
        assert split in ["train", "valid", "test"]
        PROJECT_ROOT:Path = Path(__file__).resolve().parents[1]
        dataset_path:str = f"{PROJECT_ROOT}/{dataset_root_path}/{split}"

        self.meta_data:pd.DataFrame = pd.read_csv(f"{dataset_path}/meta_data.csv")
        
        self.map_list:list[str] = []
        self.start_goal_list:list[str] = []
        self.gt_list:list[str] = []
        # self.gt_clr_list = []
        # self.gt_ss_list = []

        for row in tqdm(self.meta_data.iterrows(), total=len(self.meta_data)):
            self.map_list.append(f"{dataset_path}/map/{row[1]['map']}")
            self.start_goal_list.append(f"{dataset_path}/start_goal/{row[1]['start_goal']}")
            self.gt_list.append(f"{dataset_path}/gt_path/{row[1]['gt']}")
    
    def __len__(self):
        return len(self.meta_data)
    
    def __getitem__(self, idx):
        map_path:str = self.map_list[idx]
        start_goal_path:str = self.start_goal_list[idx]
        gt_path:str = self.gt_list[idx]

        map_img:Image = Image.open(map_path).convert('L')
        map_np:np.ndarray = np.array(map_img, dtype=np.float32) / 255.0

        map_tensor:torch.Tensor = torch.from_numpy(map_np).unsqueeze(0).float()

        start_goal:torch.Tensor = torch.from_numpy(np.load(start_goal_path)).float()
        start, goal = start_goal[0], start_goal[1]

        gt:torch.Tensor = torch.from_numpy(np.load(gt_path)).float()
        
        return {
            "map": map_tensor,
            "start": start,
            "goal": goal,
            "gt": gt
        }