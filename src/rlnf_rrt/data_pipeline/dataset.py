from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]

import torch
from torch.utils.data import Dataset

from rlnf_rrt.utils.utils import load_cspace_img_to_np

class RLNFDataset(Dataset):
    def __init__(self, split:str="train", noise_std:float=0.0):
        assert split in ["train", "valid", "test"]
        self.split:str = split
        self.noise_std:float = noise_std
        self.data_path:str = f"{PROJECT_ROOT}/data/{split}"
        self.meta_data:pd.DataFrame = pd.read_csv(f"{self.data_path}/meta.csv")
        
        self.map_list: list[str] = []
        self.start_goal_list: list[str] = []
        self.gt_list: list[str] = []

        # experiment: only use clearance 1 and step size 1
        self.filtered_meta = self.meta_data[  
            (self.meta_data["clearance"] == 1) &
            (self.meta_data["step_size"] == 1)
        ]

        for _, row in tqdm(self.filtered_meta.iterrows(), total=len(self.filtered_meta), desc=f"Loading {split} dataset"):
            self.map_list.append(f"{self.data_path}/map/{row['map_file']}")
            self.start_goal_list.append(f"{self.data_path}/start_goal/{row['start_goal_file']}")
            self.gt_list.append(f"{self.data_path}/gt_path/{row['gt_path_file']}")
            
    
    def __len__(self):
        return len(self.filtered_meta)
    
    def __getitem__(self, idx: int):
        map_path = self.map_list[idx]
        start_goal_path = self.start_goal_list[idx]
        gt_path = self.gt_list[idx]

        # load map data (H, W) [255: free, 0: obstacle]
        map_data = load_cspace_img_to_np(map_path)
        
        # load start and goal data [(x, y), (x, y)]
        start_goal_data = np.load(start_goal_path)
        start_data = start_goal_data[0]
        goal_data = start_goal_data[1]

        # load gt path [(x, y), (x, y), ...]
        gt_path_data = np.load(gt_path)

        # normalize map data (0 ~ 1)
        map_data = map_data / 255.0

        # normalize start and goal data (0 ~ 1)
        start_data = start_data / map_data.shape[0]
        goal_data = goal_data / map_data.shape[0]

        # normalize gt path data (0 ~ 1)
        gt_path_data = gt_path_data / map_data.shape[0]

        # gt path sampling (512 points)
        is_train = (self.split == "train")

        if len(gt_path_data) >= 512:
            indices = np.linspace(0, len(gt_path_data) - 1, 512).astype(int)
            path_512 = gt_path_data[indices]
        else:
            if is_train:
                indices = np.random.choice(len(gt_path_data), 512, replace=True)
                path_512 = gt_path_data[indices]
            else:
                reps = int(np.ceil(512 / len(gt_path_data)))
                path_512 = np.tile(gt_path_data, (reps, 1))[:512]

        # add noise to gt path
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, path_512.shape)
            path_512 = path_512 + noise

        return {
            "map": torch.from_numpy(map_data).float().unsqueeze(0),
            "start": torch.from_numpy(start_data).float(),
            "goal": torch.from_numpy(goal_data).float(),
            "gt_path": torch.from_numpy(path_512).float()
        }
