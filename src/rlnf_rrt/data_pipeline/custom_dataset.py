import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset

class RLNFDataset(Dataset):
    def __init__(self, dataset_root_path: str="data", split: str="train",
                 gt_points_per_sample: int=512,
                 gt_noise_px: float=0.8,
                 gt_noise_trials: int=10,
                 free_thresh: float=0.5):
        assert split in ["train", "valid", "test"]
        PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
        dataset_path: str = f"{PROJECT_ROOT}/{dataset_root_path}/{split}"

        self.meta_data: pd.DataFrame = pd.read_csv(f"{dataset_path}/meta.csv")
        self.map_list: list[str] = []
        self.start_goal_list: list[str] = []
        self.gt_list: list[str] = []

        for _, row in tqdm(self.meta_data.iterrows(), total=len(self.meta_data)):
            if row['clearance'] == 1 and row['step_size'] == 1:
                self.map_list.append(f"{dataset_path}/map/{row['map_file']}")
                self.start_goal_list.append(f"{dataset_path}/start_goal/{row['start_goal_file']}")
                self.gt_list.append(f"{dataset_path}/gt_path/{row['gt_path_file']}")

        self.gt_points_per_sample = gt_points_per_sample
        self.gt_noise_px = gt_noise_px
        self.gt_noise_trials = gt_noise_trials
        self.free_thresh = free_thresh

        self.H = 224
        self.W = 224
        self.gt_sigma = gt_noise_px / 224.0  # normalize to [0,1] coordinates

    def __len__(self):
        return len(self.map_list)

    def _is_free(self, map_np01: np.ndarray, x01: float, y01: float) -> bool:
        # x01,y01 in [0,1] roughly
        x = int(round(x01 * (self.W - 1)))
        y = int(round(y01 * (self.H - 1)))
        if x < 0 or x >= self.W or y < 0 or y >= self.H:
            return False
        return map_np01[y, x] > self.free_thresh

    def _tube_augment(self, gt01: np.ndarray, map_np01: np.ndarray) -> np.ndarray:
        # gt01: (N,2) in [0,1]
        N = gt01.shape[0]
        out = np.empty_like(gt01)

        for i in range(N):
            p = gt01[i]
            ok = False
            for _ in range(self.gt_noise_trials):
                q = p + np.random.randn(2).astype(np.float32) * self.gt_sigma
                # optional: clip to [0,1] to avoid out-of-map
                q = np.clip(q, 0.0, 1.0)
                if self._is_free(map_np01, q[0], q[1]):
                    out[i] = q
                    ok = True
                    break
            if not ok:
                out[i] = p  # fallback: original point
        return out

    def __getitem__(self, idx):
        map_path = self.map_list[idx]
        start_goal_path = self.start_goal_list[idx]
        gt_path = self.gt_list[idx]

        # map
        map_img = Image.open(map_path).convert('L')
        map_np = np.array(map_img, dtype=np.float32) / 255.0  # (H,W) in [0,1]
        map_tensor = torch.from_numpy(map_np).unsqueeze(0).float()

        # start/goal
        start_goal = np.load(start_goal_path).astype(np.float32) / 224.0
        start = torch.from_numpy(start_goal[0])
        goal  = torch.from_numpy(start_goal[1])

        # gt points
        gt_all = np.load(gt_path).astype(np.float32) / 224.0  # (M,2) in [0,1]

        K = self.gt_points_per_sample
        # mix: uniform + random
        K1 = K // 3
        K2 = K - K1
        idx_uniform = np.linspace(0, gt_all.shape[0] - 1, K1).astype(int)
        idx_random  = np.random.choice(gt_all.shape[0], K2, replace=True)
        gt = gt_all[np.concatenate([idx_uniform, idx_random])]

        # tube augmentation (only train/valid? 보통 train만)
        if True:  # 혹은 (split == "train")
            gt = self._tube_augment(gt, map_np)

        gt_points = torch.from_numpy(gt).float()

        return {"map": map_tensor, "start": start, "goal": goal, "gt": gt_points}
