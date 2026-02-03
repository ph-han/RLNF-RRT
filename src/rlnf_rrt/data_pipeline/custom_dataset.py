import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset

class RLNFDataset(Dataset):
    def __init__(
        self,
        dataset_root_path: str = "data",
        split: str = "train",
        gt_points_per_sample: int = 512,
        gt_noise_px: float = 2.0,          # <-- 추천: 0.8px는 얇을 수 있음. 일단 2.0부터
        gt_noise_trials: int = 10,
        free_thresh: float = 0.5,
        valid_deterministic: bool = True,  # <-- valid/test에서 랜덤 샘플링 고정
    ):
        assert split in ["train", "valid", "test"]
        self.split = split
        self.valid_deterministic = valid_deterministic

        PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
        dataset_path: str = f"{PROJECT_ROOT}/{dataset_root_path}/{split}"

        self.meta_data: pd.DataFrame = pd.read_csv(f"{dataset_path}/meta.csv")
        self.map_list: list[str] = []
        self.start_goal_list: list[str] = []
        self.gt_list: list[str] = []

        for _, row in tqdm(self.meta_data.iterrows(), total=len(self.meta_data)):
            if row["clearance"] == 1 and row["step_size"] == 1:
                self.map_list.append(f"{dataset_path}/map/{row['map_file']}")
                self.start_goal_list.append(f"{dataset_path}/start_goal/{row['start_goal_file']}")
                self.gt_list.append(f"{dataset_path}/gt_path/{row['gt_path_file']}")

        self.gt_points_per_sample = gt_points_per_sample
        self.gt_noise_px = float(gt_noise_px)
        self.gt_noise_trials = int(gt_noise_trials)
        self.free_thresh = float(free_thresh)

        # 데이터가 224x224라고 가정(너 코드와 동일)
        self.H = 224
        self.W = 224
        self.norm = float(self.W - 1)  # <-- 핵심: /224 말고 /(W-1)로 통일

        # 픽셀 단위 노이즈를 [0,1] 좌표 노이즈로 변환
        self.gt_sigma = self.gt_noise_px / self.norm

    def __len__(self):
        return len(self.map_list)

    def _is_free(self, map_np01: np.ndarray, x01: float, y01: float) -> bool:
        # x01,y01 in [0,1]
        if not (0.0 <= x01 <= 1.0 and 0.0 <= y01 <= 1.0):
            return False
        x = int(round(x01 * (self.W - 1)))
        y = int(round(y01 * (self.H - 1)))
        if x < 0 or x >= self.W or y < 0 or y >= self.H:
            return False

        # map이 흰색=free(1), 검정=obstacle(0)일 때:
        return map_np01[y, x] > self.free_thresh
        # 만약 네 맵이 반대로(흰색=obstacle)라면 위 줄을 아래로 바꿔:
        # return map_np01[y, x] < self.free_thresh

    def _tube_augment(self, gt01: np.ndarray, map_np01: np.ndarray, rng: np.random.RandomState | None = None) -> np.ndarray:
        """
        gt01: (N,2) in [0,1], 좌표는 (x,y)라고 가정
        map_np01: (H,W) in [0,1]
        rng: deterministic하게 만들고 싶으면 RandomState 넘기기
        """
        if rng is None:
            rng = np.random

        N = gt01.shape[0]
        out = np.empty_like(gt01)

        for i in range(N):
            p = gt01[i].astype(np.float32)
            ok = False
            for _ in range(self.gt_noise_trials):
                q = p + rng.randn(2).astype(np.float32) * self.gt_sigma

                # 핵심: clip하지 말고, 범위 밖이면 그냥 reject
                if self._is_free(map_np01, float(q[0]), float(q[1])):
                    out[i] = q
                    ok = True
                    break

            if not ok:
                out[i] = p  # fallback
        return out

    def __getitem__(self, idx: int):
        map_path = self.map_list[idx]
        start_goal_path = self.start_goal_list[idx]
        gt_path = self.gt_list[idx]

        # ------------------
        # map
        # ------------------
        map_img = Image.open(map_path).convert("L")
        map_np = np.array(map_img, dtype=np.float32) / 255.0  # (H,W) in [0,1]
        map_tensor = torch.from_numpy(map_np).unsqueeze(0).float()  # (1,H,W)

        # ------------------
        # start/goal  (픽셀 -> [0,1])
        # ------------------
        start_goal = np.load(start_goal_path).astype(np.float32) / self.norm  # <-- /(W-1)
        start = torch.from_numpy(start_goal[0]).float()
        goal  = torch.from_numpy(start_goal[1]).float()

        # ------------------
        # gt points (픽셀 -> [0,1])
        # ------------------
        gt_all = np.load(gt_path).astype(np.float32) / self.norm  # (M,2)

        M = gt_all.shape[0]
        if M == 0:
            # 방어 코드: 혹시 빈 gt가 있을 때
            gt = np.zeros((self.gt_points_per_sample, 2), dtype=np.float32)
            return {"map": map_tensor, "start": start, "goal": goal, "gt": torch.from_numpy(gt).float()}

        K = int(self.gt_points_per_sample)
        K1 = K // 3
        K2 = K - K1

        # valid/test의 흔들림을 줄이기 위해 deterministic하게 만들기
        if self.valid_deterministic and self.split != "train":
            rng = np.random.RandomState(idx)
        else:
            rng = np.random

        idx_uniform = np.linspace(0, M - 1, K1).astype(int)
        idx_random  = rng.choice(M, K2, replace=True)
        gt = gt_all[np.concatenate([idx_uniform, idx_random])].astype(np.float32)

        # ------------------
        # tube augmentation: train에만!
        # ------------------
        if self.split == "train":
            gt = self._tube_augment(gt, map_np, rng=rng)

        gt_points = torch.from_numpy(gt).float()  # (K,2)

        return {"map": map_tensor, "start": start, "goal": goal, "gt": gt_points}
