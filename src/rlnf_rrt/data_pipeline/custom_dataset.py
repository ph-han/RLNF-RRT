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
        gt_noise_px: float = 10.0,
        gt_noise_trials: int = 10,
        free_thresh: float = 0.5,
        valid_deterministic: bool = True,
        scale_factor: float = 1.0  # [-1, 1]에 곱할 값 (결과적으로 -1 ~ 1)
    ):
        assert split in ["train", "valid", "test"]
        self.split = split
        self.valid_deterministic = valid_deterministic
        self.scale_factor = scale_factor

        # 경로 설정 (사용자 환경에 맞게 조정 필요)
        PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
        dataset_path: str = f"{PROJECT_ROOT}/{dataset_root_path}/{split}"

        self.meta_data: pd.DataFrame = pd.read_csv(f"{dataset_path}/meta.csv")
        self.map_list: list[str] = []
        self.start_goal_list: list[str] = []
        self.gt_list: list[str] = []

        for _, row in tqdm(self.meta_data.iterrows(), total=len(self.meta_data), desc=f"Loading {split} metadata"):
            if row["clearance"] in [1, 2] and row["step_size"] in [1, 2, 4, 6]:
                self.map_list.append(f"{dataset_path}/map/{row['map_file']}")
                self.start_goal_list.append(f"{dataset_path}/start_goal/{row['start_goal_file']}")
                self.gt_list.append(f"{dataset_path}/gt_path/{row['gt_path_file']}")

        self.gt_points_per_sample = gt_points_per_sample
        self.gt_noise_px = float(gt_noise_px)
        self.gt_noise_trials = int(gt_noise_trials)
        self.free_thresh = float(free_thresh)

        self.H = 224
        self.W = 224
        self.norm = float(self.W - 1)

        # [-1, 1] 스케일에 맞춘 노이즈 표준편차 계산
        # (픽셀노이즈 / 전체픽셀) * 2.0(범위폭) * scale_factor(스케일)
        self.gt_sigma = (self.gt_noise_px / self.norm) * 2.0 * self.scale_factor

    def _to_custom_range(self, val01: np.ndarray) -> np.ndarray:
        return (val01 * 2.0 - 1.0) * self.scale_factor

    def _to_zero_one(self, val_scaled: float) -> float:
        """[-scale, scale] 범위를 다시 [0, 1]로 복구 (맵 체크용)"""
        return ((val_scaled / self.scale_factor) + 1.0) / 2.0

    def scaled_to_pixel(self, points_scaled: np.ndarray) -> np.ndarray:
        """[-scale, scale] 좌표를 픽셀 좌표로 변환 (시각화용)."""
        pts = np.asarray(points_scaled, dtype=np.float32).reshape(-1, 2)
        pts01 = ((pts / self.scale_factor) + 1.0) / 2.0
        xy = np.empty_like(pts01)
        xy[:, 0] = pts01[:, 0] * (self.W - 1)
        xy[:, 1] = pts01[:, 1] * (self.H - 1)
        return xy

    def _is_free(self, map_np01: np.ndarray, x_scaled: float, y_scaled: float) -> bool:
        # 스케일된 좌표를 다시 [0, 1]로 돌려서 픽셀 위치 확인
        x01 = self._to_zero_one(x_scaled)
        y01 = self._to_zero_one(y_scaled)
        
        if not (0.0 <= x01 <= 1.0 and 0.0 <= y01 <= 1.0):
            return False
            
        x = int(round(x01 * (self.W - 1)))
        y = int(round(y01 * (self.H - 1)))
        
        if x < 0 or x >= self.W or y < 0 or y >= self.H:
            return False

        # map: 흰색(1)=통과가능, 검정(0)=장애물
        return map_np01[y, x] > self.free_thresh

    def _tube_augment(self, gt_scaled: np.ndarray, map_np01: np.ndarray, rng: np.random.RandomState | None = None) -> np.ndarray:
        if rng is None:
            rng = np.random

        N = gt_scaled.shape[0]
        out = np.empty_like(gt_scaled)

        for i in range(N):
            p = gt_scaled[i].astype(np.float32)
            ok = False
            for _ in range(self.gt_noise_trials):
                q = p + rng.randn(2).astype(np.float32) * self.gt_sigma

                if self._is_free(map_np01, float(q[0]), float(q[1])):
                    out[i] = q
                    ok = True
                    break

            if not ok:
                out[i] = p  # 실패 시 원본 유지
        return out

    def _create_gaussian_heatmap(self, center_x: float, center_y: float, sigma: float = 4.0) -> np.ndarray:
        """
        Create a 2D Gaussian heatmap centered at (center_x, center_y) in pixel coordinates.
        
        Args:
            center_x: X coordinate in pixels (0 to W-1)
            center_y: Y coordinate in pixels (0 to H-1)
            sigma: Standard deviation of the Gaussian in pixels
            
        Returns:
            Heatmap of shape (H, W) with values in [0, 1]
        """
        x = np.arange(0, self.W, dtype=np.float32)
        y = np.arange(0, self.H, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        
        # Compute Gaussian
        heatmap = np.exp(-((xx - center_x)**2 + (yy - center_y)**2) / (2 * sigma**2))
        return heatmap.astype(np.float32)

    def __len__(self):
        return len(self.map_list)

    def __getitem__(self, idx: int):
        map_path = self.map_list[idx]
        start_goal_path = self.start_goal_list[idx]
        gt_path = self.gt_list[idx]

        # Map 로드
        map_img = Image.open(map_path).convert("L")
        map_np = np.array(map_img, dtype=np.float32) / 255.0

        # Start/Goal 로드 (pixel coordinates)
        start_goal_px = np.load(start_goal_path).astype(np.float32)  # shape: (2, 2) - [[start_x, start_y], [goal_x, goal_y]]
        start_px = start_goal_px[0]
        goal_px = start_goal_px[1]
        
        # Create Gaussian heatmaps for start and goal (in pixel coordinates)
        start_heatmap = self._create_gaussian_heatmap(start_px[0], start_px[1], sigma=4.0)
        goal_heatmap = self._create_gaussian_heatmap(goal_px[0], goal_px[1], sigma=4.0)
        
        # Stack into 3-channel tensor: (occupancy, start_heatmap, goal_heatmap)
        map_3ch = np.stack([map_np, start_heatmap, goal_heatmap], axis=0)
        map_tensor = torch.from_numpy(map_3ch).float()

        # Start/Goal 스케일 변환 for conditioning vector
        start_goal01 = start_goal_px / self.norm
        start_goal_scaled = self._to_custom_range(start_goal01)
        start = torch.from_numpy(start_goal_scaled[0]).float()
        goal  = torch.from_numpy(start_goal_scaled[1]).float()

        # GT Path 로드 및 스케일 변환
        gt_all01 = np.load(gt_path).astype(np.float32) / self.norm
        gt_all_scaled = self._to_custom_range(gt_all01)

        M = gt_all_scaled.shape[0]
        if M == 0:
            gt = np.zeros((self.gt_points_per_sample, 2), dtype=np.float32)
            return {"map": map_tensor, "start": start, "goal": goal, "gt": torch.from_numpy(gt).float()}

        # 샘플링 개수 맞추기
        K = int(self.gt_points_per_sample)
        K1 = K // 3
        K2 = K - K1

        if self.valid_deterministic and self.split != "train":
            rng = np.random.RandomState(idx)
        else:
            rng = np.random

        idx_uniform = np.linspace(0, M - 1, K1).astype(int)
        idx_random  = rng.choice(M, K2, replace=True)
        gt = gt_all_scaled[np.concatenate([idx_uniform, idx_random])].astype(np.float32)

        # Train 데이터인 경우 Tube Augmentation 적용
        # if self.split == "train":
        gt = self._tube_augment(gt, map_np, rng=rng)

        gt_points = torch.from_numpy(gt).float()

        return {
            "map": map_tensor, 
            "start": start, 
            "goal": goal, 
            "gt": gt_points
        }

