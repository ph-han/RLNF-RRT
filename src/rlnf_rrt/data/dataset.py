from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import torch
from torch.utils.data import Dataset, Sampler

from rlnf_rrt.utils.utils import load_cspace_img_to_np

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resample_points(path_xy: np.ndarray, target_points: int) -> np.ndarray:
    path_xy = np.asarray(path_xy, dtype=np.float32).reshape(-1, 2)

    if target_points <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    if len(path_xy) == target_points:
        return path_xy.copy()
    if len(path_xy) == 0:
        return np.zeros((target_points, 2), dtype=np.float32)
    if len(path_xy) == 1:
        return np.repeat(path_xy, target_points, axis=0)

    # Distance coordinate for each original point along the polyline.
    seg_lens = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    dist = np.concatenate(([0.0], np.cumsum(seg_lens))).astype(np.float32)
    total = float(dist[-1])
    if total <= 0.0:
        return np.repeat(path_xy[:1], target_points, axis=0)

    # Uniform distance targets, then interpolate x/y independently.
    target_dist = np.linspace(0.0, total, target_points, dtype=np.float32)
    out_x = np.interp(target_dist, dist, path_xy[:, 0]).astype(np.float32)
    out_y = np.interp(target_dist, dist, path_xy[:, 1]).astype(np.float32)
    return np.stack([out_x, out_y], axis=1)


class DifficultyBatchSampler(Sampler[list[int]]):
    """난이도 기반 Stratified Batch Sampler.

    각 배치를 easy/medium/hard 비율에 맞춰 구성.
    학습 진행도(progress)에 따라 비율을 자동 조정 (curriculum).

    Args:
        difficulties: 샘플별 난이도 [0, 1] 배열.
        batch_size: 배치 크기.
        total_batches: 전체 배치 수 (epoch 개념 대신 총 iteration 기반).
        easy_thresh: easy/medium 경계 (미만이면 easy).
        hard_thresh: medium/hard 경계 (이상이면 hard).
        warmup_ratio: 전체 진행도 중 warmup 비율 (medium 위주).
        transition_ratio: warmup 이후 easy+medium → 전체 전환 비율.
    """

    def __init__(
        self,
        difficulties: np.ndarray,
        batch_size: int,
        total_batches: int,
        easy_thresh: float = 0.3,
        hard_thresh: float = 0.7,
        warmup_ratio: float = 0.1,
        transition_ratio: float = 0.3,
    ) -> None:
        self.difficulties = np.asarray(difficulties, dtype=np.float32)
        self.batch_size = batch_size
        self.total_batches = total_batches
        self.warmup_ratio = warmup_ratio
        self.transition_ratio = transition_ratio

        # 난이도 그룹 인덱스
        self.easy_ids = np.where(self.difficulties < easy_thresh)[0]
        self.med_ids = np.where(
            (self.difficulties >= easy_thresh) & (self.difficulties < hard_thresh)
        )[0]
        self.hard_ids = np.where(self.difficulties >= hard_thresh)[0]

        # 빈 그룹 fallback
        all_ids = np.arange(len(self.difficulties))
        if len(self.easy_ids) == 0:
            self.easy_ids = all_ids
        if len(self.med_ids) == 0:
            self.med_ids = all_ids
        if len(self.hard_ids) == 0:
            self.hard_ids = all_ids

        self._batch_idx = 0

    def _get_ratios(self, progress: float) -> tuple[float, float, float]:
        """학습 진행도에 따른 (easy, medium, hard) 비율 반환."""
        if progress < self.warmup_ratio:
            # warmup: medium 위주 + easy 약간
            return (0.2, 0.6, 0.2)
        elif progress < self.transition_ratio:
            # transition: easy + medium 중심, hard 점진 증가
            t = (progress - self.warmup_ratio) / max(self.transition_ratio - self.warmup_ratio, 1e-6)
            hard_r = 0.2 + 0.15 * t  # 0.2 → 0.35
            easy_r = 0.3 - 0.05 * t  # 0.3 → 0.25
            med_r = 1.0 - easy_r - hard_r
            return (easy_r, med_r, hard_r)
        else:
            # main phase: hard 비중 점진 증가
            t = (progress - self.transition_ratio) / max(1.0 - self.transition_ratio, 1e-6)
            hard_r = 0.35 + 0.15 * t  # 0.35 → 0.50
            easy_r = max(0.25 - 0.15 * t, 0.10)  # 0.25 → 0.10
            med_r = 1.0 - easy_r - hard_r
            return (easy_r, med_r, hard_r)

    def __iter__(self):
        for i in range(self.total_batches):
            progress = i / max(self.total_batches, 1)
            easy_r, med_r, hard_r = self._get_ratios(progress)

            n_easy = max(int(round(self.batch_size * easy_r)), 1)
            n_hard = max(int(round(self.batch_size * hard_r)), 1)
            n_med = max(self.batch_size - n_easy - n_hard, 1)

            # 총합 보정
            total = n_easy + n_med + n_hard
            if total > self.batch_size:
                n_med = max(self.batch_size - n_easy - n_hard, 0)
            elif total < self.batch_size:
                n_med += self.batch_size - total

            batch = []
            batch.extend(np.random.choice(self.easy_ids, size=n_easy, replace=True).tolist())
            batch.extend(np.random.choice(self.med_ids, size=n_med, replace=True).tolist())
            batch.extend(np.random.choice(self.hard_ids, size=n_hard, replace=True).tolist())

            np.random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.total_batches


class RLNFDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        noise_std: float = 0.0,
        num_points: int | None = None,
        clearance: int | None = None,
        step_size: int | None = None,
        data_root: str | Path | None = None,
    ):
        assert split in ["train", "val", "test", "test_circle"]
        self.split = split
        self.noise_std = float(noise_std)
        self.num_points = num_points

        root = Path(data_root) if data_root is not None else PROJECT_ROOT / "data"
        self.data_path = root / split
        self.meta_data = pd.read_csv(self.data_path / "meta.csv")

        if clearance is not None:
            self.meta_data = self.meta_data[self.meta_data["clearance"] == clearance]
        if step_size is not None:
            self.meta_data = self.meta_data[self.meta_data["step_size"] == step_size]

        self.meta_data = self.meta_data.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.meta_data)

    def __getitem__(self, idx: int):
        row = self.meta_data.iloc[idx]
        map_path = self.data_path / "map" / row["map_file"]
        start_goal_path = self.data_path / "start_goal" / row["start_goal_file"]
        gt_path_path = self.data_path / "gt_path" / row["gt_path_file"]

        # (H, W) uint8 with free=255, obstacle=0 -> float32 [0,1]
        map_data = load_cspace_img_to_np(str(map_path)).astype(np.float32) / 255.0

        start_goal = np.load(start_goal_path).astype(np.float32)  # pixel coords, shape (2, 2)
        gt_path = np.load(gt_path_path).astype(np.float32)  # normalized [0,1] from generator

        h, w = map_data.shape
        start_goal[:, 0] = np.clip(start_goal[:, 0] / max(1, (w - 1)), 0.0, 1.0)
        start_goal[:, 1] = np.clip(start_goal[:, 1] / max(1, (h - 1)), 0.0, 1.0)
        start = start_goal[0]
        goal = start_goal[1]

        target_points = int(self.num_points) if self.num_points is not None else int(row.get("num_points", len(gt_path)))
        gt_path = _resample_points(gt_path, target_points)

        if self.split == "train" and self.noise_std > 0:
            noise = np.random.normal(0.0, self.noise_std, gt_path.shape).astype(np.float32)
            gt_path = np.clip(gt_path + noise, 0.0, 1.0)

        # Build 3-channel input:
        # channel 0: binary map
        # channel 1: start/goal channel (start +1, goal -1)
        # channel 2: signed distance field (sdf)
        
        map_np = (map_data > 0.5).astype(np.uint8)
        norm = float(max(h, w))
        
        free = map_np.astype(bool)
        obstacle = ~free
        dist_to_obstacle = ndi.distance_transform_edt(free)
        dist_to_free = ndi.distance_transform_edt(obstacle)
        sdf = (dist_to_obstacle - dist_to_free) / max(norm, 1.0)
        sdf = np.clip(sdf, -1.0, 1.0).astype(np.float32)
        
        sg_channel = np.zeros((h, w), dtype=np.float32)
        sx = int(np.clip(round(start[0] * (w - 1)), 0, w - 1))
        sy = int(np.clip(round(start[1] * (h - 1)), 0, h - 1))
        gx = int(np.clip(round(goal[0] * (w - 1)), 0, w - 1))
        gy = int(np.clip(round(goal[1] * (h - 1)), 0, h - 1))
        sg_channel[sy, sx] = 1.0
        sg_channel[gy, gx] = -1.0
        
        cond_image = np.stack([map_data, sg_channel, sdf], axis=0)

        return {
            "cond_image": torch.from_numpy(cond_image).float(),  # (3, H, W)
            "start": torch.from_numpy(start).float(),  # (2,)
            "goal": torch.from_numpy(goal).float(),  # (2,)
            "gt_path": torch.from_numpy(gt_path).float(),  # (N, 2)
        }
