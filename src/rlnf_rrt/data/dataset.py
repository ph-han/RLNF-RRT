from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from rlnf_rrt.utils.utils import load_cspace_img_to_np

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resample_points(path_xy: np.ndarray, target_points: int) -> np.ndarray:
    if len(path_xy) == target_points:
        return path_xy.astype(np.float32)
    if len(path_xy) == 0:
        return np.zeros((target_points, 2), dtype=np.float32)
    if len(path_xy) == 1:
        return np.repeat(path_xy.astype(np.float32), target_points, axis=0)

    diffs = np.diff(path_xy, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg_lens)))
    total = float(cum[-1])
    if total == 0.0:
        return np.repeat(path_xy[:1].astype(np.float32), target_points, axis=0)

    targets = np.linspace(0.0, total, target_points, dtype=np.float32)
    out = np.empty((target_points, 2), dtype=np.float32)
    for i, t in enumerate(targets):
        j = int(np.searchsorted(cum, t, side="right") - 1)
        j = max(0, min(j, len(seg_lens) - 1))
        seg_start = cum[j]
        seg_len = seg_lens[j]
        alpha = 0.0 if seg_len == 0 else float((t - seg_start) / seg_len)
        out[i] = path_xy[j] * (1.0 - alpha) + path_xy[j + 1] * alpha
    return out


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
        assert split in ["train", "val", "test"]
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

        # when gt trajectory is not normalized.
        if gt_path.size > 0 and (float(gt_path.max()) > 1.0 or float(gt_path.min()) < 0.0):
            gt_path[:, 0] = np.clip(gt_path[:, 0] / max(1, (w - 1)), 0.0, 1.0)
            gt_path[:, 1] = np.clip(gt_path[:, 1] / max(1, (h - 1)), 0.0, 1.0)

        target_points = int(self.num_points) if self.num_points is not None else int(row.get("num_points", len(gt_path)))
        gt_path = _resample_points(gt_path, target_points)

        if self.split == "train" and self.noise_std > 0:
            noise = np.random.normal(0.0, self.noise_std, gt_path.shape).astype(np.float32)
            gt_path = np.clip(gt_path + noise, 0.0, 1.0)

        return {
            "map": torch.from_numpy(map_data).float().unsqueeze(0),  # (1, H, W)
            "start": torch.from_numpy(start).float(),  # (2,)
            "goal": torch.from_numpy(goal).float(),  # (2,)
            "condition": torch.from_numpy(np.concatenate([start, goal], axis=0)).float(),  # (4,)
            "gt_path": torch.from_numpy(gt_path).float(),  # (N, 2)
        }
