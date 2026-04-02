from __future__ import annotations

import numpy as np
from tqdm import tqdm
from rlnf_rrt.data.dataset import RLNFDataset

def extract_gt_subsegment(
    gt_path: np.ndarray,   # (N, 2)
    seg_start: np.ndarray, # (2,)
    seg_goal: np.ndarray,  # (2,)
) -> np.ndarray:
    if gt_path is None or len(gt_path) < 2:
        return np.stack([seg_start, seg_goal])
    i_start = int(np.argmin(np.linalg.norm(gt_path - seg_start, axis=1)))
    i_goal  = int(np.argmin(np.linalg.norm(gt_path - seg_goal,  axis=1)))
    if i_start > i_goal:
        i_start, i_goal = i_goal, i_start
    sub = gt_path[i_start : i_goal + 1]
    return sub if len(sub) >= 2 else np.stack([seg_start, seg_goal])

def segment_complexity(
    seg_state: tuple[np.ndarray, np.ndarray], # (seg_start, seg_goal)
    sdf_map: np.ndarray,                      # (H, W)
    gt_path: np.ndarray | None,               # (N, 2)
) -> float:
    seg_start, seg_goal = seg_state
    H, W = sdf_map.shape
    sdf_scale = float(max(H, W))
    straight = float(np.hypot((seg_goal[0] - seg_start[0]) * W, (seg_goal[1] - seg_start[1]) * H))

    if gt_path is not None and len(gt_path) >= 2:
        gt_sub = extract_gt_subsegment(gt_path, seg_start, seg_goal)
        diffs = np.diff(gt_sub, axis=0) * np.array([W, H])
        gt_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
        detour = gt_len / max(straight, 1e-6)
        detour_norm = float(np.clip((detour - 1.0) / 2.0, 0.0, 1.0))
        
        px = np.clip(np.round(gt_sub[:, 0] * (W - 1)).astype(int), 0, W - 1)
        py = np.clip(np.round(gt_sub[:, 1] * (H - 1)).astype(int), 0, H - 1)
        min_clr_px = float(sdf_map[py, px].min()) * sdf_scale
        clr_norm = float(np.clip(1.0 - min_clr_px / 20.0, 0.0, 1.0))
        return 0.7 * detour_norm + 0.3 * clr_norm
    else:
        line = np.linspace(seg_start, seg_goal, 20)
        px = np.clip(np.round(line[:, 0] * (W - 1)).astype(int), 0, W - 1)
        py = np.clip(np.round(line[:, 1] * (H - 1)).astype(int), 0, H - 1)
        min_sdf_px = float(sdf_map[py, px].min()) * sdf_scale
        return float(np.clip(1.0 - min_sdf_px / 20.0, 0.0, 1.0))

def precompute_difficulty(ds: RLNFDataset, **_kwargs) -> np.ndarray:
    n = len(ds)
    detour_ratios, min_clearances = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
    for idx in tqdm(range(n)):
        sample = ds[idx]
        gt_path, sdf, start, goal = sample["gt_path"].numpy(), sample["cond_image"][2].numpy(), sample["start"].numpy(), sample["goal"].numpy()
        H, W = sdf.shape
        diffs = np.diff(gt_path, axis=0)
        detour_ratios[idx] = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1]))) / max(float(np.hypot(goal[0] - start[0], goal[1] - start[1])), 1e-6)
        px, py = np.clip(np.round(gt_path[:, 0] * (W - 1)).astype(int), 0, W - 1), np.clip(np.round(gt_path[:, 1] * (H - 1)).astype(int), 0, H - 1)
        min_clearances[idx] = float((sdf[py, px] * max(H, W)).min())
    detour_norm, clearance_norm = np.clip((detour_ratios - 1.0) / 3.0, 0.0, 1.0), np.clip(1.0 - min_clearances / 20.0, 0.0, 1.0)
    difficulties = np.clip(0.7 * detour_norm + 0.3 * clearance_norm, 0.0, 1.0)
    return difficulties