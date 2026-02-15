from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import cv2
import numpy as np

from rlnf_rrt.utils.astar import AStar  # type: ignore


def generate_2d_grid_map(width: int, height: int) -> np.ndarray:
    grid_map: np.ndarray = np.ones((height, width), dtype=np.uint8)

    grid_map[0, :] = 0
    grid_map[-1, :] = 0
    grid_map[:, 0] = 0
    grid_map[:, -1] = 0

    obs_num: int = random.randint(3, 30)
    for _ in range(obs_num):
        obs_width: int = random.randint(5, 60)
        obs_height: int = random.randint(5, 60)
        obs_x: int = random.randint(1, width - obs_width - 1)
        obs_y: int = random.randint(1, height - obs_height - 1)
        grid_map[obs_y : obs_y + obs_height, obs_x : obs_x + obs_width] = 0

    return grid_map


def generate_2d_start_goal(map_info: np.ndarray, dist: float = 15.0, max_tries: int = 10_000) -> np.ndarray:
    free = np.argwhere(map_info != 0)
    if free.shape[0] < 2:
        raise ValueError("Not enough free cells to sample start/goal.")

    for _ in range(max_tries):
        sy, sx = free[np.random.randint(len(free))]
        gy, gx = free[np.random.randint(len(free))]

        if (sx - gx) ** 2 + (sy - gy) ** 2 >= dist**2:
            return np.array([(sx, sy), (gx, gy)], dtype=int)

    raise RuntimeError(f"Failed to sample start/goal with dist>={dist} in {max_tries} tries.")


def _is_free_with_clearance(map_info: np.ndarray, x: int, y: int, clearance: int) -> bool:
    height, width = map_info.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        return False

    x_min, x_max = x - clearance, x + clearance + 1
    y_min, y_max = y - clearance, y + clearance + 1
    if x_min < 0 or x_max > width or y_min < 0 or y_max > height:
        return False

    return not np.any(map_info[y_min:y_max, x_min:x_max] == 0)

def generate_2d_start_goal_clearance(
    map_info: np.ndarray,
    dist: float = 15.0,
    clearance: int = 2,
    max_tries: int = 10_000,
) -> np.ndarray:
    free = np.argwhere(map_info != 0)
    if free.shape[0] < 2:
        raise ValueError("Not enough free cells to sample start/goal.")

    for _ in range(max_tries):
        sy, sx = free[np.random.randint(len(free))]
        gy, gx = free[np.random.randint(len(free))]

        if not _is_free_with_clearance(map_info, int(sx), int(sy), clearance):
            continue
        if not _is_free_with_clearance(map_info, int(gx), int(gy), clearance):
            continue

        if (sx - gx) ** 2 + (sy - gy) ** 2 >= dist**2:
            return np.array([(sx, sy), (gx, gy)], dtype=int)

    raise RuntimeError(
        f"Failed to sample clearance-valid start/goal with dist>={dist} in {max_tries} tries."
    )


def _resample_polyline(polyline_xy: np.ndarray, num_points: int) -> np.ndarray:
    if polyline_xy.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)
    if polyline_xy.shape[0] == 1:
        return np.repeat(polyline_xy.astype(np.float32), num_points, axis=0)

    diffs = np.diff(polyline_xy, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg_lens)))
    total_len = float(cum[-1])

    if total_len == 0.0:
        return np.repeat(polyline_xy[:1].astype(np.float32), num_points, axis=0)

    targets = np.linspace(0.0, total_len, num_points, dtype=np.float32)
    out = np.empty((num_points, 2), dtype=np.float32)

    for i, t in enumerate(targets):
        seg_idx = int(np.searchsorted(cum, t, side="right") - 1)
        seg_idx = max(0, min(seg_idx, len(seg_lens) - 1))
        seg_start = cum[seg_idx]
        seg_len = seg_lens[seg_idx]
        alpha = 0.0 if seg_len == 0 else float((t - seg_start) / seg_len)
        out[i] = polyline_xy[seg_idx] * (1.0 - alpha) + polyline_xy[seg_idx + 1] * alpha

    return out


def generate_2d_gt_path(
    map_info: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    clr: int = 1,
    ss: int = 1,
    num_points: int = 128,
) -> np.ndarray:
    astar = AStar(map_info, clr, ss)
    is_success = astar.planning(start[0], start[1], goal[0], goal[1])
    if not is_success:
        return np.empty((0, 2), dtype=np.float32)

    opt_path: list[tuple[int, int]] = astar.get_final_path()
    if len(opt_path) == 0:
        return np.empty((0, 2), dtype=np.float32)

    path_xy = np.array(opt_path, dtype=np.float32)
    # Remove duplicated consecutive nodes to keep arc-length resampling stable.
    keep = np.ones(len(path_xy), dtype=bool)
    keep[1:] = np.any(path_xy[1:] != path_xy[:-1], axis=1)
    path_xy = path_xy[keep]

    if len(path_xy) == 0:
        return np.empty((0, 2), dtype=np.float32)

    resampled_xy = _resample_polyline(path_xy, num_points=num_points)

    width = map_info.shape[1]
    height = map_info.shape[0]
    denom_x = max(1, width - 1)
    denom_y = max(1, height - 1)
    resampled_xy[:, 0] = np.clip(resampled_xy[:, 0] / denom_x, 0.0, 1.0)
    resampled_xy[:, 1] = np.clip(resampled_xy[:, 1] / denom_y, 0.0, 1.0)
    return resampled_xy.astype(np.float32)


def generate_2d_dataset(
    num_of_map_data: int = 100,
    num_of_start_goal: int = 10,
    split: str = "train",
    data_root: str | Path | None = None,
    width: int = 224,
    height: int = 224,
    num_points: int = 128,
    clearance: int = 2,
    step_size: int = 1,
    min_start_goal_dist: int = 15,
    max_start_goal_dist: int = 100,
    seed: int | None = None,
) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if data_root is None:
        repo_root = Path(__file__).resolve().parents[3]
        data_root = repo_root / "data"
    else:
        data_root = Path(data_root)

    base_path = Path(data_root) / split
    for sub_dir in ["map", "start_goal", "gt_path"]:
        os.makedirs(base_path / sub_dir, exist_ok=True)

    meta_file_path = base_path / "meta.csv"
    with open(meta_file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "map_file",
                "start_goal_file",
                "gt_path_file",
                "clearance",
                "step_size",
                "num_points",
                "width",
                "height",
                "start_x",
                "start_y",
                "goal_x",
                "goal_y",
            ]
        )

        data_id = 0
        for i in range(num_of_map_data):
            grid_map = generate_2d_grid_map(width, height)
            map_filename = f"map2d_{i:06d}.png"
            cv2.imwrite(str(base_path / "map" / map_filename), grid_map * 255)

            valid_pairs = 0
            attempts = 0
            max_attempts = max(100, num_of_start_goal * 20)

            while valid_pairs < num_of_start_goal and attempts < max_attempts:
                attempts += 1
                start_goal_dist = random.randint(min_start_goal_dist, max_start_goal_dist)

                try:
                    start_goal = generate_2d_start_goal_clearance(
                        grid_map,
                        dist=start_goal_dist,
                        clearance=clearance,
                        max_tries=2000,
                    )
                except RuntimeError:
                    continue

                gt_path = generate_2d_gt_path(
                    grid_map,
                    tuple(start_goal[0]),
                    tuple(start_goal[1]),
                    clr=clearance,
                    ss=step_size,
                    num_points=num_points,
                )
                if len(gt_path) == 0:
                    continue

                sg_filename = f"start_goal_2d_{data_id:07d}.npy"
                path_filename = f"path_2d_{data_id:07d}.npy"
                np.save(base_path / "start_goal" / sg_filename, start_goal)
                np.save(base_path / "gt_path" / path_filename, gt_path)

                writer.writerow(
                    [
                        data_id,
                        map_filename,
                        sg_filename,
                        path_filename,
                        clearance,
                        step_size,
                        num_points,
                        width,
                        height,
                        int(start_goal[0][0]),
                        int(start_goal[0][1]),
                        int(start_goal[1][0]),
                        int(start_goal[1][1]),
                    ]
                )
                data_id += 1
                valid_pairs += 1

            print(f"Progress: Map {i + 1}/{num_of_map_data} done.")
            if valid_pairs < num_of_start_goal:
                print(
                    f"  Warning: only {valid_pairs}/{num_of_start_goal} valid pairs generated "
                    f"(attempts={attempts})."
                )

    print(f"Dataset generation complete. Meta data saved to {meta_file_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate 2D coordinate GT dataset for RLNF-RRT")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--num-maps", type=int, default=100)
    parser.add_argument("--num-start-goal", type=int, default=10)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--num-points", type=int, default=128)
    parser.add_argument("--clearance", type=int, default=2)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--min-start-goal-dist", type=int, default=15)
    parser.add_argument("--max-start-goal-dist", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    generate_2d_dataset(
        num_of_map_data=args.num_maps,
        num_of_start_goal=args.num_start_goal,
        split=args.split,
        data_root=args.data_root,
        width=args.width,
        height=args.height,
        num_points=args.num_points,
        clearance=args.clearance,
        step_size=args.step_size,
        min_start_goal_dist=args.min_start_goal_dist,
        max_start_goal_dist=args.max_start_goal_dist,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
