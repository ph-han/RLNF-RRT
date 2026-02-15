import argparse
from pathlib import Path

import cv2
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rlnf_rrt.data.dataset import RLNFDataset


def draw_sample(sample: dict[str, np.ndarray | object], idx: int, total: int) -> np.ndarray:
    map_tensor = sample["map"]
    start_tensor = sample["start"]
    goal_tensor = sample["goal"]
    path_tensor = sample["gt_path"]

    map_np = map_tensor.squeeze(0).numpy()
    start = start_tensor.numpy()
    goal = goal_tensor.numpy()
    path = path_tensor.numpy()

    h, w = map_np.shape
    canvas = cv2.cvtColor((map_np * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    if len(path) > 0:
        px = np.clip(path[:, 0] * (w - 1), 0, w - 1).astype(np.int32)
        py = np.clip(path[:, 1] * (h - 1), 0, h - 1).astype(np.int32)

        overlay = canvas.copy()
        for x, y in zip(px, py):
            cv2.circle(overlay, (int(x), int(y)), 1, (0, 255, 255), -1)
        canvas = cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0.0)

    sx = int(np.clip(start[0] * (w - 1), 0, w - 1))
    sy = int(np.clip(start[1] * (h - 1), 0, h - 1))
    gx = int(np.clip(goal[0] * (w - 1), 0, w - 1))
    gy = int(np.clip(goal[1] * (h - 1), 0, h - 1))
    cv2.circle(canvas, (sx, sy), 3, (0, 0, 255), -1)
    cv2.circle(canvas, (gx, gy), 3, (255, 0, 0), -1)

    info1 = f"idx: {idx}/{total - 1}   map: {h}x{w}   path_points: {len(path)}"
    info2 = "keys: n/right=next, p/left=prev, j=jump, s=save, q/esc=quit"
    cv2.putText(canvas, info1, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, info2, (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 220, 255), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="RLNF dataset viewer")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--clearance", type=int, default=2)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=800)
    args = parser.parse_args()

    dataset = RLNFDataset(
        split=args.split,
        noise_std=args.noise_std,
        num_points=args.num_points,
        clearance=args.clearance,
        step_size=args.step_size,
    )

    if len(dataset) == 0:
        print("No samples found with current filters.")
        return

    idx = max(0, min(args.index, len(dataset) - 1))
    window_name = f"RLNF Dataset Viewer [{args.split}]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.window_size, args.window_size)

    print("Viewer started.")
    print("Controls: n/right(next), p/left(prev), j(jump), s(save), q/esc(quit)")

    while True:
        sample = dataset[idx]
        frame = draw_sample(sample, idx, len(dataset))
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):
            break
        if key in (ord("n"), 83):  # right arrow
            idx = (idx + 1) % len(dataset)
            continue
        if key in (ord("p"), 81):  # left arrow
            idx = (idx - 1 + len(dataset)) % len(dataset)
            continue
        if key == ord("j"):
            raw = input(f"Jump to index (0~{len(dataset)-1}): ").strip()
            if raw.isdigit():
                idx = max(0, min(int(raw), len(dataset) - 1))
            continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
