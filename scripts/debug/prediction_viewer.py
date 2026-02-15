from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rlnf_rrt.data.dataset import RLNFDataset
from rlnf_rrt.utils.config import load_toml, resolve_project_path


def _find_latest_prediction_dir(eval_cfg: dict) -> Path:
    pred_root = resolve_project_path(eval_cfg.get("prediction_dir", "outputs/predictions"))
    if not pred_root.exists():
        raise FileNotFoundError(f"Prediction root not found: {pred_root}")

    ckpt_stem = Path(eval_cfg["checkpoint"]).stem
    entries = sorted(
        [p for p in pred_root.iterdir() if p.is_dir() and p.name.startswith(f"{ckpt_stem}_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not entries:
        raise FileNotFoundError(
            f"No prediction directory found under {pred_root} with prefix '{ckpt_stem}_'"
        )
    return entries[0]


def _load_sample_predictions(pred_dir: Path, ds_idx: int) -> list[np.ndarray]:
    files = sorted(pred_dir.glob(f"pred_{ds_idx:07d}_s*.npy"))
    preds: list[np.ndarray] = []
    for f in files:
        preds.append(np.load(f).astype(np.float32))
    return preds


def _to_pixels(path_xy: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    if len(path_xy) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    px = np.clip(path_xy[:, 0] * (w - 1), 0, w - 1).astype(np.int32)
    py = np.clip(path_xy[:, 1] * (h - 1), 0, h - 1).astype(np.int32)
    return px, py


def _draw(sample: dict, preds: list[np.ndarray], idx: int, total: int, pred_dir: Path) -> np.ndarray:
    map_np = sample["map"].squeeze(0).numpy()
    start = sample["start"].numpy()
    goal = sample["goal"].numpy()
    gt = sample["gt_path"].numpy()

    h, w = map_np.shape
    canvas = cv2.cvtColor((map_np * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    gt_x, gt_y = _to_pixels(gt, w, h)
    if len(gt_x) > 0:
        for x, y in zip(gt_x, gt_y):
            cv2.circle(canvas, (int(x), int(y)), 1, (0, 255, 255), -1)  # yellow GT

    pred_colors = [
        (0, 200, 0),
        (255, 120, 0),
        (200, 0, 200),
        (0, 180, 255),
        (100, 255, 100),
    ]
    for s, pred in enumerate(preds):
        px, py = _to_pixels(pred, w, h)
        color = pred_colors[s % len(pred_colors)]
        for x, y in zip(px, py):
            cv2.circle(canvas, (int(x), int(y)), 1, color, -1)

    sx = int(np.clip(start[0] * (w - 1), 0, w - 1))
    sy = int(np.clip(start[1] * (h - 1), 0, h - 1))
    gx = int(np.clip(goal[0] * (w - 1), 0, w - 1))
    gy = int(np.clip(goal[1] * (h - 1), 0, h - 1))
    cv2.circle(canvas, (sx, sy), 4, (0, 0, 255), -1)
    cv2.circle(canvas, (gx, gy), 4, (255, 0, 0), -1)

    info1 = f"idx: {idx}/{total - 1}  preds: {len(preds)}  map: {h}x{w}"
    info2 = "GT=yellow, Start=red, Goal=blue, Pred=multi-color"
    info3 = f"pred_dir: {pred_dir.name}"
    info4 = "keys: n/right next, p/left prev, j jump, q/esc quit"
    cv2.putText(canvas, info1, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, info2, (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, info3, (8, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, info4, (8, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 220, 255), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Prediction overlay viewer")
    parser.add_argument("--eval-config", type=str, default="configs/eval/default.toml")
    parser.add_argument("--pred-dir", type=str, default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=900)
    args = parser.parse_args()

    cfg = load_toml(args.eval_config)
    data_cfg = cfg["data"]
    eval_cfg = cfg["eval"]

    pred_dir = resolve_project_path(args.pred_dir) if args.pred_dir else _find_latest_prediction_dir(eval_cfg)
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")

    ds = RLNFDataset(
        split=str(data_cfg.get("split", "test")),
        data_root=resolve_project_path(data_cfg.get("data_root", "data")),
        noise_std=0.0,
        num_points=int(data_cfg["num_points"]),
        clearance=int(data_cfg["clearance"]),
        step_size=int(data_cfg["step_size"]),
    )
    if len(ds) == 0:
        print("Dataset is empty with current config filters.")
        return

    idx = max(0, min(args.index, len(ds) - 1))

    win = "RLNF Prediction Viewer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.window_size, args.window_size)

    print(f"Prediction dir: {pred_dir}")
    print("Controls: n/right(next), p/left(prev), j(jump), q/esc(quit)")

    while True:
        sample = ds[idx]
        preds = _load_sample_predictions(pred_dir, idx)
        frame = _draw(sample, preds, idx, len(ds), pred_dir)
        cv2.imshow(win, frame)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("n"), 83):
            idx = (idx + 1) % len(ds)
            continue
        if key in (ord("p"), 81):
            idx = (idx - 1 + len(ds)) % len(ds)
            continue
        if key == ord("j"):
            raw = input(f"Jump to index (0~{len(ds)-1}): ").strip()
            if raw.isdigit():
                idx = max(0, min(int(raw), len(ds) - 1))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
