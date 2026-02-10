import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path

from rlnf_rrt.models.conditional_flow_planner import ConditionalFlowPlanner
from rlnf_rrt.data_pipeline.dataset import RLNFDataset

# ----------------------------
# Utils
# ----------------------------
def map_diff_stats(a: torch.Tensor, b: torch.Tensor):
    """
    a,b: (1,1,H,W) in [0,1] (assumed)
    """
    da = (a - b).abs()
    return {
        "l1_mean": da.mean().item(),
        "l1_max": da.max().item(),
        "l2_mean": (da**2).mean().sqrt().item(),
        "occ_ratio_a": (a > 0.5).float().mean().item(),
        "occ_ratio_b": (b > 0.5).float().mean().item(),
    }

def rbf_mmd2(x: np.ndarray, y: np.ndarray, sigma: float = 0.1) -> float:
    """
    Unbiased-ish MMD^2 with RBF kernel.
    x,y: (N,2) in [0,1]
    """
    # subsample to keep it fast
    N = min(len(x), 400)
    M = min(len(y), 400)
    x = x[np.random.choice(len(x), N, replace=False)]
    y = y[np.random.choice(len(y), M, replace=False)]

    def k(a, b):
        # (N,2) (M,2) -> (N,M)
        aa = (a**2).sum(axis=1, keepdims=True)
        bb = (b**2).sum(axis=1, keepdims=True).T
        dist2 = aa + bb - 2.0 * (a @ b.T)
        return np.exp(-dist2 / (2.0 * sigma * sigma))

    Kxx = k(x, x)
    Kyy = k(y, y)
    Kxy = k(x, y)

    # remove diagonal bias
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    mmd2 = Kxx.sum() / (N * (N - 1) + 1e-8) + Kyy.sum() / (M * (M - 1) + 1e-8) - 2.0 * Kxy.mean()
    return float(mmd2)

def ensure_dir(p: str):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Main
# ----------------------------
def visualize_map_ablation_3way(checkpoint_path: str,
                                idx: int = 182,
                                other_idx: int = None,
                                num_samples: int = 400,
                                out_path: str = "result/visualization/ablation_3way.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = ConditionalFlowPlanner(
        num_blocks=4,
        position_embed_dim=32,
        map_embed_dim=256,
        cond_dim=128,
        hidden_dim=128
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = RLNFDataset(split="test")

    data = dataset[idx]
    real_map = data["map"].unsqueeze(0).to(device)   # (1,1,H,W)
    start = data["start"].unsqueeze(0).to(device)    # (1,2)
    goal  = data["goal"].unsqueeze(0).to(device)     # (1,2)
    gt_path = data["gt_path"].cpu().numpy()          # (T,2)
    zero_map = torch.zeros_like(real_map)

    # choose other map index (make it "very different" if not provided)
    if other_idx is None:
        # pick the first map that is sufficiently different by L1 mean
        with torch.no_grad():
            base = real_map.cpu()
            other_idx = None
            for j in range(len(dataset)):
                if j == idx:
                    continue
                mj = dataset[j]["map"].unsqueeze(0).cpu()
                stats = map_diff_stats(base, mj)
                if stats["l1_mean"] > 0.20:   # threshold; adjust if needed
                    other_idx = j
                    break
            if other_idx is None:
                other_idx = (idx + 1) % len(dataset)

    other_map = dataset[other_idx]["map"].unsqueeze(0).to(device)

    # --- sampling ---
    with torch.no_grad():
        s_full  = model.sample(real_map,  start, goal, num_samples=num_samples).cpu().numpy()[0]   # (N,2)
        s_zero  = model.sample(zero_map,  start, goal, num_samples=num_samples).cpu().numpy()[0]
        s_other = model.sample(other_map, start, goal, num_samples=num_samples).cpu().numpy()[0]

    # --- quantitative diagnostics ---
    real_vs_other = map_diff_stats(real_map.cpu(), other_map.cpu())
    real_vs_zero  = map_diff_stats(real_map.cpu(), zero_map.cpu())

    mmd_full_zero  = rbf_mmd2(s_full, s_zero, sigma=0.08)
    mmd_full_other = rbf_mmd2(s_full, s_other, sigma=0.08)
    mmd_zero_other = rbf_mmd2(s_zero, s_other, sigma=0.08)

    print("\n--- Map difference ---")
    print(f"idx={idx} vs other_idx={other_idx}: {real_vs_other}")
    print(f"idx={idx} vs zero_map:           {real_vs_zero}")

    print("\n--- Sample distribution difference (MMD^2, RBF) ---")
    print(f"MMD(full, zero)  = {mmd_full_zero:.6f}")
    print(f"MMD(full, other) = {mmd_full_other:.6f}")
    print(f"MMD(zero, other) = {mmd_zero_other:.6f}")

    # Interpretation hint
    print("\nInterpretation:")
    print("- If MMD(full, zero) ~ 0 and MMD(full, other) ~ 0 => model ignores map.")
    print("- If MMD(full, other) is large => model uses map (good).")
    print("- If MMD(full, zero) is large => model heavily relies on map; check robustness.\n")

    # --- plot ---
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # helper to draw background map (for reference) and points
    def draw(ax, bg_map_np, samples, title, color):
        ax.imshow(bg_map_np, cmap="gray_r", extent=[0,1,0,1], alpha=0.5)
        ax.scatter(samples[:,0], samples[:,1], c=color, s=8, alpha=0.35, label="Generated")
        ax.scatter(gt_path[:,0], gt_path[:,1], c="green", s=8, alpha=0.8, label="GT")
        ax.add_patch(Circle(data["start"].cpu().numpy(), 0.02, color="red"))
        ax.add_patch(Circle(data["goal"].cpu().numpy(), 0.02, color="lime"))
        ax.set_title(title)
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_aspect("equal", "box")

    bg_real = data["map"][0].cpu().numpy()
    bg_zero = np.zeros_like(bg_real)
    bg_other = dataset[other_idx]["map"][0].cpu().numpy()

    draw(axes[0], bg_real,  s_full,  f"FULL (real_map)\nidx={idx}", "blue")

    # IMPORTANT: show the *actual input* map (zero) for the ablation panel
    draw(axes[1], bg_zero,  s_zero,  "NO-MAP (zero_map input)\n(background is zero)", "orange")

    draw(axes[2], bg_other, s_other, f"OTHER-MAP (different map input)\nother_idx={other_idx}", "purple")

    # compact legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)

    # annotate metrics
    fig.suptitle(
        f"Map Ablation 3-way | MMD2(full,zero)={mmd_full_zero:.4f} | "
        f"MMD2(full,other)={mmd_full_other:.4f} | "
        f"MapL1mean(real,other)={real_vs_other['l1_mean']:.3f}",
        fontsize=14
    )

    ensure_dir(out_path)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    print(f"✅ Saved {out_path}")

if __name__ == "__main__":
    visualize_map_ablation_3way(
        checkpoint_path="result/models/v6_2_best_model.pt",
        idx=182,
        other_idx=None,         # 자동으로 '충분히 다른' 맵을 찾음
        num_samples=400,
        out_path="result/visualization/ablation_3way.png"
    )
