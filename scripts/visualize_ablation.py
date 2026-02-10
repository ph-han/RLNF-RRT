import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path

from rlnf_rrt.models.conditional_flow_planner import ConditionalFlowPlanner
from rlnf_rrt.data_pipeline.dataset import RLNFDataset


def infer_conditioning_mode(checkpoint):
    config = checkpoint.get("config", None)
    if config is not None and hasattr(config, "conditioning_mode"):
        return getattr(config, "conditioning_mode")
    state_dict = checkpoint.get("model_state_dict", {})
    has_film = any(".film1." in k or ".film2." in k for k in state_dict.keys())
    return "film" if has_film else "concat"

# ----------------------------
# Utils
# ----------------------------
def ensure_dir(p: str):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def map_diff_stats(a: torch.Tensor, b: torch.Tensor):
    da = (a - b).abs()
    return {
        "l1_mean": da.mean().item(),
        "l1_max": da.max().item(),
        "l2_mean": (da**2).mean().sqrt().item(),
        "occ_ratio_a": (a > 0.5).float().mean().item(),
        "occ_ratio_b": (b > 0.5).float().mean().item(),
    }

def rbf_mmd2(x: np.ndarray, y: np.ndarray, sigma: float = 0.08) -> float:
    N = min(len(x), 400)
    M = min(len(y), 400)
    x = x[np.random.choice(len(x), N, replace=False)]
    y = y[np.random.choice(len(y), M, replace=False)]

    def k(a, b):
        aa = (a**2).sum(axis=1, keepdims=True)
        bb = (b**2).sum(axis=1, keepdims=True).T
        dist2 = aa + bb - 2.0 * (a @ b.T)
        return np.exp(-dist2 / (2.0 * sigma * sigma))

    Kxx = k(x, x); np.fill_diagonal(Kxx, 0.0)
    Kyy = k(y, y); np.fill_diagonal(Kyy, 0.0)
    Kxy = k(x, y)

    mmd2 = Kxx.sum() / (N * (N - 1) + 1e-8) + Kyy.sum() / (M * (M - 1) + 1e-8) - 2.0 * Kxy.mean()
    return float(mmd2)

def draw_map(ax, map_np, alpha=0.5, title=None):
    ax.imshow(map_np, cmap="gray_r", origin='lower', extent=[0,1,0,1], alpha=alpha)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_aspect("equal", "box")
    if title is not None:
        ax.set_title(title)

def draw_points(ax, samples, gt_path=None, start=None, goal=None, color="blue", label="Generated"):
    ax.scatter(samples[:,0], samples[:,1], c=color, s=8, alpha=0.35, label=label)
    if gt_path is not None:
        ax.scatter(gt_path[:,0], gt_path[:,1], c="green", s=8, alpha=0.8, label="GT")
    if start is not None:
        ax.add_patch(Circle(start, 0.02, color="red"))
    if goal is not None:
        ax.add_patch(Circle(goal, 0.02, color="lime"))

# ----------------------------
# Random Map Generator
# ----------------------------
def generate_random_map(H: int, W: int, p_free: float = 0.65, seed: int = None) -> np.ndarray:
    """
    Returns map in [0,1] with convention similar to your dataset:
    - 1 = free
    - 0 = obstacle
    We'll create random rectangles/circles on a free canvas.
    """
    rng = np.random.default_rng(seed)
    m = np.ones((H, W), dtype=np.float32)

    # border as obstacle
    m[0,:]=0; m[-1,:]=0; m[:,0]=0; m[:,-1]=0

    # obstacle count scales with size
    num_obs = rng.integers(8, 25)

    for _ in range(num_obs):
        kind = rng.choice(["rect", "circle"], p=[0.7, 0.3])
        if kind == "rect":
            oh = int(rng.integers(max(3, H//30), max(8, H//6)))
            ow = int(rng.integers(max(3, W//30), max(8, W//6)))
            y = int(rng.integers(1, H-oh-1))
            x = int(rng.integers(1, W-ow-1))
            m[y:y+oh, x:x+ow] = 0.0
        else:
            r = int(rng.integers(max(3, min(H,W)//40), max(8, min(H,W)//10)))
            cy = int(rng.integers(r+1, H-r-1))
            cx = int(rng.integers(r+1, W-r-1))
            yy, xx = np.ogrid[:H, :W]
            mask = (yy-cy)**2 + (xx-cx)**2 <= r*r
            m[mask] = 0.0

    return m  # (H,W) in [0,1]

def sample_random_sg(map01: np.ndarray, max_tries=500, seed=None):
    """
    sample start/goal in free space if possible
    returns (start_xy01, goal_xy01) in [0,1]^2
    """
    rng = np.random.default_rng(seed)
    H, W = map01.shape
    for _ in range(max_tries):
        sy = rng.integers(0, H); sx = rng.integers(0, W)
        gy = rng.integers(0, H); gx = rng.integers(0, W)
        if map01[sy, sx] > 0.5 and map01[gy, gx] > 0.5:
            start = np.array([sx / W, sy / H], dtype=np.float32)
            goal  = np.array([gx / W, gy / H], dtype=np.float32)
            return start, goal
    # fallback uniform
    return rng.random(2, dtype=np.float32), rng.random(2, dtype=np.float32)

# ----------------------------
# Main
# ----------------------------
def ablation_random_map_and_sg(
    checkpoint_path: str,
    idx: int = 182,
    num_samples: int = 400,
    out_path: str = "result/visualization/ablation_random_map_sg.png",
    seed: int = 0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", None)
    conditioning_mode = infer_conditioning_mode(ckpt)

    model = ConditionalFlowPlanner(
        num_blocks=getattr(config, "num_blocks", 4) if config else 4,
        sg_dim=getattr(config, "sg_dim", 2) if config else 2,
        position_embed_dim=getattr(config, "position_embed_dim", 32) if config else 32,
        map_embed_dim=getattr(config, "map_embed_dim", 256) if config else 256,
        cond_dim=getattr(config, "cond_dim", 128) if config else 128,
        hidden_dim=getattr(config, "hidden_dim", 128) if config else 128,
        s_max=getattr(config, "s_max", 2.0) if config else 2.0,
        conditioning_mode=conditioning_mode,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Conditioning mode: {conditioning_mode}")

    dataset = RLNFDataset(split="test")
    data = dataset[idx]

    real_map = data["map"].unsqueeze(0).to(device)  # (1,1,H,W)
    start = data["start"].unsqueeze(0).to(device)   # (1,2)
    goal  = data["goal"].unsqueeze(0).to(device)    # (1,2)
    gt_path = data["gt_path"].cpu().numpy()

    zero_map = torch.zeros_like(real_map)

    # random map same size
    H, W = data["map"].shape[-2], data["map"].shape[-1]
    rand_map_np = generate_random_map(H, W, seed=seed)
    rand_map = torch.from_numpy(rand_map_np).float().unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)

    # random sg (prefer free space in rand_map)
    rand_start_np, rand_goal_np = sample_random_sg(rand_map_np, seed=seed+1)
    rand_start = torch.from_numpy(rand_start_np).float().unsqueeze(0).to(device)
    rand_goal  = torch.from_numpy(rand_goal_np).float().unsqueeze(0).to(device)

    zero_sg = torch.zeros_like(start)

    # configs: (name, map, start, goal)
    cases = [
        ("FULL(real_map, real_sg)", real_map, start, goal),
        ("NO-MAP(zero_map, real_sg)", zero_map, start, goal),
        ("RAND-MAP(rand_map, real_sg)", rand_map, start, goal),
        ("NO-SG(real_map, zero_sg)", real_map, zero_sg, zero_sg),
        ("RAND-SG(real_map, rand_sg)", real_map, rand_start, rand_goal),
        ("NO-BOTH(zero_map, zero_sg)", zero_map, zero_sg, zero_sg),
    ]

    # sample all
    samples = {}
    with torch.no_grad():
        for name, m, s, g in cases:
            samples[name] = model.sample(m, s, g, num_samples=num_samples).cpu().numpy()[0]

    # quantitative: compare each to FULL
    base = samples[cases[0][0]]
    print("\n--- MMD^2 vs FULL ---")
    for name, *_ in cases[1:]:
        mmd = rbf_mmd2(base, samples[name], sigma=0.08)
        print(f"MMD(FULL, {name}) = {mmd:.6f}")

    # also print map diffs
    print("\n--- Map differences ---")
    print("real vs zero:", map_diff_stats(real_map.cpu(), zero_map.cpu()))
    print("real vs rand:", map_diff_stats(real_map.cpu(), rand_map.cpu()))

    # plot: 2 rows x 3 cols
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))

    # backgrounds: use input map for each case (so visualization is honest)
    bg = {
        "FULL(real_map, real_sg)": data["map"][0].cpu().numpy(),
        "NO-MAP(zero_map, real_sg)": np.zeros((H, W), dtype=np.float32),
        "RAND-MAP(rand_map, real_sg)": rand_map_np,
        "NO-SG(real_map, zero_sg)": data["map"][0].cpu().numpy(),
        "RAND-SG(real_map, rand_sg)": data["map"][0].cpu().numpy(),
        "NO-BOTH(zero_map, zero_sg)": np.zeros((H, W), dtype=np.float32),
    }

    # start/goal for drawing (convert to numpy 2D)
    real_start_np = data["start"].cpu().numpy()
    real_goal_np  = data["goal"].cpu().numpy()

    for ax, (name, m, s, g) in zip(axes.flatten(), cases):
        draw_map(ax, bg[name], alpha=0.5, title=name)
        # decide which start/goal to draw
        if "RAND-SG" in name:
            st = rand_start_np; gl = rand_goal_np
        elif "NO-SG" in name or "NO-BOTH" in name:
            st = (0.0, 0.0); gl = (0.0, 0.0)
        else:
            st = real_start_np; gl = real_goal_np

        draw_points(ax, samples[name], gt_path=gt_path, start=st, goal=gl, color="blue")

    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    plt.tight_layout()
    ensure_dir(out_path)
    plt.savefig(out_path, dpi=180)
    print(f"\n✅ Saved {out_path}")

if __name__ == "__main__":
    ablation_random_map_and_sg(
        checkpoint_path="result/models/v10_best_model.pt",
        idx=np.random.randint(200),
        num_samples=512,
        out_path="result/visualization/ablation_random_map_sg.png",
        seed=0
    )
