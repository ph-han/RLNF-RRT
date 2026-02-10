from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    # Data
    batch_size: int = 32
    num_workers: int = 4
    num_samples_per_path: int = 512  # Number of points to sample from each gt_path
    noise_std: float = 0.01  # Noise std for data augmentation
    
    # Model
    num_blocks: int = 8
    cond_dim: int = 128
    hidden_dim: int = 256
    s_max: float = 2.0  # max scaling factor in coupling layer
    conditioning_mode: str = "film"  # "concat"(paper-style) or "film"
    position_embed_dim: int = 32
    map_embed_dim: int = 256
    sg_dim: int = 2  # start/goal dimension (x, y)
    
    # Optimizer
    lr: float = 5e-4
    weight_decay: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    
    # Scheduler
    scheduler: str = "cosine"  # "cosine" or "step"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    T_max: int = 50
    
    # Training
    epochs: int = 100
    grad_clip: float = 1.0
    log_interval: int = 10  # Log every N batches
    val_interval: int = 1  # Validate every N epochs
    save_interval: int = 10  # Save checkpoint every N epochs
    
    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3])
    checkpoint_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3] / "result" / "checkpoints")
    log_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3] / "result" / "logs")
    
    # Device
    device: str = "cuda"
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
