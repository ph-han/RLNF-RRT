# Option A vs Option B: Quick Start Guide

This guide will help you quickly train and compare both options.

## Overview

- **Option A**: Point-based normalizing flow (models individual waypoints)
- **Option B**: Trajectory-based normalizing flow (models complete paths with temporal correlations)

## Quick Start

### 1. Train Option A (Point-Based Flow)

```bash
# Basic training
python scripts/train_option_a.py \
    --num_epochs 50 \
    --batch_size 64 \
    --lr 1e-4 \
    --save_dir experiments/option_a

# With W&B logging
python scripts/train_option_a.py \
    --num_epochs 100 \
    --batch_size 64 \
    --tube_noise_std 0.01 \
    --use_wandb \
    --save_dir experiments/option_a
```

**Key parameters:**
- `--points_per_traj`: How many points to extract from each trajectory (default: 512)
- `--tube_noise_std`: Add noise around GT path for data augmentation (default: 0.01)
- `--num_blocks`: Number of coupling blocks (default: 4)

**Expected training time:** ~2-3 hours on GPU for 100 epochs

### 2. Train Option B (Trajectory-Based Flow)

```bash
# Conv1D coupling (recommended)
python scripts/train_option_b.py \
    --coupling_type conv \
    --num_epochs 50 \
    --batch_size 32 \
    --kernel_size 5 \
    --save_dir experiments/option_b

# Transformer coupling (experimental)
python scripts/train_option_b.py \
    --coupling_type transformer \
    --num_epochs 100 \
    --batch_size 16 \
    --num_heads 4 \
    --use_wandb \
    --save_dir experiments/option_b
```

**Key parameters:**
- `--coupling_type`: 'conv' (recommended) or 'transformer'
- `--kernel_size`: For conv coupling, temporal receptive field (default: 5)
- `--num_heads`: For transformer coupling, attention heads (default: 4)
- `--grad_clip`: Gradient clipping for stability (default: 1.0)

**Expected training time:** ~4-6 hours on GPU for 100 epochs (conv), longer for transformer

### 3. Compare Results

```bash
python scripts/compare_options.py \
    --option_a_checkpoint experiments/option_a/best_model.pth \
    --option_b_checkpoint experiments/option_b/conv/best_model.pth \
    --option_b_coupling conv \
    --output_dir experiments/comparison \
    --num_test_cases 20 \
    --num_samples 1000 \
    --num_trajectories 10
```

This will generate:
- Visual comparisons for each test case
- Quantitative metrics summary
- `comparison_summary.json` with aggregated results

## Understanding the Outputs

### Option A Visualizations

Each visualization shows:
1. **Density Heatmap**: Learned probability distribution over points
2. **Sampled Points**: Scatter plot of generated waypoints
3. **Distance Histogram**: How close samples are to GT path

**Good indicators:**
- High density along GT path corridor
- Most samples cluster near feasible paths
- Mean distance to GT < 0.05

### Option B Visualizations

Each visualization shows:
1. **Sampled Trajectories**: Generated paths overlaid on map
2. **Path Length Distribution**: Variation in generated path lengths
3. **Smoothness Distribution**: Acceleration magnitudes (lower = smoother)

**Good indicators:**
- Trajectories follow free space
- Path lengths similar to GT
- Low acceleration variance (smooth paths)

## Key Metrics

### Option A
- **Mean distance to GT**: How close sampled points are to ground truth path
- **Sample coverage**: What % of GT path points have nearby samples

### Option B
- **Mean path length**: Average length of generated trajectories
- **Mean acceleration**: Path smoothness measure (lower = smoother)
- **Validation NLL**: Lower is better (indicates better likelihood modeling)

## Troubleshooting

### Option A Issues

**Problem: Samples scatter everywhere**
- Solution: Increase `num_blocks` (try 6-8)
- Solution: Reduce `tube_noise_std` (try 0.005)
- Solution: Train longer (100+ epochs)

**Problem: Training NLL not decreasing**
- Solution: Reduce learning rate (try 5e-5)
- Solution: Check data loading (ensure normalized to [0,1])

### Option B Issues

**Problem: Gradients explode/vanish**
- Solution: Adjust `grad_clip` (try 0.5 or 2.0)
- Solution: Reduce `kernel_size` (try 3)
- Solution: Use conv instead of transformer

**Problem: Generated paths not smooth**
- Solution: Increase `kernel_size` (try 7 or 9)
- Solution: Add more coupling blocks
- Solution: Try transformer coupling

**Problem: Out of memory**
- Solution: Reduce `batch_size` (try 16 or 8)
- Solution: Reduce trajectory length in dataset
- Solution: Use gradient checkpointing (code modification needed)

## Advanced Usage

### Custom Dataset

Modify `points_per_trajectory` or trajectory length:

```python
# In your custom script
from rlnf_rrt.models.option_a import DynamicPointDataset

train_dataset = DynamicPointDataset(
    base_trajectory_dataset,
    points_per_trajectory=256,  # Sample fewer points
    tube_noise_std=0.02  # More aggressive augmentation
)
```

### Visualization During Training

Add this to training loop:

```python
if epoch % 10 == 0:
    # For Option A
    with torch.no_grad():
        density, extent = model.sample_density_grid(
            sample_map, sample_start, sample_goal
        )
    plt.imshow(density)
    plt.savefig(f'density_epoch_{epoch}.png')
    
    # For Option B
    with torch.no_grad():
        trajs = model.sample(sample_map, sample_start, sample_goal, 
                            num_samples=5, seq_len=512)
    # Plot trajectories...
```

## Expected Results

Based on the design analysis:

### Option A
- **Pros**: Faster training, stable, interpretable density maps
- **Cons**: Needs post-processing to form trajectories
- **Best for**: Integration with classical planners, density-based guidance

### Option B (Conv)
- **Pros**: Direct trajectory generation, smooth paths, captures temporal structure
- **Cons**: Slower training, higher memory, fixed length
- **Best for**: End-to-end learning, research on flow models

### Option B (Transformer)
- **Pros**: Most expressive, global context
- **Cons**: Slowest, most unstable, highest memory
- **Best for**: Research experimentation with large compute budget

## Recommended Workflow

1. **Start with Option A** (1-2 days)
   - Quick validation of concept
   - Establish baseline metrics
   - Visualize learned densities

2. **Then try Option B-Conv** (2-3 days)
   - Compare trajectory quality
   - Measure smoothness improvements
   - Evaluate end-to-end generation

3. **Optional: Option B-Transformer** (if results warrant)
   - Experiment with attention mechanisms
   - Push expressiveness limits

4. **Compare and decide** (1 day)
   - Run comparison script
   - Analyze metrics
   - Choose direction for further development

## Next Steps After Training

### For Production/Application
→ Choose **Option A** if you need:
- Flexible waypoint generation
- Integration with existing planners
- Interpretable probabilistic guidance

→ Choose **Option B-Conv** if you need:
- Direct trajectory output
- Smooth paths without post-processing
- End-to-end learnable system

### For Research Publication
→ **Option B** provides:
- More novel contribution
- Addresses harder problem
- Interesting architectural innovations

→ **Option A** provides:
- Clear baseline and analysis
- Practical integration story
- Strong empirical foundation

## Citation & References

If you use this code for research, consider citing:

- PlannerFlows (original trajectory-based flows for motion planning)
- Real-NVP (affine coupling layers)
- Glow (flow-based generative models)

---

**Questions?** Check the design document: `docs/OPTION_A_B_DESIGN.md`
