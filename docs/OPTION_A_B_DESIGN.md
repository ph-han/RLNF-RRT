# Option A vs Option B: Design Document for Normalizing Flow Motion Planning

## Executive Summary

This document provides a comprehensive technical comparison of two approaches to fixing the fundamental modeling issue in our conditional Normalizing Flow for motion planning.

**The Core Problem**: The current model treats trajectory generation as independent point-wise transformations across time, ignoring temporal correlations. This causes samples to spread across free space rather than concentrate along meaningful path manifolds.

---

## Problem Analysis

### Current Architecture Issues

1. **Independent Time-Step Processing**
   - Current `CouplingBlock` processes each time step independently
   - Apply same affine transform to `x[:, t, :]` for all t
   - No information sharing across time dimension
   - Ignores path continuity constraints

2. **Mismatch Between Data and Model**
   - **Data**: Full trajectories with strong temporal correlations
   - **Model**: T independent 2D transformations
   - **Result**: Model cannot capture path structure

3. **Consequence**
   - Samples scatter in free space
   - Poor mode coverage
   - Doesn't respect path smoothness

---

## OPTION A: Point-Based Flow

### Conceptual Foundation

**Key Insight**: Model the distribution over **individual waypoints**, not full trajectories.

```
p(q | map, start, goal)  where q ∈ ℝ²
```

This is a **density estimation** problem over the configuration space, conditioned on:
- Environmental constraints (map)
- Task specification (start, goal)

### Why This Works

1. **Simplified Problem**
   - 2D flow is well-studied and stable
   - Each sample is i.i.d. given conditions
   - No temporal dependency modeling needed

2. **Probabilistic Path Representation**
   - High density near feasible paths
   - Low density in obstacles/unlikely regions
   - Natural multi-modality support

3. **Flexible Sampling**
   - Can sample arbitrary number of waypoints
   - Post-process into trajectories
   - Use with existing planners (RRT*, etc.)

### Architecture

```
Input:  q ∈ ℝ²  (single configuration point)
Condition: c ∈ ℝᶜ  (map + start + goal encoding)

Flow: f₁ ∘ f₂ ∘ ... ∘ fₖ
where each fᵢ: ℝ² × ℝᶜ → ℝ²

Base Distribution: z ~ N(0, I₂)
```

#### Network Architecture Details

**Condition Encoder** (unchanged):
```python
c = ConditionEncoder(map, start, goal)
# Output: (B, cond_dim)
```

**Coupling Blocks** (modified for 2D):
```python
class PointBasedCouplingBlock:
    Input: q (B, 2), c (B, cond_dim)
    
    # Split along feature dimension
    q_a = q[:, 0:1]  # (B, 1)
    q_b = q[:, 1:2]  # (B, 1)
    
    # Affine transforms conditioned on other component + condition
    s_a, t_a = STNet_a([q_b, c])  # Scale and translation
    q_a' = q_a * exp(s_a) + t_a
    
    s_b, t_b = STNet_b([q_a', c])
    q_b' = q_b * exp(s_b) + t_b
    
    Output: [q_a', q_b'], log_det
```

### Training Procedure

**Dataset Transformation**:
```python
# Original: (map, start, goal, trajectory)
# New: Multiple (map, start, goal, point) samples

For each trajectory with T points:
    For each point q_t in trajectory:
        # Optional: Add tube noise
        q_noisy = q_t + N(0, σ_tube²)
        
        yield (map, start, goal, q_noisy)
```

**Loss Function**:
```python
# Standard Normalizing Flow NLL
z, log_det = flow.forward(q, condition)
log_p_z = Normal(0, 1).log_prob(z).sum(dim=-1)
nll = -(log_p_z + log_det)
loss = nll.mean()
```

### Sampling & Path Construction

```python
# Sample many points
c = encoder(map, start, goal)
z = torch.randn(N, 2)  # N >> trajectory length
points = flow.inverse(z, c)  # (N, 2)

# Post-processing options:
# 1. Density visualization (heatmap)
# 2. K-means clustering → waypoints
# 3. Use as RRT* guiding distribution
# 4. Nearest-neighbor chaining
```

### Advantages

✅ **Simple and Stable**
- 2D flows are well-established
- No complex temporal modeling
- Easier to train and debug

✅ **Flexible Output**
- Generate any number of waypoints
- Not locked to fixed trajectory length
- Easy integration with planners

✅ **Interpretable**
- Visualize as density map
- Clear probabilistic interpretation
- Easy to diagnose failures

### Disadvantages

❌ **No Explicit Path Structure**
- Points generated independently
- Must reconstruct connectivity
- May not respect smoothness

❌ **Limited Path Diversity**
- All points conditioned same way
- No sequential dependency
- May miss elaborate maneuvers

❌ **Post-Processing Required**
- Need additional step to form trajectory
- No guaranteed path validity
- Potential computational overhead

---

## OPTION B: Trajectory-Based Flow

### Conceptual Foundation

**Key Insight**: Model **joint distribution over entire trajectories**, with explicit temporal modeling.

```
p(τ | map, start, goal)  where τ = [q₀, q₁, ..., qₜ] ∈ ℝᵀˣ²
```

This is a **structured prediction** problem over trajectory manifold.

### Why This Is Harder But More Expressive

1. **High-Dimensional Problem**
   - Input dimension: T × 2 (often T=512 → 1024D)
   - Requires powerful invertible transformations
   - Curse of dimensionality

2. **Must Model Temporal Correlations**
   - Adjacent points should be close (smoothness)
   - Global path structure matters
   - Need architectural inductive biases

3. **Potential for Better Performance**
   - One-shot trajectory generation
   - Captures complex multi-step maneuvers
   - Natural smoothness from correlations

### Architecture

```
Input: τ ∈ ℝᵀˣ²  (full trajectory)
Condition: c ∈ ℝᶜ  (map + start + goal encoding)

Flow: f₁ ∘ f₂ ∘ ... ∘ fₖ
where each fᵢ: ℝᵀˣ² × ℝᶜ → ℝᵀˣ²
with temporal awareness

Base Distribution: z ~ N(0, I_{T×2})
```

#### Key Architectural Change: Temporal Coupling

**Problem with Current Approach**:
```python
# Current: Process each time step independently
for t in range(T):
    q_t' = coupling_transform(q_t, c)  # ❌ Ignores other time steps
```

**Solution: THREE APPROACHES**

##### Approach B1: 1D Convolutional Coupling

```python
class ConvCouplingBlock:
    def __init__(self, cond_dim, hidden_dim, kernel_size=5):
        # Split along feature dimension (not time)
        # Input shape: (B, T, 2)
        
        # s/t networks now operate on sequences
        self.s_net = nn.Sequential(
            # Combine [trajectory_component, condition] along channel
            nn.Conv1d(1 + cond_dim, hidden_dim, kernel_size, padding='same'),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding='same'),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, 1)
        )
        
    def forward(self, traj, cond):
        # traj: (B, T, 2)
        # cond: (B, cond_dim) → expand to (B, T, cond_dim)
        
        traj_a = traj[:, :, 0:1]  # (B, T, 1)
        traj_b = traj[:, :, 1:2]
        
        # Prepare for Conv1d: (B, C, T)
        cond_expanded = cond.unsqueeze(1).expand(-1, T, -1)  # (B, T, cond_dim)
        
        # Process component a conditioned on b
        input_a = torch.cat([traj_b, cond_expanded], dim=-1)  # (B, T, 1+cond_dim)
        input_a = input_a.transpose(1, 2)  # (B, 1+cond_dim, T)
        
        s_a = self.s_net(input_a).transpose(1, 2)  # (B, T, 1)
        t_a = self.t_net(input_a).transpose(1, 2)
        
        traj_a' = traj_a * exp(s_a) + t_a
        
        # Similar for component b...
```

**Why This Works**:
- Receptive field captures local smoothness
- Each point influenced by neighbors
- Maintains invertibility (affine coupling)

##### Approach B2: Transformer-Based Coupling

```python
class TransformerCouplingBlock:
    def __init__(self, cond_dim, hidden_dim, num_heads=4):
        self.s_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=1 + cond_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim
            ),
            num_layers=2
        )
        self.s_proj = nn.Linear(1 + cond_dim, 1)
        
    def forward(self, traj, cond):
        # traj: (B, T, 2)
        # cond: (B, cond_dim)
        
        traj_a = traj[:, :, 0:1]  # (B, T, 1)
        traj_b = traj[:, :, 1:2]
        
        cond_expanded = cond.unsqueeze(1).expand(-1, T, -1)
        
        # Self-attention over time dimension
        input_a = torch.cat([traj_b, cond_expanded], dim=-1)  # (B, T, 1+cond)
        input_a = input_a.transpose(0, 1)  # (T, B, 1+cond) for transformer
        
        s_a_features = self.s_transformer(input_a)  # (T, B, 1+cond)
        s_a_features = s_a_features.transpose(0, 1)  # (B, T, 1+cond)
        s_a = self.s_proj(s_a_features)  # (B, T, 1)
        
        # Apply affine transform...
```

**Why This Works**:
- Global context via self-attention
- Long-range dependencies
- Flexible receptive field

##### Approach B3: Autoregressive Coupling

```python
class AutoregressiveCouplingBlock:
    def __init__(self, cond_dim, hidden_dim):
        self.s_rnn = nn.LSTM(
            input_size=1 + cond_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True  # For forward pass
        )
        self.s_proj = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, traj, cond):
        # Similar structure, but use RNN/LSTM
        # Bidirectional for forward pass
        # Unidirectional for inverse (more complex)
```

**Recommendation**: Start with **B1 (1D Conv)** - simpler and more stable.

### Training Procedure

**Dataset** (minimal change):
```python
# Existing dataset already provides full trajectories!
# No major changes needed

# Optional: Add tube noise to entire trajectory
trajectory_noisy = trajectory + N(0, σ_tube²)
```

**Loss Function**:
```python
# Same NLL, but over high-dimensional space
z, log_det = flow.forward(trajectory, condition)  # z: (B, T, 2)
log_p_z = Normal(0, 1).log_prob(z).sum(dim=(-1, -2))  # Sum over T and 2
nll = -(log_p_z + log_det)
loss = nll.mean()
```

### Sampling

```python
# Sample from base distribution
c = encoder(map, start, goal)
z = torch.randn(B, T, 2)

# One-shot trajectory generation
trajectory = flow.inverse(z, c)  # (B, T, 2)
# Ready to use, no post-processing needed!
```

### Advantages

✅ **Direct Trajectory Generation**
- One forward pass → complete path
- No post-processing
- Natural temporal structure

✅ **Smooth Paths**
- Temporal modeling enforces continuity
- Better path quality
- Captures complex maneuvers

✅ **Theoretically More Expressive**
- Can model intricate correlations
- Better fit to true data distribution

### Disadvantages

❌ **High Complexity**
- Harder to implement correctly
- More hyperparameters
- Unstable training possible

❌ **Fixed Output Length**
- Must commit to T upfront
- Less flexible than points

❌ **Computational Cost**
- Higher memory (O(T) vs O(1))
- Slower inference
- May need smaller T in practice

❌ **Debugging Difficulty**
- Hard to isolate failures
- Less interpretable intermediate states

---

## Implementation Strategy

### Minimal Code Changes Approach

To preserve your existing codebase, I recommend:

1. **Create Separate Modules**
   ```
   models/
   ├── option_a/
   │   ├── point_coupling_block.py
   │   ├── point_flow_planner.py
   │   └── point_dataset.py
   ├── option_b/
   │   ├── temporal_coupling_block.py
   │   └── trajectory_flow_planner.py
   └── [existing files unchanged]
   ```

2. **Shared Components**
   - Keep `ConditionEncoder` as-is
   - Reuse visualization utilities
   - Common training loop structure

3. **Configuration-Based Selection**
   ```python
   config = {
       'model_type': 'option_a',  # or 'option_b'
       # ... other params
   }
   ```

### Experimental Protocol

**Metrics for Comparison**:

1. **Likelihood Metrics**
   - Train NLL (lower = better)
   - Validation NLL (lower = better)
   - Per-dimension variance

2. **Sample Quality**
   - % of samples in free space
   - Distance to nearest GT path point
   - Path coverage (how many GT modes discovered)

3. **Path Metrics** (Option B specific)
   - Smoothness (sum of accelerations)
   - Path length distribution
   - Success rate (start → goal)

4. **Computational**
   - Training time per epoch
   - Sampling time
   - Memory usage

**Visualization Checklist**:
- [ ] Density heatmaps (Option A)
- [ ] Sample trajectories overlaid on map
- [ ] GT path comparison
- [ ] Training curves (NLL over time)
- [ ] Multi-modal environment tests

---

## Recommendations

### For Practical Motion Planning: **Option A**

**Rationale**:
1. **Robustness**: More stable, easier to deploy
2. **Flexibility**: Can generate variable number of waypoints
3. **Integration**: Works well with existing planners (RRT*, PRM)
4. **Interpretability**: Density maps are intuitive for debugging

**Use Case**: Production motion planning systems, especially when:
- Need to interface with classical planners
- Want probabilistic guidance for sampling
- Require interpretable failure modes

### For Research Experimentation: **Option B (with 1D Conv)**

**Rationale**:
1. **Novel Contribution**: More interesting theoretically
2. **End-to-End**: Direct trajectory generation is elegant
3. **Learning**: May discover better path structure
4. **Publication**: Addresses harder problem

**Use Case**: Research projects where:
- Goal is to push SOTA
- Have computational resources
- Can invest in hyperparameter tuning
- Want to explore flow models deeply

### Hybrid Approach (Future Work)

Consider combining both:
1. **Option A for waypoint generation**
2. **Option B fine-tuned on waypoint-to-trajectory**
3. **Best of both worlds**: Flexible + structured

---

## Technical Gotchas & Tips

### Common Pitfalls - Option A

1. **Too Few Samples**: Need 1000+ points for good density estimation
2. **Connectivity**: Post-processing must handle disconnected components
3. **Boundary Effects**: May sample outside valid region

### Common Pitfalls - Option B

1. **Exploding/Vanishing Gradients**: Long sequences are hard
   - Solution: Gradient clipping, careful initialization
   
2. **Memory**: Full trajectories can OOM
   - Solution: Gradient checkpointing, reduce T or batch size

3. **Inverse Stability**: Autoregressive inverses can accumulate errors
   - Solution: Stick with Conv or bidirectional models

### Implementation Tips

**Option A**:
```python
# Efficient point generation
def create_point_dataset(traj_dataset):
    points = []
    for traj in traj_dataset:
        # Sample or use all points
        for point in traj:
            points.append((map, start, goal, point))
    return PointDataset(points)
```

**Option B**:
```python
# Ensure proper broadcasting
cond = encoder(map, start, goal)  # (B, C)
cond_exp = cond.unsqueeze(1).expand(-1, T, -1)  # (B, T, C)
# Now can concatenate with trajectory features
```

---

## Conclusion

Both options address the core issue but from different angles:

- **Option A**: Sidestep temporal modeling by generating points
- **Option B**: Fix temporal modeling with architectural improvements

Start with **Option A** for quick validation and interpretability.

Invest in **Option B** if research goals justify the complexity.

**Most Important**: Both are valid solutions - the choice depends on your constraints (time, compute, research vs. application).

---

## Next Steps

1. ✅ Review this document
2. ⬜ Implement Option A (estimated 4-6 hours)
3. ⬜ Implement Option B (estimated 8-12 hours)
4. ⬜ Run comparison experiments (estimated 1-2 days)
5. ⬜ Analyze results and decide direction

**I'm ready to implement both when you are!**
