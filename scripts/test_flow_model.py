#!/usr/bin/env python
"""
Test script for Normalizing Flow model components.
Tests: CouplingBlock, ConditionalFlowPlanner
"""
import torch
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rlnf_rrt.models.coupling_block import CouplingBlock, STNet
from rlnf_rrt.models.conditional_flow_planner import ConditionalFlowPlanner


def test_stnet():
    """Test STNet basic forward pass."""
    print("=" * 60)
    print("🧪 Test 1: STNet Forward Pass")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 512
    cond_dim = 128
    hidden_dim = 128
    
    stnet = STNet(cond_dim=cond_dim, hidden_dim=hidden_dim)
    
    # Input: z_component (1) + cond (128) = 129
    # Output: 1 (scale or translation value)
    x = torch.randn(batch_size, seq_len, 1 + cond_dim)
    
    out = stnet(x)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Expected:     ({batch_size}, {seq_len}, 1)")
    
    assert out.shape == (batch_size, seq_len, 1), f"Shape mismatch! Got {out.shape}"
    print("  ✅ STNet test PASSED\n")
    return True


def test_coupling_block_forward():
    """Test CouplingBlock forward pass."""
    print("=" * 60)
    print("🧪 Test 2: CouplingBlock Forward Pass")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 512
    sg_dim = 2
    cond_dim = 128
    
    block = CouplingBlock(cond_dim=cond_dim)
    
    x = torch.randn(batch_size, seq_len, sg_dim)  # (B, T, 2)
    cond = torch.randn(batch_size, seq_len, cond_dim)  # (B, T, cond_dim)
    
    z, log_det = block.forward(x, cond)
    
    print(f"  Input x shape:     {x.shape}")
    print(f"  Input cond shape:  {cond.shape}")
    print(f"  Output z shape:    {z.shape}")
    print(f"  Output log_det shape: {log_det.shape}")
    print(f"  Expected z:        ({batch_size}, {seq_len}, {sg_dim})")
    print(f"  Expected log_det:  ({batch_size},)")
    
    assert z.shape == x.shape, f"z shape mismatch! Got {z.shape}, expected {x.shape}"
    assert log_det.shape == (batch_size,), f"log_det shape mismatch! Got {log_det.shape}"
    print("  ✅ CouplingBlock forward test PASSED\n")
    return True


def test_coupling_block_inverse():
    """Test CouplingBlock forward-inverse consistency."""
    print("=" * 60)
    print("🧪 Test 3: CouplingBlock Forward-Inverse Consistency")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 512
    sg_dim = 2
    cond_dim = 128
    
    block = CouplingBlock(cond_dim=cond_dim)
    block.eval()  # Set to eval mode for consistent results
    
    x = torch.randn(batch_size, seq_len, sg_dim)
    cond = torch.randn(batch_size, seq_len, cond_dim)
    
    # Forward: x -> z
    z, log_det = block.forward(x, cond)
    
    # Inverse: z -> x_reconstructed
    x_recon = block.inverse(z, cond)
    
    # Check reconstruction error
    error = (x - x_recon).abs().max().item()
    mean_error = (x - x_recon).abs().mean().item()
    
    print(f"  Original x shape:      {x.shape}")
    print(f"  Transformed z shape:   {z.shape}")
    print(f"  Reconstructed x shape: {x_recon.shape}")
    print(f"  Max reconstruction error:  {error:.2e}")
    print(f"  Mean reconstruction error: {mean_error:.2e}")
    
    # Should be very small (numerical precision)
    tolerance = 1e-5
    if error < tolerance:
        print(f"  ✅ Forward-Inverse consistency PASSED (error < {tolerance})\n")
        return True
    else:
        print(f"  ❌ Forward-Inverse consistency FAILED (error >= {tolerance})\n")
        return False


def test_conditional_flow_planner_forward():
    """Test full ConditionalFlowPlanner forward pass."""
    print("=" * 60)
    print("🧪 Test 4: ConditionalFlowPlanner Forward Pass")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 512
    img_size = 64
    sg_dim = 2
    num_blocks = 4
    
    model = ConditionalFlowPlanner(
        num_blocks=num_blocks,
        sg_dim=sg_dim,
        position_embed_dim=128,
        map_embed_dim=256,
        cond_dim=128
    )
    
    # Input tensors
    gt_trajs = torch.randn(batch_size, seq_len, sg_dim)  # (B, T, 2)
    map_img = torch.randn(batch_size, 1, img_size, img_size)  # (B, 1, H, W)
    start = torch.randn(batch_size, sg_dim)  # (B, 2)
    goal = torch.randn(batch_size, sg_dim)  # (B, 2)
    
    z, log_det = model.forward(gt_trajs, map_img, start, goal)
    
    print(f"  gt_trajs shape: {gt_trajs.shape}")
    print(f"  map_img shape:  {map_img.shape}")
    print(f"  start shape:    {start.shape}")
    print(f"  goal shape:     {goal.shape}")
    print(f"  Output z shape: {z.shape}")
    print(f"  log_det shape:  {log_det.shape}")
    print(f"  Expected z:     ({batch_size}, {seq_len}, {sg_dim})")
    print(f"  Expected log_det: ({batch_size},)")
    
    assert z.shape == gt_trajs.shape, f"z shape mismatch!"
    assert log_det.shape == (batch_size,), f"log_det shape mismatch!"
    print("  ✅ ConditionalFlowPlanner forward test PASSED\n")
    return True


def test_conditional_flow_planner_sample():
    """Test ConditionalFlowPlanner sampling."""
    print("=" * 60)
    print("🧪 Test 5: ConditionalFlowPlanner Sample")
    print("=" * 60)
    
    batch_size = 2
    num_samples = 100
    img_size = 64
    sg_dim = 2
    num_blocks = 4
    
    model = ConditionalFlowPlanner(
        num_blocks=num_blocks,
        sg_dim=sg_dim,
        position_embed_dim=128,
        map_embed_dim=256,
        cond_dim=128
    )
    model.eval()
    
    # Input tensors
    map_img = torch.randn(batch_size, 1, img_size, img_size)
    start = torch.randn(batch_size, sg_dim)
    goal = torch.randn(batch_size, sg_dim)
    
    samples = model.sample(map_img, start, goal, num_samples=num_samples)
    
    print(f"  map_img shape:   {map_img.shape}")
    print(f"  start shape:     {start.shape}")
    print(f"  goal shape:      {goal.shape}")
    print(f"  samples shape:   {samples.shape}")
    print(f"  Expected:        ({batch_size}, {num_samples}, {sg_dim})")
    
    assert samples.shape == (batch_size, num_samples, sg_dim), f"samples shape mismatch!"
    print("  ✅ ConditionalFlowPlanner sample test PASSED\n")
    return True


def test_flow_invertibility():
    """Test full model forward-inverse consistency."""
    print("=" * 60)
    print("🧪 Test 6: Full Model Forward-Inverse Consistency")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 100
    img_size = 64
    sg_dim = 2
    num_blocks = 4
    cond_dim = 128
    
    model = ConditionalFlowPlanner(
        num_blocks=num_blocks,
        sg_dim=sg_dim,
        position_embed_dim=128,
        map_embed_dim=256,
        cond_dim=cond_dim
    )
    model.eval()
    
    # Input tensors
    gt_trajs = torch.randn(batch_size, seq_len, sg_dim)
    map_img = torch.randn(batch_size, 1, img_size, img_size)
    start = torch.randn(batch_size, sg_dim)
    goal = torch.randn(batch_size, sg_dim)
    
    # Forward: gt_trajs -> z
    cond = model.condition_encoder(map_img, start, goal)
    cond = cond.unsqueeze(1).expand(-1, seq_len, -1)  # (B, T, cond_dim)
    
    x = gt_trajs
    for block in model.flow_model:
        x, _ = block(x, cond)
    z = x
    
    # Inverse: z -> x_recon
    x_recon = z
    for block in reversed(model.flow_model):
        x_recon = block.inverse(x_recon, cond)
    
    error = (gt_trajs - x_recon).abs().max().item()
    mean_error = (gt_trajs - x_recon).abs().mean().item()
    
    print(f"  Original gt_trajs shape: {gt_trajs.shape}")
    print(f"  Reconstructed shape:     {x_recon.shape}")
    print(f"  Max reconstruction error:  {error:.2e}")
    print(f"  Mean reconstruction error: {mean_error:.2e}")
    
    tolerance = 1e-4
    if error < tolerance:
        print(f"  ✅ Full model invertibility PASSED (error < {tolerance})\n")
        return True
    else:
        print(f"  ❌ Full model invertibility FAILED (error >= {tolerance})\n")
        return False


def main():
    print("\n" + "🚀 " * 20)
    print("  RLNF-RRT Flow Model Test Suite")
    print("🚀 " * 20 + "\n")
    
    tests = [
        ("STNet", test_stnet),
        ("CouplingBlock Forward", test_coupling_block_forward),
        ("CouplingBlock Inverse", test_coupling_block_inverse),
        ("ConditionalFlowPlanner Forward", test_conditional_flow_planner_forward),
        ("ConditionalFlowPlanner Sample", test_conditional_flow_planner_sample),
        ("Full Model Invertibility", test_flow_invertibility),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result, None))
        except Exception as e:
            print(f"  ❌ {name} FAILED with exception: {e}\n")
            results.append((name, False, str(e)))
    
    # Summary
    print("=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, r, _ in results if r)
    total = len(results)
    
    for name, result, error in results:
        status = "✅ PASS" if result else "❌ FAIL"
        error_msg = f" ({error})" if error else ""
        print(f"  {status}: {name}{error_msg}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    print("=" * 60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
