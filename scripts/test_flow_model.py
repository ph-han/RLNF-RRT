#!/usr/bin/env python
"""
Smoke tests for FiLM-based flow components.
"""
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rlnf_rrt.models.coupling_block import CouplingBlock, STNetFiLM
from rlnf_rrt.models.conditional_flow_planner import ConditionalFlowPlanner


def test_stnet_film():
    batch_size = 4
    seq_len = 128
    hidden_dim = 64
    map_dim = 64
    sg_dim = 7

    net = STNetFiLM(sg_dim=sg_dim, map_dim=map_dim, hidden_dim=hidden_dim)
    z1 = torch.randn(batch_size, seq_len, 1)
    sg = torch.randn(batch_size, seq_len, sg_dim)
    map_feat = torch.randn(batch_size, map_dim)
    out = net(z1, sg, map_feat)
    assert out.shape == (batch_size, seq_len, 1)
    return True


def test_coupling_block_forward_inverse():
    batch_size = 4
    seq_len = 128
    block = CouplingBlock(sg_dim=7, map_dim=64, hidden_dim=64)
    block.eval()

    x = torch.randn(batch_size, seq_len, 2)
    sg = torch.randn(batch_size, seq_len, 7)
    map_feat = torch.randn(batch_size, 64)

    z, log_det = block(x, sg, map_feat)
    x_recon = block.inverse(z, sg, map_feat)

    assert z.shape == x.shape
    assert log_det.shape == (batch_size,)
    assert x_recon.shape == x.shape
    assert (x - x_recon).abs().max().item() < 1e-5
    return True


def test_planner_forward_sample():
    batch_size = 2
    seq_len = 128
    num_samples = 100

    model = ConditionalFlowPlanner(
        num_blocks=2,
        sg_dim=2,
        map_embed_dim=64,
        hidden_dim=64,
    )
    model.eval()

    gt_trajs = torch.randn(batch_size, seq_len, 2)
    map_img = torch.randn(batch_size, 1, 64, 64)
    start = torch.rand(batch_size, 2)
    goal = torch.rand(batch_size, 2)

    z, log_det = model(gt_trajs, map_img, start, goal)
    samples = model.sample(map_img, start, goal, num_samples=num_samples)
    intermediates = model.sample_with_intermediates(map_img, start, goal, num_samples=num_samples)
    forward_intermediates = model.forward_with_intermediates(gt_trajs, map_img, start, goal)

    assert z.shape == gt_trajs.shape
    assert log_det.shape == (batch_size,)
    assert samples.shape == (batch_size, num_samples, 2)
    assert len(intermediates) == 1 + len(model.flow_model)
    assert len(forward_intermediates) == 1 + len(model.flow_model)
    return True


def main():
    tests = [
        ("STNetFiLM", test_stnet_film),
        ("Coupling forward/inverse", test_coupling_block_forward_inverse),
        ("Planner forward/sample", test_planner_forward_sample),
    ]

    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}")

    print(f"\nSummary: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
