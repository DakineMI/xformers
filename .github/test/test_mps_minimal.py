#!/usr/bin/env python3
"""
Minimal test for MPS operators
"""

import torch
import sys

def main():
    print("Testing MPS operators...")

    try:
        return _run_mps_test()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def _run_mps_test():
    """Run MPS operator test with assertions"""
    # Import our MPS operators
    from xformers.ops.fmha.mps import FwOp, BwOp
    print("✓ MPS operators imported successfully")

    # Check if MPS is available
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ MPS device available")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    # Create simple test tensors
    B, M, H, K = 1, 8, 2, 32
    query = torch.randn(B, M, H, K, device=device, dtype=torch.float32)
    key = torch.randn(B, M, H, K, device=device, dtype=torch.float32)
    value = torch.randn(B, M, H, K, device=device, dtype=torch.float32)

    print(f"✓ Created tensors: {query.shape} on {device}")

    # Test forward operator
    from xformers.ops.fmha.common import Inputs
    inp = Inputs(query=query, key=key, value=value)

    print("Testing forward operator...")
    output, ctx = FwOp.apply(inp, needs_gradient=False)
    print(f"✓ Forward pass successful: {output.shape}, device: {output.device}")

    # Verify output
    assert output.device == device, f"Output device {output.device} != expected {device}"
    assert output.shape == query.shape, f"Output shape {output.shape} != query shape {query.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains inf values"

    print("🎉 MPS operators working!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
