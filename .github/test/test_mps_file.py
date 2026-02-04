#!/usr/bin/env python3
"""
Test MPS operators by writing results to file
"""

import torch
import sys
import traceback

def test_mps():
    try:
        # Import our MPS operators
        from xformers.ops.fmha.mps import FwOp, BwOp
        result = "✓ MPS operators imported successfully\n"

        # Check device
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            result += "✓ MPS device available\n"
        else:
            device = torch.device("cpu")
            result += "Using CPU device\n"

        # Create test tensors
        B, M, H, K = 1, 8, 2, 32
        query = torch.randn(B, M, H, K, device=device, dtype=torch.float32)
        key = torch.randn(B, M, H, K, device=device, dtype=torch.float32)
        value = torch.randn(B, M, H, K, device=device, dtype=torch.float32)

        result += f"✓ Created tensors: {query.shape} on {device}\n"

        # Test forward operator
        from xformers.ops.fmha.common import Inputs
        inp = Inputs(query=query, key=key, value=value)

        result += "Testing forward operator...\n"
        output, ctx = FwOp.apply(inp, needs_gradient=False)
        result += f"✓ Forward pass successful: {output.shape}, device: {output.device}\n"

        # Verify output
        assert output.device == device, f"Output device {output.device} != expected {device}"
        assert output.shape == query.shape, f"Output shape {output.shape} != query shape {query.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains inf values"

        # Test backward operator if context was created
        if ctx is not None:
            result += "Testing backward operator...\n"
            grad = torch.randn_like(output)
            grads = BwOp.apply(ctx, inp, grad)

            # Verify gradients
            assert grads.dq is not None, "dq should not be None"
            assert grads.dk is not None, "dk should not be None"
            assert grads.dv is not None, "dv should not be None"
            assert grads.dq.shape == query.shape, f"dq shape {grads.dq.shape} != query shape"
            assert grads.dk.shape == key.shape, f"dk shape {grads.dk.shape} != key shape"
            assert grads.dv.shape == value.shape, f"dv shape {grads.dv.shape} != value shape"
            result += "✓ Backward pass successful\n"

        result += "🎉 MPS operators working!\n"
        return result, True

    except Exception as e:
        error_msg = f"❌ Error: {e}\n{traceback.format_exc()}"
        return error_msg, False

if __name__ == "__main__":
    result, success = test_mps()

    # Write to file in current directory
    output_file = "mps_test_result.txt"
    with open(output_file, 'w') as f:
        f.write(result)

    # Also print to stdout
    print(result)
    print(f"\nResults written to: {output_file}")

    sys.exit(0 if success else 1)
