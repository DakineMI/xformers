#!/usr/bin/env python3
"""
Test full xFormers MPS integration
"""

import pytest
import torch


@pytest.mark.skipif(
    not hasattr(torch.backends, 'mps'),
    reason="PyTorch MPS not available on this system"
)
def test_mps_attention_forward():
    """Test MPS attention forward pass"""
    import xformers.ops as xops

    device = torch.device("mps")
    B, M, H, K = 2, 16, 4, 64
    query = torch.randn(B, M, H, K, device=device, dtype=torch.float32)
    key = torch.randn(B, M, H, K, device=device, dtype=torch.float32)
    value = torch.randn(B, M, H, K, device=device, dtype=torch.float32)

    with torch.no_grad():
        output = xops.memory_efficient_attention(query, key, value)

    assert output.shape == (B, M, H, K)
    assert output.device.type == "mps"


@pytest.mark.skipif(
    not hasattr(torch.backends, 'mps'),
    reason="PyTorch MPS not available on this system"
)
def test_mps_attention_causal_mask():
    """Test MPS attention with causal mask"""
    import xformers.ops as xops

    device = torch.device("mps")
    B, M, H, K = 2, 16, 4, 64
    query = torch.randn(B, M, H, K, device=device, dtype=torch.float32)
    key = torch.randn(B, M, H, K, device=device, dtype=torch.float32)
    value = torch.randn(B, M, H, K, device=device, dtype=torch.float32)

    with torch.no_grad():
        output = xops.memory_efficient_attention(
            query, key, value,
            attn_bias=xops.fmha.attn_bias.LowerTriangularMask()
        )

    assert output.shape == (B, M, H, K)
    assert output.device.type == "mps"


@pytest.mark.skipif(
    not hasattr(torch.backends, 'mps'),
    reason="PyTorch MPS not available on this system"
)
def test_mps_attention_backward():
    """Test MPS attention backward pass with gradients"""
    import xformers.ops as xops

    device = torch.device("mps")
    B, M, H, K = 2, 16, 4, 64
    query = torch.randn(B, M, H, K, device=device, dtype=torch.float32)
    key = torch.randn(B, M, H, K, device=device, dtype=torch.float32)
    value = torch.randn(B, M, H, K, device=device, dtype=torch.float32)

    query = query.requires_grad_(True)
    key = key.requires_grad_(True)
    value = value.requires_grad_(True)

    output = xops.memory_efficient_attention(query, key, value)
    loss = output.sum()

    loss.backward()

    assert query.grad is not None
    assert key.grad is not None
    assert value.grad is not None


@pytest.mark.skipif(
    not hasattr(torch.backends, 'mps'),
    reason="PyTorch MPS not available on this system"
)
def test_mps_attention_dtype():
    """Test MPS attention with different dtypes"""
    import xformers.ops as xops

    device = torch.device("mps")
    B, M, H, K = 2, 16, 4, 64

    for dtype in [torch.float16, torch.float32]:
        query = torch.randn(B, M, H, K, device=device, dtype=dtype)
        key = torch.randn(B, M, H, K, device=device, dtype=dtype)
        value = torch.randn(B, M, H, K, device=device, dtype=dtype)

        with torch.no_grad():
            output = xops.memory_efficient_attention(query, key, value)

        assert output.shape == (B, M, H, K)
        assert output.dtype == dtype


def test_cpu_fallback():
    """Test that CPU fallback works when MPS is not available"""
    import xformers.ops as xops

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        pytest.skip("MPS is available, cannot test CPU fallback")

    device = torch.device("cpu")
    B, M, H, K = 2, 16, 4, 64
    query = torch.randn(B, M, H, K, device=device, dtype=torch.float32)
    key = torch.randn(B, M, H, K, device=device, dtype=torch.float32)
    value = torch.randn(B, M, H, K, device=device, dtype=torch.float32)

    with torch.no_grad():
        output = xops.memory_efficient_attention(query, key, value)

    assert output.shape == (B, M, H, K)
    assert output.device.type == "cpu"
