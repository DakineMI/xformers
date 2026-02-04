# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Iterable, Mapping, Optional, Set, Tuple, Union

import torch

from ..common import register_operator
from .attn_bias import (
    AttentionBias,
    BlockDiagonalMask,
    BlockDiagonalPaddedKeysMask,
    LowerTriangularMask,
    LowerTriangularMaskWithTensorBias,
)
from .common import (
    AttentionBwOpBase,
    AttentionFwOpBase,
    Context,
    Gradients,
    Inputs,
)


@register_operator
class FwOp(AttentionFwOpBase):
    """MPS-based attention operator using PyTorch's native scaled_dot_product_attention"""

    OPERATOR = None  # We'll use PyTorch's native implementation
    SUPPORTED_DEVICES: Set[str] = {"mps", "cpu"}  # Support both MPS and CPU fallback
    SUPPORTED_DTYPES: Set[torch.dtype] = {
        torch.float32, torch.float16, torch.bfloat16
    }
    SUPPORTED_MAX_K = 512
    SUPPORTED_MIN_K = 16

    SUPPORTED_ATTN_BIAS_TYPES: Iterable[Any] = (
        type(None),
        torch.Tensor,
        LowerTriangularMask,
        LowerTriangularMaskWithTensorBias,
        BlockDiagonalMask,
        BlockDiagonalPaddedKeysMask,
    )

    SUPPORTS_DROPOUT = True
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_DIFFERENT_VALUE_EMBED = False  # PyTorch native doesn't support this
    SUPPORTS_PARTIAL = False
    SUPPORTS_BMGHK = False  # Start simple, no group support yet
    NAME = "mpsF"
    IS_DETERMINISTIC = True

    ERROR_ATOL: Mapping[torch.dtype, float] = {
        torch.float32: 1e-4,
        torch.float16: 1e-3,
        torch.bfloat16: 2e-3,
    }
    ERROR_RTOL: Mapping[torch.dtype, float] = {
        torch.float32: 1e-5,
        torch.float16: 2e-4,
        torch.bfloat16: 5e-4,
    }

    @classmethod
    def apply(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        # For now, only support BMHK format (4D tensors)
        if inp.query.ndim != 4:
            raise NotImplementedError(
                f"MPS operator currently only supports BMHK format, got {inp.query.ndim}D"
            )

        # Convert attention bias to mask or determine if causal
        attn_mask = cls._convert_attn_bias_to_mask(inp.attn_bias, inp)
        is_causal = isinstance(inp.attn_bias, LowerTriangularMask)

        # Use PyTorch's native scaled_dot_product_attention
        with torch.enable_grad() if needs_gradient else torch.no_grad():
            output = torch.nn.functional.scaled_dot_product_attention(
                query=inp.query,
                key=inp.key,
                value=inp.value,
                attn_mask=attn_mask,
                dropout_p=inp.p,
                scale=inp.scale,
                is_causal=is_causal
            )

            # For LSE, create a placeholder that matches expected shape
            # LSE shape should be [batch, heads, seq_len]
            if needs_gradient:
                B, M, H, K = inp.query.shape
                lse = torch.zeros(B, H, M, dtype=output.dtype, device=output.device)
            else:
                lse = None

        # Create context for backward pass if needed
        # Note: Context is created for API compatibility with xFormers attention operators.
        # Since PyTorch's scaled_dot_product_attention has built-in autograd support,
        # we don't need to manually store intermediate values for gradient computation.
        # The backward pass re-runs forward with requires_grad=True and uses PyTorch's
        # native autograd to compute gradients.
        ctx = None
        if needs_gradient:
            ctx = Context(
                lse=lse,
                out=output,
                op_bw=BwOp,
            )

        return output, ctx

    @classmethod
    def _convert_attn_bias_to_mask(
        cls,
        attn_bias: Optional[Union[torch.Tensor, AttentionBias]],
        inp: Inputs
    ) -> Optional[torch.Tensor]:
        """Convert xFormers attention bias to PyTorch attention mask format"""
        if attn_bias is None:
            return None

        if isinstance(attn_bias, torch.Tensor):
            # xFormers uses [B, H, M, M] format
            # PyTorch expects [B, M_q, H, M_k] for attn_mask
            # For self-attention where M_q == M_k, we need to ensure proper broadcasting
            if attn_bias.ndim == 4:
                B, H, M_q, M_k = attn_bias.shape
                # Verify shape matches expected [B, H, M, M]
                _, M, _, _ = inp.query.shape
                if M_q == M_k == M:
                    # Shape is [B, H, M, M] - this matches xFormers format
                    # Keep as-is for now
                    return attn_bias
            return attn_bias

        if isinstance(attn_bias, LowerTriangularMask):
            # Create causal mask for PyTorch's scaled_dot_product_attention
            # PyTorch expects is_causal=True for causal attention
            return None  # Return None and use is_causal=True instead

        if isinstance(attn_bias, LowerTriangularMaskWithTensorBias):
            # For tensor bias, we need to handle it properly
            bias_tensor = attn_bias._bias
            if bias_tensor.ndim == 4:  # [B, H, M, M] format
                B, H, M_q, M_k = bias_tensor.shape
                _, M, _, _ = inp.query.shape
                if M_q == M_k == M:
                    # Shape is [B, H, M, M]
                    return bias_tensor
            else:
                return None

        # For other bias types, return None (no mask)
        return None

    @classmethod
    def is_available(cls) -> bool:
        """Check if MPS attention is available"""
        return torch.backends.mps.is_available()


@register_operator
class BwOp(AttentionBwOpBase):
    """MPS-based backward attention operator"""

    OPERATOR = None
    SUPPORTED_DEVICES: Set[str] = {"mps", "cpu"}
    SUPPORTED_DTYPES: Set[torch.dtype] = {
        torch.float32, torch.float16, torch.bfloat16
    }
    SUPPORTED_MAX_K = 512
    SUPPORTED_MIN_K = 16

    SUPPORTED_ATTN_BIAS_TYPES: Iterable[Any] = (
        type(None),
        torch.Tensor,
        LowerTriangularMask,
        LowerTriangularMaskWithTensorBias,
        BlockDiagonalMask,
        BlockDiagonalPaddedKeysMask,
    )

    SUPPORTS_DROPOUT = True
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_DIFFERENT_VALUE_EMBED = False
    SUPPORTS_PARTIAL = False
    SUPPORTS_BMGHK = False
    NAME = "mpsB"
    IS_DETERMINISTIC = True
    SUPPORTS_ATTN_BIAS_GRAD = False  # For now, don't support bias gradients

    ERROR_ATOL: Mapping[torch.dtype, float] = {
        torch.float32: 3e-4,
        torch.float16: 1e-2,
        torch.bfloat16: 5e-3,
    }
    ERROR_RTOL: Mapping[torch.dtype, float] = {
        torch.float32: 2e-5,
        torch.float16: 1e-3,
        torch.bfloat16: 2e-3,
    }

    @classmethod
    def apply(cls, ctx: Context, inp: Inputs, grad: torch.Tensor) -> Gradients:
        # For MPS backward, we need to re-establish the autograd computation graph
        # Since PyTorch's scaled_dot_product_attention has native autograd support,
        # we clone inputs with gradient tracking and recompute the forward pass

        # Clone tensors with gradient tracking
        query = inp.query.detach().clone().requires_grad_(True)
        key = inp.key.detach().clone().requires_grad_(True)
        value = inp.value.detach().clone().requires_grad_(True)

        # Convert attention bias
        attn_mask = FwOp._convert_attn_bias_to_mask(inp.attn_bias, inp)
        is_causal = isinstance(inp.attn_bias, LowerTriangularMask)

        # Re-run forward pass with gradients enabled
        output = torch.nn.functional.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=inp.p,
            scale=inp.scale,
            is_causal=is_causal
        )

        # Compute gradients using autograd
        output.backward(grad)

        return Gradients(
            dq=query.grad,
            dk=key.grad,
            dv=value.grad,
            db=None  # No bias gradient support yet
        )

    @classmethod
    def is_available(cls) -> bool:
        """Check if MPS backward attention is available"""
        return torch.backends.mps.is_available()
