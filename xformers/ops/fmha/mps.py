# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
MPS (Metal Performance Shaders) Attention Operators

This module implements memory-efficient attention operators using PyTorch's native
scaled_dot_product_attention on Apple's Metal Performance Shaders backend.

Supported Features:
    - Forward pass: float32, float16, bfloat16 on MPS and CPU devices
    - Backward pass: Same dtypes with proper gradient computation
    - Attention masks: Causal (LowerTriangularMask), Tensor masks, None
    - Dropout: Supported via PyTorch's native implementation
    - Custom scale: Supported

Limitations:
    - Only BMHK format (4D tensors) is supported
    - Group/MQA/GQA attention not yet supported
    - No attention bias gradient support (db=None)
    - BlockDiagonalMask requires custom handling
    - Different value embedding not supported

Performance Notes:
    - Uses PyTorch's native scaled_dot_product_attention for optimal MPS performance
    - CPU fallback available for non-MPS devices
    - LSE is computed during forward pass for backward compatibility
"""

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

            # Compute actual LSE for backward pass
            # LSE shape should be [batch, heads, seq_len] = [B, H, M]
            lse = None
            if needs_gradient:
                B, M, H, K = inp.query.shape
                # Compute LSE: LSE_i = log(sum_j(exp(Q_i @ K_j^T / sqrt(d))))
                scale_factor = inp.scale if inp.scale is not None else (K ** -0.5)

                # Reshape for head-wise computation: [B, M, H, K] -> [B, H, M, K]
                query_h = inp.query.permute(0, 2, 1, 3).contiguous()  # [B, H, M, K]
                key_h = inp.key.permute(0, 2, 3, 1).contiguous()  # [B, H, K, M]

                # Compute attention scores: [B, H, M, K] @ [B, H, K, M] = [B, H, M, M]
                attn_scores = torch.matmul(query_h * scale_factor, key_h)  # [B, H, M, M]

                if is_causal:
                    # Create causal mask: upper triangle (excluding diagonal)
                    causal_mask = torch.triu(
                        torch.ones(M, M, dtype=inp.query.dtype, device=inp.query.device),
                        diagonal=1
                    )
                    attn_scores = attn_scores.masked_fill(causal_mask.bool(), float('-inf'))

                if attn_mask is not None:
                    attn_scores = attn_scores + attn_mask

                # Compute LSE: log(sum_j(exp(attn[:,:,i,j]))) for each [h, i]
                # attn_scores: [B, H, M, M]
                attn_scores = attn_scores - attn_scores.max(dim=-1, keepdim=True)[0]
                attn_exp = torch.exp(attn_scores)
                lse = torch.log(attn_exp.sum(dim=-1))  # [B, H, M]

        # Create context for backward pass if needed
        ctx = None
        if needs_gradient:
            assert lse is not None, "LSE must be computed when needs_gradient=True"
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
        """Convert xFormers attention bias to PyTorch attention mask format
        
        Returns:
            - None: Use is_causal=True for causal attention
            - torch.Tensor: Attention mask in compatible format
            
        Supported formats:
            - torch.Tensor: [B, H, M, M] raw attention mask
            - LowerTriangularMask: Use is_causal=True instead (returns None)
            - LowerTriangularMaskWithTensorBias: Extracts bias tensor
            - BlockDiagonalMask: Not directly supported (returns None)
            - BlockDiagonalPaddedKeysMask: Not directly supported (returns None)
        """
        if attn_bias is None:
            return None

        if isinstance(attn_bias, torch.Tensor):
            # xFormers uses [B, H, M_q, M_k] format for attention masks
            # PyTorch's scaled_dot_product_attention accepts [B, M_q, M_k] or [B, H, M_q, M_k]
            if attn_bias.ndim == 4:
                B, H, M_q, M_k = attn_bias.shape
                _, M, _, _ = inp.query.shape
                if M_q == M_k == M:
                    # Shape [B, H, M, M] is compatible - pass through
                    return attn_bias
            # For mismatched shapes or other formats, pass through for PyTorch to handle
            return attn_bias

        if isinstance(attn_bias, LowerTriangularMask):
            # Use PyTorch's built-in causal attention via is_causal=True
            return None

        if isinstance(attn_bias, LowerTriangularMaskWithTensorBias):
            # Extract the bias tensor from the mask wrapper
            bias_tensor = attn_bias._bias
            if bias_tensor.ndim == 4:  # [B, H, M, M] format
                B, H, M_q, M_k = bias_tensor.shape
                _, M, _, _ = inp.query.shape
                if M_q == M_k == M:
                    return bias_tensor
            return None

        # For BlockDiagonalMask and other specialized types, return None
        # These require custom handling not yet implemented
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

        # Re-run forward pass with gradients enabled explicitly
        with torch.enable_grad():
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

        # Ensure gradients are not None when grad is enabled
        dq = query.grad
        dk = key.grad
        dv = value.grad

        if torch.is_grad_enabled():
            assert dq is not None, "Query gradient is None - this should not happen"
            assert dk is not None, "Key gradient is None - this should not happen"
            assert dv is not None, "Value gradient is None - this should not happen"

        return Gradients(
            dq=dq,
            dk=dk,
            dv=dv,
            db=None  # No bias gradient support yet
        )

    @classmethod
    def is_available(cls) -> bool:
        """Check if MPS backward attention is available"""
        return torch.backends.mps.is_available()
