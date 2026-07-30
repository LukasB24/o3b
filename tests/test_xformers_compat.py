"""The Blackwell (sm_120) xformers workaround added on the hc3dv2 fork.

`sdpa_memory_efficient_attention` replaces a fused xformers kernel that this GPU
cannot dispatch. It is not a stub: it stands in for real attention inside the
DenseMatcher stack (ldm's UNet and SD's VAE), so if its layout handling is wrong
the features change silently and every downstream PCK number is quietly wrong.
These tests pin it against a reference implementation on all layouts that stack
actually uses.

Run on any GPU (or none) — the shim is exercised directly rather than through
the capability-gated monkeypatch.
"""

from __future__ import annotations

import math

import pytest
import torch
from o3b.model.xformers_compat import (
    needs_sdpa_fallback,
    sdpa_memory_efficient_attention,
)


def reference_attention(q, k, v):
    """Attention in xformers' BMHK layout: (B, M, H, K), attending over M."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("bmhk,bnhk->bhmn", q, k) * scale
    return torch.einsum("bhmn,bnhk->bmhk", scores.softmax(dim=-1), v)


# float64 throughout: this is an exactness check, not a tolerance check.
def _rand(*shape):
    return torch.randn(*shape, dtype=torch.float64)


@pytest.mark.parametrize(
    "b,m,h,k",
    [
        (2, 37, 1, 512),  # SD VAE self-attention — head_dim 512 is why fa2F refuses
        (2, 37, 8, 64),   # ordinary multi-head
    ],
    ids=["vae_self_attn_headdim512", "multi_head"],
)
def test_4d_layout_matches_reference(b, m, h, k):
    torch.manual_seed(0)
    q, kk, v = (_rand(b, m, h, k) for _ in range(3))

    out = sdpa_memory_efficient_attention(q, kk, v)

    assert out.shape == (b, m, h, k)
    assert torch.allclose(out, reference_attention(q, kk, v), atol=1e-12)


@pytest.mark.parametrize(
    "m,n", [(37, 37), (37, 77)], ids=["self_attn", "cross_attn_m_ne_n"]
)
def test_3d_layout_matches_reference(m, n):
    """ldm folds heads into the batch dim, giving (B*H, M, K) — one head each."""
    torch.manual_seed(0)
    bh, k = 16, 64
    q, kk, v = _rand(bh, m, k), _rand(bh, n, k), _rand(bh, n, k)

    out = sdpa_memory_efficient_attention(q, kk, v)

    expected = reference_attention(
        q.unsqueeze(2), kk.unsqueeze(2), v.unsqueeze(2)
    ).squeeze(2)
    assert out.shape == (bh, m, k)
    assert torch.allclose(out, expected, atol=1e-12)


def test_3d_attn_bias_is_applied_on_the_right_axis():
    """A 3-D bias is (B, M, N) and must gain a head axis alongside q.

    Without the unsqueeze the bias right-aligns to (1, B, M, N) and broadcasts
    to (B, B, M, N): a shape error for B > 1, and correct only by accident at
    B == 1. B is 4 here so the regression would actually be caught.
    """
    torch.manual_seed(0)
    b, m, n, k = 4, 6, 6, 8
    q, kk, v = _rand(b, m, k), _rand(b, n, k), _rand(b, n, k)

    # a bias that differs per batch element, so a misaligned broadcast shows up
    bias = torch.zeros(b, m, n, dtype=torch.float64)
    for i in range(b):
        bias[i, :, i] = -float("inf")  # forbid a different key in each element

    out = sdpa_memory_efficient_attention(q, kk, v, attn_bias=bias)

    scale = 1.0 / math.sqrt(k)
    scores = (q @ kk.transpose(-2, -1)) * scale + bias
    expected = scores.softmax(dim=-1) @ v
    assert torch.allclose(out, expected, atol=1e-12)


def test_4d_attn_bias_is_passed_through():
    torch.manual_seed(0)
    b, m, n, h, k = 2, 5, 5, 3, 8
    q, kk, v = _rand(b, m, h, k), _rand(b, n, h, k), _rand(b, n, h, k)
    bias = _rand(b, h, m, n)

    out = sdpa_memory_efficient_attention(q, kk, v, attn_bias=bias)

    scale = 1.0 / math.sqrt(k)
    scores = torch.einsum("bmhk,bnhk->bhmn", q, kk) * scale + bias
    expected = torch.einsum("bhmn,bnhk->bmhk", scores.softmax(dim=-1), v)
    assert torch.allclose(out, expected, atol=1e-12)


def test_unsupported_attn_bias_type_is_rejected():
    """xformers' structured bias objects have no SDPA equivalent here.

    Better to raise than to silently drop the mask and return plain attention.
    """
    q, kk, v = _rand(2, 4, 8), _rand(2, 4, 8), _rand(2, 4, 8)

    class LowerTriangularMask:  # stand-in for the xformers bias objects
        pass

    with pytest.raises(NotImplementedError, match="attn_bias"):
        sdpa_memory_efficient_attention(q, kk, v, attn_bias=LowerTriangularMask())


def test_capability_gate_leaves_working_gpus_alone():
    """The shim costs memory, so it must only engage where the fused path fails."""
    result = needs_sdpa_fallback()
    assert isinstance(result, bool)
    if torch.cuda.is_available():
        assert result == (torch.cuda.get_device_capability()[0] >= 12)
    else:
        assert result is False, "no CUDA means no fused kernel to work around"
