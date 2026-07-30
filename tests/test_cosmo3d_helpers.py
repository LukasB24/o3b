"""Helpers added on the hc3dv2 fork for the CoSMo3D feature model.

Two pieces of our own logic, neither needing the checkpoint or a GPU:

  * `_inject_flash_attn_stub` — the Blackwell/CUDA-13 workaround. flash_attn
    cannot be compiled there, but PointTransformerV3 imports it
    unconditionally, so we install a PyTorch-native shim.
  * `_sample_colored_surface` — surface sampling with colour, including the
    fallbacks for meshes that carry no usable texture.

The CoSMo3D backbone itself is third-party and is not under test.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from o3b.model.cosmo3d.model import CoSMo3DModel, _inject_flash_attn_stub

# ── flash_attn stub ───────────────────────────────────────────────────────────


@pytest.fixture
def stubbed_flash_attn(monkeypatch):
    """Force the absent-flash_attn path and return the installed shim."""
    monkeypatch.setitem(sys.modules, "flash_attn", None)  # makes `import` raise
    monkeypatch.delitem(sys.modules, "flash_attn.flash_attn_interface", raising=False)
    _inject_flash_attn_stub()
    return sys.modules["flash_attn"]


def test_real_flash_attn_is_left_alone(monkeypatch):
    """When the real package imports, the shim must not replace it."""
    sentinel = types.ModuleType("flash_attn")
    monkeypatch.setitem(sys.modules, "flash_attn", sentinel)

    _inject_flash_attn_stub()

    assert sys.modules["flash_attn"] is sentinel


def test_stub_exposes_both_entry_points(stubbed_flash_attn):
    """PointTransformerV3 imports these two names, from either module path."""
    for module_name in ("flash_attn", "flash_attn.flash_attn_interface"):
        module = sys.modules[module_name]
        assert hasattr(module, "flash_attn_func")
        assert hasattr(module, "flash_attn_varlen_qkvpacked_func")


def test_stub_matches_reference_attention(stubbed_flash_attn):
    """The shim must compute real attention, not merely be importable."""
    torch.manual_seed(0)
    B, S, H, D = 2, 4, 3, 8
    q, k, v = (torch.randn(B, S, H, D) for _ in range(3))

    out = stubbed_flash_attn.flash_attn_func(q, k, v)

    # reference: softmax(q @ k^T / sqrt(d)) @ v, computed head-first
    qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))
    scores = (qt @ kt.transpose(-2, -1)) * (D**-0.5)
    expected = (scores.softmax(dim=-1) @ vt).transpose(1, 2)

    assert out.shape == (B, S, H, D)
    assert torch.allclose(out, expected, atol=1e-5)


def test_varlen_stub_keeps_sequences_independent(stubbed_flash_attn):
    """Packed sequences must not attend across their boundaries.

    `cu_seqlens` marks where each sequence starts; if the shim ignored it and
    attended over the whole packed buffer, tokens would leak between point
    clouds in a batch.
    """
    torch.manual_seed(0)
    lengths = [3, 5]
    total, H, D = sum(lengths), 2, 4
    qkv = torch.randn(total, 3, H, D)
    cu_seqlens = torch.tensor([0, 3, 8])

    packed = stubbed_flash_attn.flash_attn_varlen_qkvpacked_func(
        qkv, cu_seqlens, max_seqlen=max(lengths)
    )
    assert packed.shape == (total, H, D)

    # each segment must equal attention run on that segment alone
    offset = 0
    for length in lengths:
        segment = qkv[offset : offset + length]
        alone = stubbed_flash_attn.flash_attn_varlen_qkvpacked_func(
            segment, torch.tensor([0, length]), max_seqlen=length
        )
        assert torch.allclose(packed[offset : offset + length], alone, atol=1e-6)
        offset += length


# ── coloured surface sampling ─────────────────────────────────────────────────


def test_vertex_colours_are_scaled_to_unit_range():
    """trimesh reports colours as 0-255 bytes; the backbone wants [0, 1]."""
    import trimesh

    box = trimesh.creation.box()
    box.visual = trimesh.visual.ColorVisuals(
        mesh=box,
        vertex_colors=np.tile([255, 0, 0, 255], (len(box.vertices), 1)),
    )

    n_pts = 32
    pts, rgb, nrm, has_texture = CoSMo3DModel._sample_colored_surface(box, n_pts)

    assert pts.shape == (n_pts, 3)
    assert rgb.shape == (n_pts, 3)
    assert nrm.shape == (n_pts, 3)
    assert rgb.max() <= 1.0, "colours must be scaled out of 0-255 byte range"
    assert np.allclose(rgb[:, 0], 1.0) and np.allclose(rgb[:, 1:], 0.0)
    assert has_texture, "a coloured mesh must not be reported as untextured"


def test_sampling_is_deterministic_for_a_fixed_seed():
    """PCK is a benchmark number; identical inputs must give identical samples."""
    import trimesh

    sphere = trimesh.creation.icosphere(subdivisions=2)

    first, _, _, _ = CoSMo3DModel._sample_colored_surface(sphere, 64, seed=0)
    again, _, _, _ = CoSMo3DModel._sample_colored_surface(sphere, 64, seed=0)
    other, _, _, _ = CoSMo3DModel._sample_colored_surface(sphere, 64, seed=1)

    assert np.array_equal(first, again), "same seed must reproduce the same samples"
    assert not np.array_equal(first, other), "a different seed should actually resample"


def test_untextured_mesh_falls_back_to_mid_grey(monkeypatch):
    """With no usable colour anywhere, samples must be flat mid-grey.

    This mirrors the encoder's flat fallback: a mesh with no texture still has
    to produce a colour channel, and 0.5 keeps it neutral rather than biasing
    the backbone toward black or white.
    """
    import trimesh

    n_pts = 8
    sampled_pts = np.zeros((n_pts, 3), dtype=np.float32)
    sampled_faces = np.zeros(n_pts, dtype=np.int64)

    def fake_sample_surface(mesh, count, sample_color=False, **kwargs):
        if sample_color:
            raise ValueError("mesh carries no sampleable colour")
        return sampled_pts, sampled_faces

    monkeypatch.setattr(trimesh.sample, "sample_surface", fake_sample_surface)

    class _ColourlessVisual:
        def to_color(self):
            raise ValueError("nothing to convert")

    mesh = SimpleNamespace(
        vertices=np.zeros((4, 3), dtype=np.float32),
        face_normals=np.tile([0.0, 0.0, 1.0], (2, 1)).astype(np.float32),
        visual=_ColourlessVisual(),
    )

    _, rgb, _, has_texture = CoSMo3DModel._sample_colored_surface(mesh, n_pts)

    assert rgb.shape == (n_pts, 3)
    assert np.allclose(rgb, 0.5)
    assert not has_texture, (
        "the caller relies on this flag to warn that a use_texture=True run "
        "silently became the grey ablation"
    )
