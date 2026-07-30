"""The `mesh_type` feature recipe, as extended on the hc3dv2 fork.

A `mesh_type` string like `mc32_vuni100_r256_fpfdino` *is* the recipe the
benchmark runs on: marching-cubes resolution, view rig, and which model bakes
the per-vertex descriptors. This module covers the parts the fork adds — the
`pfdino` fusion hook and the self-rendering dispatch — plus the parser they both
hang off.

All pure: no checkpoints, no GPU, no dataset. The models themselves are
upstream/third-party and are not under test here; only the dispatch and
combination logic added on this branch is.
"""

from __future__ import annotations

import o3b.data.datatypes.mesh as mesh_mod
import pytest
import torch

# Captured before any monkeypatching. `_extract_vert_feats` recurses through its
# own module global for fusion sub-encoders, so tests patch the module attribute
# to intercept the sub-calls while still invoking the real function under test
# through this reference.
_EXTRACT = mesh_mod._extract_vert_feats


class _FakeMesh:
    """Stand-in for o3b's Mesh — the paths under test never read its fields."""

    verts = None
    faces = None


# ── mesh_type parsing ─────────────────────────────────────────────────────────


def test_parses_a_full_recipe():
    assert mesh_mod._parse_mc_type("mc32_vuni100_r256_fpfdino") == {
        "res": 32,
        "view_sampling": "uni",
        "n_views": 100,
        "resolution": 256,
        "feature_model": "pfdino",
    }


def test_bare_marching_cubes_type_has_no_rig_or_model():
    """`mc32` is valid on its own — geometry only, nothing baked onto it."""
    parsed = mesh_mod._parse_mc_type("mc32")
    assert parsed["res"] == 32
    assert parsed["view_sampling"] is None
    assert parsed["n_views"] is None
    assert parsed["feature_model"] is None


def test_rejects_an_unparseable_type():
    with pytest.raises(ValueError, match="Cannot parse mesh type"):
        mesh_mod._parse_mc_type("not_a_mesh_type")


# ── pfdino fusion hook ────────────────────────────────────────────────────────


@pytest.fixture
def fused_output(monkeypatch):
    """Run the fusion branch over two sub-blocks of wildly different magnitude.

    Returns `(out, calls, dims)` where `out` is the fused (V, sum dims) tensor
    and `calls` records which sub-encoders ran, in order.
    """
    calls: list[str] = []
    V = 5
    blocks = {
        # deliberately mismatched scales: without per-block normalisation the
        # first would completely dominate the second.
        "partfield": torch.full((V, 4), 100.0),
        "dinov2s": torch.full((V, 3), 0.01),
    }

    def fake_extract(mesh, n_views, resolution, feature_model_name):
        calls.append(feature_model_name)
        return blocks[feature_model_name].clone()

    monkeypatch.setattr(mesh_mod, "_extract_vert_feats", fake_extract)

    out = _EXTRACT(_FakeMesh(), n_views=4, resolution=256, feature_model_name="pfdino")
    return out, calls, (V, 4, 3)


def test_fusion_runs_both_sub_encoders_in_declared_order(fused_output):
    _, calls, _ = fused_output
    assert calls == ["partfield", "dinov2s"]


def test_fusion_concatenates_sub_encoder_dims(fused_output):
    out, _, (V, d_pf, d_dino) = fused_output
    assert out.shape == (V, d_pf + d_dino)


def test_fusion_l2_normalises_each_block(fused_output):
    """Each block must contribute unit norm per vertex.

    This is what makes the fusion work. The benchmark matches vertices by
    euclidean nearest-neighbour over the concatenated vector, so without
    per-block normalisation PartField's larger-magnitude 448 dims would swamp
    DINOv2's 384 and the fused result would collapse toward PartField alone --
    silently, as a lower PCK rather than an error.
    """
    out, _, (V, d_pf, _) = fused_output
    pf_block, dino_block = out[:, :d_pf], out[:, d_pf:]

    ones = torch.ones(V)
    assert torch.allclose(pf_block.norm(dim=-1), ones, atol=1e-5)
    assert torch.allclose(dino_block.norm(dim=-1), ones, atol=1e-5)


def test_fusion_erases_the_input_scale_difference(fused_output):
    """A 10,000x magnitude gap on input must not survive into the output."""
    out, _, (_, d_pf, _) = fused_output
    pf_mean = out[:, :d_pf].abs().mean()
    dino_mean = out[:, d_pf:].abs().mean()
    assert pf_mean / dino_mean == pytest.approx(1.0, abs=0.3)


# ── self-rendering dispatch ───────────────────────────────────────────────────


@pytest.mark.parametrize("model_name", ["partfield", "cosmo3d"])
def test_self_rendering_models_bypass_the_view_rig(monkeypatch, model_name):
    """These consume the surface directly, so the `vuniN` rig must be skipped.

    Both are registered in `_SELF_RENDERING`; if one were dropped, the mesh
    would be rendered from N viewpoints and the features sampled from those
    images instead — wasted work and wrong descriptors. The rig is replaced
    with a tripwire so taking that path fails loudly.
    """
    import o3b.model.model as model_mod
    from o3b.data import viz

    def _tripwire(*args, **kwargs):
        raise AssertionError(f"{model_name} must not go through the render rig")

    monkeypatch.setattr(viz, "sample_uniform_viewpoints", _tripwire, raising=False)
    monkeypatch.setattr(viz, "render_mesh_from_viewpoints", _tripwire, raising=False)

    V, F = 6, 8

    class _FakeModel:
        def eval(self):
            return self

        def __call__(self, batch):
            batch.verts3d_feats = torch.zeros(1, V, F)
            return batch

    monkeypatch.setattr(
        model_mod.OD3D_Model,
        "create_by_name",
        staticmethod(lambda name, config=None: _FakeModel()),
    )

    out = _EXTRACT(
        _FakeMesh(), n_views=4, resolution=256, feature_model_name=model_name
    )
    assert out.shape == (V, F)
