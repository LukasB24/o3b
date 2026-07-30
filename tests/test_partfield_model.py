"""Config contract for the PartField feature model added on the hc3dv2 fork.

Only the wiring is checked here — constructing the model is cheap because the
1.2 GB Objaverse checkpoint is fetched lazily by `_ensure_ckpt`, not in
`__init__`. The PartField network itself is third-party and is not under test.
"""

from __future__ import annotations

from o3b.model.partfield.model import PartFieldModel


def test_out_dim_drops_the_sdf_channels():
    """Features are the triplane minus its SDF head: 512 - 64 = 448.

    The bake writes `out_dim`-wide descriptors and the fusion hook concatenates
    them with DINOv2's 384, so this width is what `fpartfield` and `fpfdino`
    both depend on.
    """
    model = PartFieldModel()

    assert model.out_dim == model.triplane_channels_high - model.sdf_channels
    assert model.out_dim == 448


def test_defaults_match_the_released_checkpoint():
    """Triplane geometry is fixed by the weights; drifting it breaks loading."""
    model = PartFieldModel()

    assert model.triplane_channels_low == 128
    assert model.triplane_channels_high == 512
    assert model.sdf_channels == 64
    assert model.pvcnn_z_triplane_channels == 256
    assert model.pvcnn_z_triplane_resolution == 128


def test_construction_does_not_touch_the_checkpoint(tmp_path):
    """`__init__` must stay cheap — no download, no 1.2 GB load."""
    missing = tmp_path / "definitely-absent.ckpt"

    model = PartFieldModel(ckpt_path=str(missing))

    assert model.ckpt_path == str(missing)
    assert not missing.exists()
