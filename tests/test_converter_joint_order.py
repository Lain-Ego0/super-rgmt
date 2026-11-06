import numpy as np

from scripts.convert_lafan1_to_pmt import ISAAC_JOINTS, JOINTS, reorder_joint_channels


def test_lafan_joint_channels_are_permuted_to_runtime_articulation_order():
    source = np.arange(2 * len(JOINTS), dtype=np.float32).reshape(2, -1)
    reordered = reorder_joint_channels(source)
    expected = [JOINTS.index(name) for name in ISAAC_JOINTS]
    np.testing.assert_array_equal(reordered, source[:, expected])
    assert reordered.shape == source.shape


def test_reorder_rejects_incompatible_joint_sets():
    try:
        reorder_joint_channels(np.zeros((1, len(JOINTS))), JOINTS, JOINTS[:-1] + ["unknown"])
    except ValueError as exc:
        assert "same unique joints" in str(exc)
    else:
        raise AssertionError("mismatched joint sets must be rejected")
