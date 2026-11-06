import numpy as np

from scripts.convert_lafan1_to_pmt import wxyz_to_xyzw


def test_mujoco_identity_is_isaac_xyzw_identity():
    assert np.array_equal(wxyz_to_xyzw(np.array([1., 0., 0., 0.])), [0., 0., 0., 1.])


def test_mujoco_quaternion_components_are_reordered_without_sign_change():
    q = np.array([[0.5, 0.1, 0.2, 0.3], [-0.5, 0.1, -0.2, 0.3]])
    assert np.array_equal(wxyz_to_xyzw(q), q[:, [1, 2, 3, 0]])
