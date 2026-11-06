from types import SimpleNamespace

import torch
import pytest

from extreme_rgmt.mdp import feet_slip


def test_feet_slip_penalizes_only_contacting_feet_horizontal_speed():
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "contact": SimpleNamespace(
                    data=SimpleNamespace(
                        net_forces_w=torch.tensor([[[0.0, 0.0, 2.0], [0.0, 0.0, 0.2]]])
                    )
                )
            },
            __getitem__=lambda self, key: None,
        ),
        num_envs=1,
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(body_lin_vel_w=torch.tensor([[[3.0, 4.0, 0.0], [6.0, 8.0, 0.0]]]))
    )

    class Scene:
        sensors = env.scene.sensors

        def __getitem__(self, key):
            assert key == "robot"
            return robot

    env.scene = Scene()
    sensor_cfg = SimpleNamespace(name="contact", body_ids=torch.tensor([0, 1]))
    asset_cfg = SimpleNamespace(name="robot", body_ids=torch.tensor([0, 1]))
    result = feet_slip(env, sensor_cfg, asset_cfg, contact_force_threshold=1.0)
    torch.testing.assert_close(result, torch.tensor([5.0]))


def test_paper_proprio_noise_respects_table_ii_ranges_and_leaves_action_clean():
    from extreme_rgmt.mdp import add_paper_proprio_noise

    observation = torch.zeros(4, 93)
    torch.manual_seed(7)
    noisy = add_paper_proprio_noise(observation)
    bounds = torch.tensor([0.05] * 3 + [0.2] * 3 + [0.01] * 29 + [0.5] * 29 + [0.0] * 29)
    assert torch.all((noisy - observation).abs() <= bounds + 1e-7)
    assert torch.count_nonzero(noisy[..., 64:]) == 0


def test_paper_proprio_noise_rejects_wrong_layout():
    from extreme_rgmt.mdp import add_paper_proprio_noise

    with pytest.raises(ValueError, match="expected 93D"):
        add_paper_proprio_noise(torch.zeros(2, 64))


def test_paper_command_noise_matches_all_table_ii_channel_ranges():
    from extreme_rgmt.mdp import add_paper_command_noise

    command = torch.zeros(3, 21, 38)
    noisy = add_paper_command_noise(command)
    bounds = torch.tensor([0.5] * 3 + [0.52] * 3 + [0.05] * 3 + [0.1] * 29)
    assert torch.all((noisy - command).abs() <= bounds + 1e-7)
    assert torch.count_nonzero(noisy[..., 9:]) > 0
