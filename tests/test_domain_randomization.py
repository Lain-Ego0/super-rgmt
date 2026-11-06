from types import SimpleNamespace

from extreme_rgmt.domain_randomization import configure_paper_domain_randomization


class Event:
    def __init__(self, *, func, mode, params):
        self.func = func
        self.mode = mode
        self.params = params


class SceneEntity:
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs


def test_paper_randomization_overrides_friction_and_adds_mass_gain_armature_terms():
    existing = lambda: None
    cfg = SimpleNamespace(
        events=SimpleNamespace(
            physics_material=SimpleNamespace(params={"static_friction_range": (0.3, 1.6)}),
            base_com=existing,
            add_joint_default_pos=existing,
            push_robot=SimpleNamespace(interval_range_s=(1.0, 3.0)),
        )
    )
    mdp = SimpleNamespace(
        randomize_rigid_body_mass="mass",
        randomize_actuator_gains="gains",
        randomize_joint_parameters="joint_params",
    )

    configure_paper_domain_randomization(
        cfg, event_term_cls=Event, scene_entity_cfg_cls=SceneEntity, mdp=mdp,
        motor_strength_func="motor_strength",
    )

    assert cfg.events.physics_material.params["static_friction_range"] == (0.10, 1.75)
    assert cfg.events.physics_material.params["dynamic_friction_range"] == (0.10, 1.75)
    assert cfg.events.add_base_mass.params["mass_distribution_params"] == (-3.0, 6.0)
    assert cfg.events.randomize_motor_strength.params["strength_scale_range"] == (0.8, 1.2)
    assert cfg.events.randomize_pd_gains.params["stiffness_distribution_params"] == (0.8, 1.2)
    assert cfg.events.randomize_pd_gains.params["damping_distribution_params"] == (0.8, 1.2)
    assert cfg.events.randomize_armature.params["armature_distribution_params"] == (1.0, 1.05)
    assert cfg.events.base_com is existing
    assert cfg.events.add_joint_default_pos is existing
    assert cfg.events.push_robot.interval_range_s == (1.0, 3.0)


def test_motor_strength_scales_effort_limits_per_environment():
    import torch
    from extreme_rgmt.mdp import randomize_motor_strength

    class Robot:
        device = "cpu"

        def __init__(self):
            self.data = SimpleNamespace(joint_effort_limits=SimpleNamespace(
                torch=torch.full((4, 3), 100.0)
            ))
            self.written = None

        def write_joint_effort_limit_to_sim_index(self, **kwargs):
            self.written = kwargs

    robot = Robot()
    env = SimpleNamespace(num_envs=4, scene={"robot": robot})
    asset_cfg = SimpleNamespace(name="robot", joint_ids=[0, 1, 2])
    torch.manual_seed(11)
    randomize_motor_strength(env, None, asset_cfg, (0.8, 1.2))
    limits = robot.written["limits"]
    assert limits.shape == (4, 3)
    assert torch.all(limits >= 80.0) and torch.all(limits <= 120.0)
    torch.testing.assert_close(limits[:, 0], limits[:, 1])
    assert robot.written["env_ids"].tolist() == [0, 1, 2, 3]
