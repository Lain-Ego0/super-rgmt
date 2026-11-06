from types import SimpleNamespace

from extreme_rgmt.configuration import configure_paper_rewards


def test_configure_paper_rewards_matches_extreme_rgmt_table_i():
    names = (
        "motion_global_anchor_pos",
        "motion_global_anchor_ori",
        "motion_body_pos",
        "motion_body_ori",
        "motion_body_lin_vel",
        "motion_body_ang_vel",
        "action_rate_l2",
        "joint_limit",
        "undesired_contacts",
    )
    rewards = SimpleNamespace(**{name: SimpleNamespace(weight=123.0) for name in names})

    def term_factory(**kwargs):
        return SimpleNamespace(**kwargs)

    def scene_entity(name, body_names):
        return {"name": name, "body_names": body_names}

    def feet_slip(*args, **kwargs):
        raise AssertionError("function is stored, not invoked during configuration")

    configure_paper_rewards(
        rewards,
        reward_term_cls=term_factory,
        scene_entity_cfg_cls=scene_entity,
        feet_slip_func=feet_slip,
    )

    assert {name: getattr(getattr(rewards, name), "weight") for name in names} == {
        "motion_global_anchor_pos": 0.0,
        "motion_global_anchor_ori": 0.5,
        "motion_body_pos": 1.0,
        "motion_body_ori": 1.0,
        "motion_body_lin_vel": 1.0,
        "motion_body_ang_vel": 1.0,
        "action_rate_l2": -0.1,
        "joint_limit": -10.0,
        "undesired_contacts": -0.1,
    }
    assert rewards.feet_slip.weight == -0.1
    assert rewards.feet_slip.params["contact_force_threshold"] == 1.0
    assert rewards.feet_slip.params["sensor_cfg"]["name"] == "contact_forces"
    assert rewards.feet_slip.params["asset_cfg"]["name"] == "robot"
