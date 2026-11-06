"""Dependency-light configuration adapters for the Extreme-RGMT task."""
from __future__ import annotations

from typing import Any, Callable


def configure_paper_rewards(
    rewards_cfg: Any,
    *,
    reward_term_cls: Callable[..., Any],
    scene_entity_cfg_cls: Callable[..., Any],
    feet_slip_func: Callable[..., Any],
) -> None:
    """Apply Extreme-RGMT Table I weights to the RGMT reward configuration."""
    # PMT's RGMT config has an extra global-anchor-position tracking term.
    rewards_cfg.motion_global_anchor_pos.weight = 0.0
    rewards_cfg.motion_global_anchor_ori.weight = 0.5
    rewards_cfg.motion_body_pos.weight = 1.0
    rewards_cfg.motion_body_ori.weight = 1.0
    rewards_cfg.motion_body_lin_vel.weight = 1.0
    rewards_cfg.motion_body_ang_vel.weight = 1.0
    rewards_cfg.action_rate_l2.weight = -0.1
    rewards_cfg.joint_limit.weight = -10.0
    rewards_cfg.undesired_contacts.weight = -0.1
    rewards_cfg.feet_slip = reward_term_cls(
        func=feet_slip_func,
        weight=-0.1,
        params={
            "sensor_cfg": scene_entity_cfg_cls(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
            "asset_cfg": scene_entity_cfg_cls(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
            "contact_force_threshold": 1.0,
        },
    )
