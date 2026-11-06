"""Paper-aligned Isaac Lab event configuration for Extreme-RGMT Stage I.

The helper is dependency-injected so its event wiring can be unit-tested without
starting Isaac Sim/Kit. Runtime support remains backend-dependent and must be
verified with a simulator rollout.
"""
from __future__ import annotations

from typing import Any


def configure_paper_domain_randomization(
    env_cfg: Any,
    *,
    event_term_cls: Any,
    scene_entity_cfg_cls: Any,
    mdp: Any,
    motor_strength_func: Any,
) -> None:
    """Apply Table II randomization ranges to a PMT RGMT env config.

    Isaac Lab exposes gain, armature, and articulation effort-limit writers. The
    effort-limit event scales all actuated joints by one independent factor per env.
    """
    env_cfg.events.physics_material.params.update(
        static_friction_range=(0.10, 1.75),
        dynamic_friction_range=(0.10, 1.75),
    )

    # PMT already matches the paper for torso CoM, joint zero offset and push
    # interval; replace no-op/mismatching terms above while retaining those.
    env_cfg.events.add_base_mass = event_term_cls(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": scene_entity_cfg_cls("robot", body_names=["pelvis"]),
            "mass_distribution_params": (-3.0, 6.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    env_cfg.events.randomize_motor_strength = event_term_cls(
        func=motor_strength_func,
        mode="startup",
        params={
            "asset_cfg": scene_entity_cfg_cls("robot", joint_names=[".*"]),
            "strength_scale_range": (0.8, 1.2),
        },
    )
    env_cfg.events.randomize_pd_gains = event_term_cls(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": scene_entity_cfg_cls("robot", joint_names=[".*"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    env_cfg.events.randomize_armature = event_term_cls(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": scene_entity_cfg_cls("robot", joint_names=[".*"]),
            "armature_distribution_params": (1.0, 1.05),
            "operation": "scale",
        },
    )
