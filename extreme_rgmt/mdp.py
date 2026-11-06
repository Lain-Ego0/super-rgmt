"""Extreme-RGMT-specific MDP terms layered over the pinned PMT task."""
from __future__ import annotations

import torch
from torch import Tensor


def feet_slip(env, sensor_cfg, asset_cfg, contact_force_threshold: float = 1.0) -> Tensor:
    """Per-environment sum of horizontal foot speed while the feet are in contact.

    Contact and body selections are supplied by Isaac Lab's ``SceneEntityCfg`` so
    body order is resolved by the active scene rather than assumed here.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    robot = env.scene[asset_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    contact = torch.linalg.vector_norm(forces, dim=-1) > contact_force_threshold
    velocity_xy = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    horizontal_speed = torch.linalg.vector_norm(velocity_xy, dim=-1)
    return torch.sum(horizontal_speed * contact.to(horizontal_speed.dtype), dim=-1)


def add_paper_proprio_noise(observation: Tensor) -> Tensor:
    """Add Table II uniform actor-side noise to PMT's 93D proprio vector.

    Layout follows ``pmt_tasks.mdp.proprio``: projected gravity (3), angular
    velocity (3), joint position (29), joint velocity (29), previous action (29).
    The critic uses the clean PMT term and must not call this helper.
    """
    if observation.shape[-1] != 93:
        raise ValueError(f"expected 93D proprioception, got {observation.shape[-1]}D")
    bounds = observation.new_zeros(93)
    bounds[:3] = 0.05
    bounds[3:6] = 0.2
    bounds[6:35] = 0.01
    bounds[35:64] = 0.5
    return observation + (torch.rand_like(observation) * 2.0 - 1.0) * bounds


def paper_noisy_proprio(env, asset_cfg=None) -> Tensor:
    """PMT proprio term with Extreme-RGMT's specified actor observation noise."""
    from pmt_tasks.mdp.observations import proprio

    observation = proprio(env) if asset_cfg is None else proprio(env, asset_cfg=asset_cfg)
    return add_paper_proprio_noise(observation)


def add_paper_command_noise(command: Tensor) -> Tensor:
    """Perturb reference velocity, gravity, and joint-pose channels per Table II.

    Command token layout is [linear velocity(3), angular velocity(3),
    gravity direction(3), joint position(29)] on its final axis.
    """
    if command.shape[-1] != 38:
        raise ValueError(f"expected 38D command tokens, got {command.shape[-1]}D")
    bounds = command.new_zeros(38)
    bounds[:3] = 0.5
    bounds[3:6] = 0.52
    bounds[6:9] = 0.05
    bounds[9:38] = 0.1
    return command + (torch.rand_like(command) * 2.0 - 1.0) * bounds


def paper_noisy_command_window(env, command_name: str, half_window: int = 10, stride: int = 1, flatten: bool = False) -> Tensor:
    """PMT reference window with Extreme-RGMT command perturbations."""
    from pmt_tasks.mdp.observations import command_window

    value = command_window(
        env,
        command_name=command_name,
        half_window=half_window,
        stride=stride,
        flatten=flatten,
    )
    return add_paper_command_noise(value)


def randomize_motor_strength(
    env,
    env_ids,
    asset_cfg,
    strength_scale_range: tuple[float, float] = (0.8, 1.2),
) -> None:
    """Randomize per-environment actuator torque limits at startup.

    This implements motor-strength scaling for the PMT/Isaac Lab implicit G1
    actuator through the articulation effort-limit writer. One scale is sampled
    per environment and shared across that robot's actuated joints.
    """
    low, high = strength_scale_range
    if not 0.0 < low <= high:
        raise ValueError(f"invalid motor strength scale range: {strength_scale_range}")
    asset = env.scene[asset_cfg.name]
    all_env_ids = torch.arange(env.num_envs, device=asset.device, dtype=torch.long)
    if env_ids is None or isinstance(env_ids, slice):
        resolved_env_ids = all_env_ids if env_ids is None else all_env_ids[env_ids]
    else:
        resolved_env_ids = torch.as_tensor(env_ids, device=asset.device, dtype=torch.long)

    joint_ids = asset_cfg.joint_ids
    base_limits = asset.data.joint_effort_limits.torch[resolved_env_ids]
    if not isinstance(joint_ids, slice):
        base_limits = base_limits[:, joint_ids]
    scale = torch.empty((len(resolved_env_ids), 1), device=asset.device, dtype=base_limits.dtype)
    scale.uniform_(low, high)
    asset.write_joint_effort_limit_to_sim_index(
        limits=base_limits * scale,
        joint_ids=None if isinstance(joint_ids, slice) else joint_ids,
        env_ids=resolved_env_ids,
    )
