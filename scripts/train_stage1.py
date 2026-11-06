#!/usr/bin/env python3
"""Launch Extreme-RGMT Stage-I training on the pinned PMT RGMT task backend.

PMT is a pinned upstream checkout under third_party/PMT-pinned/. This adapter changes the policy
class and paper hyperparameters without editing upstream source. Run inside an
Isaac Lab Python environment with PMT installed/importable.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PMT_ROOT = Path(os.environ.get("PMT_ROOT", ROOT / "third_party" / "PMT-pinned")).expanduser().resolve()
if not (PMT_ROOT / "pmt_tasks" / "builder.py").is_file():
    raise SystemExit(
        f"PMT training backend not found at {PMT_ROOT}. See docs/extreme-rgmt-stage1.md "
        "for the pinned checkout command."
    )
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PMT_ROOT))

parser = argparse.ArgumentParser(description="Train Extreme-RGMT Stage-I generalist policy.")
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=10_000)
parser.add_argument("--physics-dt", type=float, default=None, help="Override physics substep; decimation is adjusted to retain the 20 ms policy/control period")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--motion-dir", type=Path, required=True)
parser.add_argument("--robot-urdf", type=Path, default=ROOT / "data/g1_description/g1_29dof_rev_1_0.urdf")
parser.add_argument("--run-name", default=None)
parser.add_argument("--residual-scale", choices=("unit", "pmt"), default="unit",
                    help="Scale policy residuals in radians (unit matches q_target=q_ref+a) or retain PMT per-joint scaling")
parser.add_argument("--initial-action-std", type=float, default=1.0,
                    help="Initial Gaussian policy standard deviation in normalized action units")
parser.add_argument("--eval-checkpoint", type=Path, default=None,
                    help="Evaluate a saved checkpoint deterministically instead of training")
parser.add_argument("--eval-zero-action", action="store_true",
                    help="Evaluate the reference-pose PD baseline with zero policy residuals")
parser.add_argument("--eval-steps", type=int, default=2000, help="Number of deterministic evaluation steps")
parser.add_argument("--eval-frame", type=int, default=0, help="Fixed reference frame used for every evaluation reset")
parser.add_argument("--eval-output", type=Path, default=None, help="Write complete evaluation metrics as JSON")
parser.add_argument("--eval-paper-split", action="store_true", help="Evaluate exactly five seeded rollouts per motion clip and write the mastered/challenging partition")
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--diagnose-reset", action="store_true", help="Print one-step reference-vs-robot reset errors and exit")
parser.add_argument("--diagnose-reference-steps", type=int, default=0, help="Track the unperturbed reference with zero residuals for N steps, then exit")
parser.add_argument("--diagnose-hold-reference-steps", type=int, default=0, help="Hold one fixed reference pose under PhysX PD control for N control steps, then exit; requires --eval-zero-action")
if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    parser.print_help()
    raise SystemExit(0)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.motion_dir.exists():
    raise SystemExit(f"motion directory does not exist: {args.motion_dir}")
if not args.robot_urdf.is_file():
    raise SystemExit(f"retarget-matched G1 URDF not found: {args.robot_urdf}; pass --robot-urdf explicitly")
motion_files = sorted(args.motion_dir.rglob("*.npz")) if args.motion_dir.is_dir() else [args.motion_dir]
from scripts.convert_lafan1_to_pmt import ISAAC_JOINTS
import numpy as np
for motion_path in motion_files:
    with np.load(motion_path, allow_pickle=False) as motion_data:
        if "joint_names" not in motion_data.files:
            raise SystemExit(f"{motion_path}: missing joint_names metadata; reconvert using scripts/convert_lafan1_to_pmt.py")
        saved_joint_names = motion_data["joint_names"].astype(str).tolist()
        if "quat_order" not in motion_data.files or str(motion_data["quat_order"].item()) != "xyzw":
            raise SystemExit(f"{motion_path}: body_quat_w must declare quat_order=xyzw; reconvert motions")
    if saved_joint_names != ISAAC_JOINTS:
        raise SystemExit(f"{motion_path}: joint_names do not match the pinned Isaac G1 articulation order")
if not motion_files or any(path.suffix.lower() != ".npz" for path in motion_files):
    raise SystemExit(
        f"no PMT-compatible .npz motion files found under {args.motion_dir}; "
        "convert and validate G1 motion clips before launching Isaac Sim"
    )

# Isaac Lab app must launch before importing task/env modules.
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from motion_tracking_rl import registry  # noqa: E402
from pmt_tasks.registry_gym import register_pmt_tasks  # noqa: E402
from pmt_tasks.robots.g1 import G1_ACTION_SCALE  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm, RewardTermCfg as RewTerm, SceneEntityCfg  # noqa: E402
from isaaclab.envs import mdp as isaac_mdp  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from pmt_tasks.mdp.commands.multi_motion_command import (  # noqa: E402
    MultiMotionCommandV2, MultiMotionCommandV2Cfg, SamplingResult,
)
from pmt_tasks.mdp.observations import command_window as clean_command_window, proprio as clean_proprio  # noqa: E402
from extreme_rgmt.isaac_actions import ReferenceResidualJointPositionAction  # noqa: E402
from extreme_rgmt.mdp import (  # noqa: E402
    feet_slip, paper_noisy_command_window, paper_noisy_proprio, randomize_motor_strength,
)
from extreme_rgmt.configuration import configure_paper_rewards  # noqa: E402
from extreme_rgmt.domain_randomization import configure_paper_domain_randomization  # noqa: E402
from extreme_rgmt.models import register_with_motion_tracking_rl  # noqa: E402
from extreme_rgmt.evaluation import (  # noqa: E402
    build_motion_partition, configure_paper_evaluation_terminations, fixed_motion_schedule,
    randomized_rollout_tasks, validate_fixed_evaluation_env,
)

class FixedMotionEvaluationCommand(MultiMotionCommandV2):
    """Use a fixed motion/frame schedule while retaining PMT's normal reset flow."""

    def _draw_samples(self, env_ids):
        motion_schedule = torch.as_tensor(self.cfg.eval_motion_ids, device=self.device, dtype=torch.long)
        frame_schedule = torch.as_tensor(self.cfg.eval_frame_ids, device=self.device, dtype=torch.long)
        if motion_schedule.numel() != self.num_envs or frame_schedule.numel() != self.num_envs:
            raise RuntimeError("fixed evaluation schedule must have one entry per environment")
        selected_motion = motion_schedule[env_ids]
        selected_frame = frame_schedule[env_ids]
        if torch.any(selected_motion < 0) or torch.any(selected_motion >= self.data_store.num_motions):
            raise RuntimeError("fixed evaluation schedule contains invalid motion IDs")
        lengths = self.data_store.motion_lengths[selected_motion]
        if torch.any(selected_frame < 0) or torch.any(selected_frame >= lengths):
            raise RuntimeError("fixed evaluation schedule contains a frame outside its motion")
        return SamplingResult(motion_ids=selected_motion, frame_ids=selected_frame)


@configclass
class FixedMotionEvaluationCommandCfg(MultiMotionCommandV2Cfg):
    class_type: type = FixedMotionEvaluationCommand
    eval_motion_ids: list[int] = []
    eval_frame_ids: list[int] = []


registry.autoload()
register_with_motion_tracking_rl()
register_pmt_tasks()

task_id = "RGMT-G1-v0"
env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")

if args.eval_paper_split:
    if (args.eval_checkpoint is None) == (not args.eval_zero_action):
        raise SystemExit("--eval-paper-split requires exactly one of --eval-checkpoint or --eval-zero-action")
    if args.num_envs < 1:
        raise SystemExit("--num-envs must be positive and acts as the simulator-memory cap")
    if len(motion_files) < 1:
        raise SystemExit("--eval-paper-split requires full motion files")
    motion_lengths = []
    motion_fps = []
    for path in motion_files:
        with np.load(path, allow_pickle=False) as data:
            motion_lengths.append(int(data["joint_pos"].shape[0]))
            motion_fps.append(float(np.asarray(data["fps"]).item()))
    if any(abs(fps - 50.0) > 1e-6 for fps in motion_fps):
        raise SystemExit("--eval-paper-split requires all motions resampled to the paper's 50 Hz control rate")
    paper_names = [path.stem for path in motion_files]
    if len(set(paper_names)) != len(paper_names):
        raise SystemExit("--eval-paper-split requires unique motion file stems")
    paper_rollout_tasks = randomized_rollout_tasks(motion_lengths, rollouts_per_motion=5, seed=args.seed)
    eval_num_envs = min(args.num_envs, len(paper_rollout_tasks))
else:
    paper_rollout_tasks = None
    eval_num_envs = args.num_envs

# Retain PMT's tested G1 flat tracking environment, adaptive multi-motion command,
# reward terms and simulator wrapper; use Extreme-RGMT's actor-critic and PPO values.
env_cfg.scene.num_envs = eval_num_envs
env_cfg.seed = args.seed
if args.device is not None:
    env_cfg.sim.device = args.device
env_cfg.pmt_motion_paths = [str(path.resolve()) for path in motion_files]
if args.eval_paper_split:
    # Full-motion success requires the episode timeout to exceed the longest
    # reference duration; the root-height criterion remains the failure gate.
    env_cfg.episode_length_s = max(motion_lengths) / 50.0 + 1.0
env_cfg.__post_init__()
if args.physics_dt is not None:
    if args.physics_dt <= 0.0:
        raise SystemExit("--physics-dt must be positive")
    decimation = round(0.02 / args.physics_dt)
    if decimation < 1 or abs(decimation * args.physics_dt - 0.02) > 1e-8:
        raise SystemExit("--physics-dt must divide the 20 ms control period exactly (for example 0.005 or 0.002)")
    env_cfg.sim.dt = float(args.physics_dt)
    env_cfg.decimation = int(decimation)
    env_cfg.sim.render_interval = int(decimation)
    print(f"STAGE1_PHYSICS_CONFIG physics_dt={env_cfg.sim.dt} decimation={env_cfg.decimation} control_dt={env_cfg.sim.dt * env_cfg.decimation}", flush=True)
env_cfg.scene.robot.spawn.asset_path = str(args.robot_urdf.resolve())
# The paper defines q_target = q_ref + a, so use unit residual scale by default.
# Keep PMT scaling as an explicit ablation option.
env_cfg.actions.joint_pos.scale = 1.0 if args.residual_scale == "unit" else G1_ACTION_SCALE
env_cfg.actions.joint_pos.use_default_offset = True
env_cfg.actions.joint_pos.class_type = ReferenceResidualJointPositionAction
env_cfg.commands.motion.update_action_offset_with_ref = False
configure_paper_domain_randomization(
    env_cfg, event_term_cls=EventTerm, scene_entity_cfg_cls=SceneEntityCfg,
    mdp=isaac_mdp, motor_strength_func=randomize_motor_strength,
)
for group_name in ("policy", "proprio", "proprio_history"):
    group = getattr(env_cfg.observations, group_name)
    group.proprio.func = paper_noisy_proprio
env_cfg.observations.command_window.command_window.func = paper_noisy_command_window
configure_paper_rewards(
    env_cfg.rewards,
    reward_term_cls=RewTerm,
    scene_entity_cfg_cls=SceneEntityCfg,
    feet_slip_func=feet_slip,
)
if args.eval_checkpoint is not None or args.eval_zero_action:
    import torch
    if args.eval_frame < 0:
        raise SystemExit("--eval-frame must be nonnegative")
    if args.eval_checkpoint is not None and not args.eval_checkpoint.is_file():
        raise SystemExit(f"evaluation checkpoint not found: {args.eval_checkpoint}")
    if args.eval_steps < 1:
        raise SystemExit("--eval-steps must be positive")
    # Paper evaluation counts failure only when root-height error exceeds 0.2 m.
    # PMT's extra anchor-orientation and end-effector termination rules are
    # disabled for evaluation; training keeps its task configuration.
    configure_paper_evaluation_terminations(env_cfg.terminations, height_threshold=0.2)
    # Deterministic validation: remove training randomization and sensor corruption.
    for event_name in ("physics_material", "add_joint_default_pos", "base_com", "add_base_mass",
                       "randomize_motor_strength", "randomize_pd_gains", "randomize_armature", "push_robot"):
        if hasattr(env_cfg.events, event_name):
            setattr(env_cfg.events, event_name, None)
    env_cfg.commands.motion.pose_range = {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")}
    env_cfg.commands.motion.velocity_range = {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")}
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    if args.eval_frame < 0:
        raise SystemExit("--eval-frame must be nonnegative")
    if args.eval_paper_split:
        initial_tasks = paper_rollout_tasks[:eval_num_envs]
        fixed_motion_ids = torch.tensor([task[0] for task in initial_tasks], dtype=torch.long)
        fixed_frame_ids = torch.tensor([task[1] for task in initial_tasks], dtype=torch.long)
    else:
        fixed_motion_ids, fixed_frame_ids = fixed_motion_schedule(
            eval_num_envs, len(motion_files), frame_id=args.eval_frame
        )
    eval_command_cfg = FixedMotionEvaluationCommandCfg()
    for config_name, config_value in vars(env_cfg.commands.motion).items():
        setattr(eval_command_cfg, config_name, config_value)
    eval_command_cfg.class_type = FixedMotionEvaluationCommand
    # PMT chunks long training sequences internally; deterministic evaluation needs
    # one stable motion ID per manifest entry for auditable per-motion metrics.
    eval_command_cfg.chunk_length = 0
    eval_command_cfg.eval_motion_ids = fixed_motion_ids.tolist()
    eval_command_cfg.eval_frame_ids = fixed_frame_ids.tolist()
    if args.eval_paper_split:
        # Preserve the explicit randomized trial frames exactly; PMT's regular
        # evaluation defaults can otherwise force frame zero or alter reset poses.
        eval_command_cfg.force_start_frame_zero = False
        eval_command_cfg.default_reset_joint_prob = 0.0
    env_cfg.commands.motion = eval_command_cfg
    for group_name in ("policy", "proprio", "proprio_history"):
        group = getattr(env_cfg.observations, group_name, None)
        if group is not None:
            group.enable_corruption = False
            if hasattr(group, "proprio"):
                group.proprio.func = clean_proprio
    env_cfg.observations.command_window.enable_corruption = False
    env_cfg.observations.command_window.command_window.func = clean_command_window
agent_cfg.seed = args.seed
agent_cfg.max_iterations = args.iterations
agent_cfg.experiment_name = "extreme_rgmt_stage1"
agent_cfg.run_name = args.run_name
agent_cfg.num_steps_per_env = 24
agent_cfg.save_interval = 100
agent_cfg.policy.class_name = "ExtremeRGMTActorCritic"
agent_cfg.policy.init_noise_std = args.initial_action_std
agent_cfg.policy.actor_hidden_dims = [1024, 1024, 512, 256]
agent_cfg.policy.critic_hidden_dims = [1024, 1024, 512, 512]
agent_cfg.policy.n_embd = 64
agent_cfg.policy.n_heads = 4
agent_cfg.policy.history_len = 10
agent_cfg.policy.cmd_len = 21
agent_cfg.policy.mlp_ratio = 4
# The policy implements the paper's separate LayerNorm feature branches;
# disable PMT's additional empirical normalizers, which are not specified here.
agent_cfg.policy.actor_obs_normalization = False
agent_cfg.policy.critic_obs_normalization = False
agent_cfg.policy.history_obs_normalization = False
agent_cfg.policy.command_obs_normalization = False
agent_cfg.algorithm.learning_rate = 1.0e-3
agent_cfg.algorithm.schedule = "adaptive"
agent_cfg.algorithm.desired_kl = 0.01
agent_cfg.algorithm.gamma = 0.99
agent_cfg.algorithm.lam = 0.95
agent_cfg.algorithm.clip_param = 0.2
agent_cfg.algorithm.entropy_coef = 0.005
agent_cfg.algorithm.num_learning_epochs = 5
agent_cfg.algorithm.num_mini_batches = 4
agent_cfg.device = args.device or "cuda:0"
if args.diagnose_reference_steps or args.diagnose_hold_reference_steps:
    if args.diagnose_reference_steps < 0 or args.diagnose_hold_reference_steps < 0:
        raise SystemExit("diagnostic step counts must be nonnegative")
    if args.diagnose_reference_steps and args.diagnose_hold_reference_steps:
        raise SystemExit("choose either --diagnose-reference-steps or --diagnose-hold-reference-steps")
    if args.diagnose_reference_steps < 1 and args.diagnose_hold_reference_steps < 1:
        raise SystemExit("--diagnose-reference-steps must be positive")
    env_cfg.commands.motion.pose_range = {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")}
    env_cfg.commands.motion.velocity_range = {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")}
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    for name in ("physics_material", "add_base_mass", "randomize_motor_strength", "randomize_pd_gains", "randomize_armature", "push_robot"):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)
    for name in ("anchor_pos", "anchor_ori", "ee_body_pos"):
        term = getattr(env_cfg.terminations, name, None)
        if term is not None and term.params and "threshold" in term.params:
            term.params["threshold"] = 10.0

env = gym.make(task_id, cfg=env_cfg, render_mode=None)
from pmt_tasks.isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

env = RslRlVecEnvWrapper(env)
if args.eval_checkpoint is not None or args.eval_zero_action:
    eval_base_env = validate_fixed_evaluation_env(
        env,
        expected_motion_ids=fixed_motion_ids,
        expected_frame_ids=fixed_frame_ids,
        expected_motion_files=[path.name for path in motion_files],
        clean_proprio_fn=clean_proprio,
        clean_command_fn=clean_command_window,
    )
    print(
        f"STAGE1_EVAL_RESET_OK motion_ids={eval_base_env.command_manager.get_term('motion').motion_ids.tolist()} "
        f"frame_ids={eval_base_env.command_manager.get_term('motion').frame_ids.tolist()} "
        f"active_events={eval_base_env.event_manager.active_terms}",
        flush=True,
    )
if args.diagnose_hold_reference_steps:
    if not args.eval_zero_action:
        raise SystemExit("--diagnose-hold-reference-steps requires --eval-zero-action for a fixed, clean reference start")
    import torch
    base_env = env.unwrapped
    base_env.reset()
    command = base_env.command_manager.get_term("motion")
    robot = base_env.scene["robot"]
    action_term = base_env.action_manager.get_term("joint_pos")
    held_reference = command.joint_pos.clone()
    zero_action = torch.zeros((args.num_envs, 29), device=agent_cfg.device)
    base_env.action_manager.process_action(zero_action)
    if (action_term.processed_actions - held_reference).abs().max().item() > 1e-5:
        raise RuntimeError("fixed-reference PD target does not match the reset motion pose")
    start_root = robot.data.root_link_pos_w.clone()
    contact_sensor = base_env.scene.sensors.get("contact_forces")
    foot_sensor_ids = None
    if contact_sensor is not None:
        foot_sensor_ids, _ = contact_sensor.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)
    print(f"HOLD_DIAG motion_ids={command.motion_ids.tolist()} frame_ids={command.frame_ids.tolist()} steps={args.diagnose_hold_reference_steps}", flush=True)
    print(f"HOLD_DIAG initial_stiffness={robot.data.joint_stiffness[0].detach().cpu().tolist()}", flush=True)
    import warp as wp
    material_props = wp.to_torch(robot.root_view.get_material_properties()).detach().cpu()
    print(
        f"HOLD_DIAG robot_material_shape={tuple(material_props.shape)} "
        f"static_friction_range=({material_props[..., 0].min().item():.4f},{material_props[..., 0].max().item():.4f}) "
        f"dynamic_friction_range=({material_props[..., 1].min().item():.4f},{material_props[..., 1].max().item():.4f}) "
        f"restitution_range=({material_props[..., 2].min().item():.4f},{material_props[..., 2].max().item():.4f}) "
        f"ground_material_cfg={base_env.scene.terrain.cfg.physics_material}", flush=True,
    )
    for step in range(args.diagnose_hold_reference_steps):
        for _ in range(base_env.cfg.decimation):
            base_env.action_manager.apply_action()
            base_env.scene.write_data_to_sim()
            base_env.sim.step(render=False)
            base_env.scene.update(dt=base_env.physics_dt)
        joint_error = held_reference - robot.data.joint_pos
        if step == 0 or (step + 1) % 5 == 0 or step + 1 == args.diagnose_hold_reference_steps:
            foot_force_max = None
            if contact_sensor is not None and foot_sensor_ids:
                foot_force_max = contact_sensor.data.net_forces_w[:, foot_sensor_ids].norm(dim=-1).amax(dim=-1).mean().item()
            print(
                f"HOLD_DIAG step={step + 1} ref_hold_error_max={joint_error.abs().max().item():.5f} "
                f"joint_error_l2_mean={torch.linalg.vector_norm(joint_error, dim=-1).mean().item():.5f} "
                f"root_drift_m={torch.linalg.vector_norm(robot.data.root_link_pos_w - start_root, dim=-1).mean().item():.5f} "
                f"root_z={robot.data.root_link_pos_w[:, 2].mean().item():.5f} "
                f"foot_contact_force_max={foot_force_max}", flush=True,
            )
    env.close()
    app.close()
    raise SystemExit(0)
if args.diagnose_reference_steps:
    import torch
    base_env = env.unwrapped
    env.reset()
    command = base_env.command_manager.get_term("motion")
    zero_action = torch.zeros((args.num_envs, 29), device=agent_cfg.device)
    action_term = base_env.action_manager.get_term("joint_pos")
    robot = base_env.scene["robot"]
    print(f"REFERENCE_DIAG action_term={type(action_term).__name__} scale={args.residual_scale}", flush=True)
    print(f"REFERENCE_DIAG runtime_stiffness={robot.data.joint_stiffness[0].detach().cpu().tolist()}", flush=True)
    print(f"REFERENCE_DIAG runtime_damping={robot.data.joint_damping[0].detach().cpu().tolist()}", flush=True)
    print(f"REFERENCE_DIAG runtime_effort_limits={robot.data.joint_effort_limits[0].detach().cpu().tolist()}", flush=True)
    body_names = command.cfg.body_names
    selected = [body_names.index(name) for name in ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link")]
    selected = torch.tensor(selected, device=command.device)
    for step in range(args.diagnose_reference_steps):
        command_frame_before = getattr(command, "frame_ids", None)
        command_frame_before = command_frame_before.clone() if command_frame_before is not None else None
        reference_joint_before = command.joint_pos.clone()
        _, reward, done, extras = env.step(zero_action)
        action_target = action_term.processed_actions
        target_reference_error = (action_target - reference_joint_before).abs().amax(dim=-1)
        frame_ids_after = getattr(command, "frame_ids", None)
        if command_frame_before is not None and frame_ids_after is not None:
            frame_delta = frame_ids_after - command_frame_before
            if not torch.all(frame_delta == 1):
                raise RuntimeError(
                    f"motion reference must advance exactly one frame per 50 Hz control step; got {frame_delta.tolist()}"
                )
        else:
            frame_delta = None
        if torch.any(target_reference_error > 1e-5):
            raise RuntimeError(
                f"zero residual action target diverged from the pre-step reference: {target_reference_error.tolist()}"
            )
        errors = command.body_pos_relative_w[:, selected] - command.robot_body_pos_w[:, selected]
        joint_error = command.joint_pos - command.robot_joint_pos
        if step < 5:
            frame_ids = getattr(command, "time_steps", getattr(command, "frame_ids", None))
            actuator_diagnostics = {}
            for name, actuator in robot.actuators.items():
                computed = getattr(actuator, "computed_effort", None)
                applied = getattr(actuator, "applied_effort", None)
                limits = getattr(actuator, "actuator_effort_limit", None)
                if computed is not None:
                    effort = computed[0].abs()
                    applied_effort = applied[0].abs() if applied is not None else effort
                    limit = limits[0] if limits is not None and limits.ndim > 1 else limits
                    saturation = (
                        float((effort >= limit * 0.999).float().mean().item())
                        if limit is not None and torch.isfinite(limit).all()
                        else None
                    )
                    actuator_diagnostics[name] = {
                        "computed_effort_max": float(effort.max().item()),
                        "applied_effort_max": float(applied_effort.max().item()),
                        "limit_min": float(limit.min().item()) if limit is not None else None,
                        "saturated_fraction": saturation,
                    }
            print(
                f"REFERENCE_DIAG actuator_step={step + 1} applied_frame={command_frame_before[:min(4, args.num_envs)].detach().cpu().tolist() if command_frame_before is not None else None} "
                f"next_observation_frame={frame_ids[:min(4, args.num_envs)].detach().cpu().tolist() if frame_ids is not None else None} "
                f"frame_delta={frame_delta[:min(4, args.num_envs)].detach().cpu().tolist() if frame_delta is not None else None} "
                f"zero_action_target_ref_error={target_reference_error[:min(4, args.num_envs)].detach().cpu().tolist()} "
                f"ref_joint_speed_max={command.joint_vel[0].abs().max().item():.4f} "
                f"ref_body_linear_speed_max={command.body_lin_vel_w[0].norm(dim=-1).max().item():.4f} "
                f"ref_body_angular_speed_max={command.body_ang_vel_w[0].norm(dim=-1).max().item():.4f} "
                f"joint_error_max={joint_error[0].abs().max().item():.4f} "
                f"actuators={actuator_diagnostics}", flush=True,
            )
        if step == 0:
            print(f"REFERENCE_DIAG motion_ids={command.motion_ids.tolist()} frame_ids={command.frame_ids.tolist()}", flush=True)
            print(f"REFERENCE_DIAG selected_body_names={[body_names[i] for i in selected.tolist()]}", flush=True)
            print(f"REFERENCE_DIAG body_xyz_error={errors[0].detach().cpu().tolist()}", flush=True)
            print(f"REFERENCE_DIAG reference_xyz={command.body_pos_relative_w[0, selected].detach().cpu().tolist()}", flush=True)
            print(f"REFERENCE_DIAG robot_xyz={command.robot_body_pos_w[0, selected].detach().cpu().tolist()}", flush=True)
            print(f"REFERENCE_DIAG anchors={command.anchor_pos_w[0].detach().cpu().tolist()},{command.robot_anchor_pos_w[0].detach().cpu().tolist()}", flush=True)
        print(
            f"REFERENCE_DIAG step={step + 1} done={int(done.sum().item())} "
            f"reward_mean={reward.mean().item():.6f} body_error_max={torch.linalg.vector_norm(errors, dim=-1).max().item():.4f} "
            f"ee_z_error_max={errors[..., 2].abs().max().item():.4f} "
            f"joint_error_l2_mean={torch.linalg.vector_norm(joint_error, dim=-1).mean().item():.4f}",
            flush=True,
        )
    env.close()
    app.close()
    raise SystemExit(0)
if args.diagnose_reset:
    import torch
    base_env = env.unwrapped
    env.reset()
    zero_action = torch.zeros((args.num_envs, 29), device=agent_cfg.device)
    command_before = base_env.command_manager.get_term("motion")
    reference_before = command_before.joint_pos.clone()
    _, _, done, extras = env.step(zero_action)
    action_term = base_env.action_manager.get_term("joint_pos")
    processed_target = action_term.processed_actions
    equation_error = (processed_target - reference_before).abs().max().item()
    print(f"RESET_DIAG action_equation_max_abs_error={equation_error:.8g} action_term={type(action_term).__name__}", flush=True)
    if equation_error > 1e-5:
        raise RuntimeError("action term violated q_target = q_ref + a for zero residual")
    terminated = done
    time_out = torch.zeros_like(done)
    command = base_env.command_manager.get_term("motion")
    body_names = command.cfg.body_names
    selected_names = ["left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    indices = torch.tensor([body_names.index(name) for name in selected_names], device=command.device)
    body_error = command.body_pos_relative_w[:, indices] - command.robot_body_pos_w[:, indices]
    joint_error = command.joint_pos - command.robot_joint_pos
    n = min(8, args.num_envs)
    print(f"RESET_DIAG robot_urdf={args.robot_urdf.resolve()}", flush=True)
    print(f"RESET_DIAG bodies={selected_names}", flush=True)
    robot = base_env.scene["robot"]
    print(f"RESET_DIAG cfg_bodies={body_names}", flush=True)
    print(f"RESET_DIAG robot_bodies={robot.body_names}", flush=True)
    print(f"RESET_DIAG body_indexes={command.body_indices.detach().cpu().tolist()}", flush=True)
    runtime_joint_order = ISAAC_JOINTS
    runtime_joint_names = list(robot.joint_names)
    print(f"RESET_DIAG runtime_joint_names={runtime_joint_names}", flush=True)
    order_matches = runtime_joint_names == runtime_joint_order
    print(f"RESET_DIAG joint_order_matches_converter={order_matches}", flush=True)
    if not order_matches:
        raise RuntimeError("Isaac runtime joint order differs from the converter target order")
    print(f"RESET_DIAG step_extras={extras}", flush=True)
    print(f"RESET_DIAG motion_ids={command.motion_ids[:n].detach().cpu().tolist()}", flush=True)
    frame_ids = getattr(command, "time_steps", getattr(command, "frame_ids", None))
    if frame_ids is not None:
        print(f"RESET_DIAG frame_ids={frame_ids[:n].detach().cpu().tolist()}", flush=True)
    print(f"RESET_DIAG terminated={terminated[:n].detach().cpu().tolist()} timeout={time_out[:n].detach().cpu().tolist()}", flush=True)
    print(f"RESET_DIAG body_z_error={body_error[:n, :, 2].detach().cpu().tolist()}", flush=True)
    print(f"RESET_DIAG body_xyz_norm={torch.linalg.vector_norm(body_error[:n], dim=-1).detach().cpu().tolist()}", flush=True)
    print(f"RESET_DIAG ref_body_z={command.body_pos_relative_w[0, indices, 2].detach().cpu().tolist()}", flush=True)
    print(f"RESET_DIAG robot_body_z={command.robot_body_pos_w[0, indices, 2].detach().cpu().tolist()}", flush=True)
    print(f"RESET_DIAG anchor_ref_robot_z={command.anchor_pos_w[0, 2].item():.4f},{command.robot_anchor_pos_w[0, 2].item():.4f}", flush=True)
    print(f"RESET_DIAG root_link_z={base_env.scene["robot"].data.root_link_pos_w[0, 2].item():.4f}", flush=True)
    print(f"RESET_DIAG joint_error_max={joint_error[:n].abs().max(dim=-1).values.detach().cpu().tolist()}", flush=True)
    env.close()
    app.close()
    raise SystemExit(0)
runner_class = registry.get_runner(agent_cfg.class_name)
run_dir = ROOT / "logs" / "extreme_rgmt_stage1" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
runner_cfg = agent_cfg.to_dict()
# PMT's Transformer config carries optional auxiliary estimators that this policy
# does not implement; keep the runner/algorithm settings and strip only those keys.
for pmt_auxiliary_key in (
    "anchor_estimator_detach",
    "anchor_estimator_hidden_dims",
    "anchor_estimator_latent_inputs",
    "anchor_estimator_output_dim",
    "anchor_gt_normalization",
    "forward_hidden_dims",
    "forward_obs_normalization",
    "use_anchor_estimator",
    "use_vel_estimator",
    "vel_estimator_detach",
    "vel_estimator_hidden_dims",
    "vel_estimator_output_dim",
    "vel_gt_normalization",
):
    runner_cfg["policy"].pop(pmt_auxiliary_key, None)
runner = runner_class(env, runner_cfg, log_dir=str(run_dir), device=agent_cfg.device)
runner.add_git_repo_to_log(__file__)
if args.eval_checkpoint is None and not args.eval_zero_action:
    print(f"STAGE1_CONFIG residual_scale={args.residual_scale} initial_action_std={args.initial_action_std}", flush=True)
else:
    print(f"STAGE1_EVAL_CONFIG residual_scale={args.residual_scale} observation_noise=disabled domain_randomization=disabled", flush=True)
training_error = None
try:
    if args.eval_checkpoint is not None or args.eval_zero_action:
        policy_fn = None
        if args.eval_checkpoint is not None:
            runner.load(str(args.eval_checkpoint), map_location=agent_cfg.device)
            policy_fn = runner.get_inference_policy(device=agent_cfg.device)
        obs = env.get_observations()
        episode_returns = torch.zeros(eval_num_envs, device=agent_cfg.device)
        episode_lengths = torch.zeros(eval_num_envs, device=agent_cfg.device)
        finished_returns, finished_lengths = [], []
        action_abs_sum = 0.0
        joint_error_sum = 0.0
        action_max_abs = 0.0
        base_env = validate_fixed_evaluation_env(
            env, expected_motion_ids=fixed_motion_ids, expected_frame_ids=fixed_frame_ids,
            expected_motion_files=[path.name for path in motion_files],
            clean_proprio_fn=clean_proprio, clean_command_fn=clean_command_window,
        )
        motion_names = [Path(path).stem for path in motion_files]
        per_motion = {name: {"failed_rollouts": 0, "completed_rollouts": 0, "failed_rollout_length_sum": 0,
                            "completed_rollout_length_sum": 0, "terminations": {}, "rollout_trials": []}
                      for name in sorted(set(motion_names))}
        term_counts = {name: 0 for name in base_env.termination_manager.active_terms}
        successful_episodes = 0
        motion_elapsed = torch.zeros(eval_num_envs, device=agent_cfg.device, dtype=torch.long)
        paper_eval = bool(args.eval_paper_split)
        task_cursor = eval_num_envs if paper_eval else 0
        completed_trials = 0
        active_trial = torch.ones(eval_num_envs, dtype=torch.bool, device=agent_cfg.device)
        if paper_eval:
            task_limit = len(paper_rollout_tasks)
            max_eval_steps = task_limit * max(motion_lengths) + 1
        else:
            task_limit = 0
            max_eval_steps = args.eval_steps
        with torch.inference_mode():
            for eval_step_index in range(max_eval_steps):
                if paper_eval and completed_trials >= task_limit:
                    break
                action = (torch.zeros((eval_num_envs, 29), device=agent_cfg.device)
                          if policy_fn is None else policy_fn(obs))
                motion_command = base_env.command_manager.get_term("motion")
                episode_motion_ids = motion_command.motion_ids.clone()
                episode_start_frames = motion_command.start_frame_ids.clone()
                previous_frame_ids = motion_command.frame_ids.clone()
                current_lengths = motion_command.data_store.motion_lengths[episode_motion_ids]
                action_abs_sum += float(action.abs().mean().item())
                action_max_abs = max(action_max_abs, float(action.abs().max().item()))
                obs, reward, done, extras = env.step(action)
                motion_command = base_env.command_manager.get_term("motion")
                joint_error_sum += float(torch.linalg.vector_norm(
                    motion_command.joint_pos - motion_command.robot_joint_pos, dim=-1).mean().item())
                for name in term_counts:
                    term_mask = base_env.termination_manager.get_term(name).bool()
                    if paper_eval:
                        term_mask = term_mask & active_trial
                    term_counts[name] += int(term_mask.sum().item())
                episode_returns += reward
                episode_lengths += 1
                motion_elapsed += 1
                completed_mask = (previous_frame_ids + 1 >= current_lengths) & (~done)
                if paper_eval:
                    completed_mask &= active_trial
                    done = done.bool() & active_trial
                for env_id in torch.where(completed_mask)[0].tolist():
                    motion_name = motion_names[int(episode_motion_ids[env_id].item())]
                    metrics = per_motion[motion_name]
                    metrics["completed_rollouts"] += 1
                    metrics["completed_rollout_length_sum"] += int(motion_elapsed[env_id].item())
                    if paper_eval:
                        metrics["rollout_trials"].append({
                            "start_frame": int(episode_start_frames[env_id].item()),
                            "success": True, "steps": int(motion_elapsed[env_id].item()),
                        })
                    successful_episodes += 1
                    finished_returns.append(float(episode_returns[env_id].item()))
                    finished_lengths.append(int(episode_lengths[env_id].item()))
                    episode_returns[env_id] = 0
                    episode_lengths[env_id] = 0
                    motion_elapsed[env_id] = 0
                    if paper_eval:
                        completed_trials += 1
                for env_id in torch.where(done)[0].tolist():
                    finished_returns.append(float(episode_returns[env_id].item()))
                    finished_lengths.append(int(episode_lengths[env_id].item()))
                    motion_name = motion_names[int(episode_motion_ids[env_id].item())]
                    metrics = per_motion[motion_name]
                    metrics["failed_rollouts"] += 1
                    metrics["failed_rollout_length_sum"] += int(motion_elapsed[env_id].item())
                    if paper_eval:
                        metrics["rollout_trials"].append({
                            "start_frame": int(episode_start_frames[env_id].item()),
                            "success": False, "steps": int(motion_elapsed[env_id].item()),
                        })
                    for term_name in term_counts:
                        if bool(base_env.termination_manager.get_term(term_name)[env_id].item()):
                            term_dict = metrics["terminations"]
                            term_dict[term_name] = term_dict.get(term_name, 0) + 1
                    episode_returns[env_id] = 0
                    episode_lengths[env_id] = 0
                    motion_elapsed[env_id] = 0
                    if paper_eval:
                        completed_trials += 1
                if paper_eval:
                    outcome_env_ids = sorted(set(torch.where(completed_mask)[0].tolist()) | set(torch.where(done)[0].tolist()))
                    if outcome_env_ids:
                        next_env_ids = []
                        for env_id in outcome_env_ids:
                            active_trial[env_id] = False
                            if task_cursor < task_limit:
                                motion_id, frame_id = paper_rollout_tasks[task_cursor]
                                motion_command.cfg.eval_motion_ids[env_id] = int(motion_id)
                                motion_command.cfg.eval_frame_ids[env_id] = int(frame_id)
                                task_cursor += 1
                                active_trial[env_id] = True
                                next_env_ids.append(env_id)
                        if next_env_ids:
                            reset_ids = torch.as_tensor(next_env_ids, device=base_env.device, dtype=torch.long)
                            base_env._reset_idx(reset_ids)
                            episode_returns[reset_ids] = 0
                            episode_lengths[reset_ids] = 0
                            motion_elapsed[reset_ids] = 0
                            obs = env.get_observations()
            if paper_eval and completed_trials != task_limit:
                raise RuntimeError(
                    f"paper evaluation completed {completed_trials}/{task_limit} rollouts within {max_eval_steps} steps"
                )
        for metrics in per_motion.values():
            total_rollouts = metrics["completed_rollouts"] + metrics["failed_rollouts"]
            metrics["success_rate"] = metrics["completed_rollouts"] / total_rollouts if total_rollouts else None
        steps_run = eval_step_index + 1
        if paper_eval:
            success_counts = {
                name: (metrics["completed_rollouts"], metrics["completed_rollouts"] + metrics["failed_rollouts"])
                for name, metrics in per_motion.items()
            }
            partition = build_motion_partition(motion_names, success_counts, expected_rollouts=5, threshold=0.8)
            print(
                f"STAGE1_PAPER_SPLIT checkpoint={args.eval_checkpoint} motions={len(motion_names)} "
                f"rollouts={completed_trials} successes={successful_episodes} "
                f"mastered={len(partition['mastered'])} challenging={len(partition['challenging'])} "
                f"steps={steps_run} env_cap={args.num_envs} envs={eval_num_envs} "
                f"terminations={term_counts}", flush=True,
            )
        else:
            print(f"STAGE1_EVAL checkpoint={args.eval_checkpoint or 'zero_action_baseline'} steps={steps_run} "
                  f"episodes={len(finished_lengths)} mean_return={np.mean(finished_returns) if finished_returns else float('nan'):.4f} "
                  f"mean_episode_length={np.mean(finished_lengths) if finished_lengths else float('nan'):.2f} "
                  f"completed_rollouts={successful_episodes} mean_abs_action={action_abs_sum / steps_run:.4f} "
                  f"max_abs_action={action_max_abs:.4f} mean_joint_error_l2={joint_error_sum / steps_run:.4f} "
                  f"terminations={term_counts} per_motion={per_motion} "
                  f"censored_envs={int((episode_lengths > 0).sum().item())}", flush=True)
        if args.eval_output is not None or paper_eval:
            import json
            result = {
                "checkpoint": str(args.eval_checkpoint) if args.eval_checkpoint is not None else "zero_action_baseline",
                "seed": args.seed, "num_envs": eval_num_envs,
                "steps": steps_run, "requested_max_num_envs": args.num_envs,
                "motion_files": [str(path.resolve()) for path in motion_files],
                "mean_return": float(np.mean(finished_returns)) if finished_returns else None,
                "mean_episode_length": float(np.mean(finished_lengths)) if finished_lengths else None,
                "completed_rollouts": successful_episodes, "terminations": term_counts,
                "evaluation_mode": (
                    "five_seeded_first_frame_rollouts_per_motion_clip_clean_observation_no_domain_randomization"
                    if paper_eval else "fixed_motion_fixed_frame_clean_observation_no_domain_randomization"
                ),
                "mean_abs_action": action_abs_sum / steps_run,
                "max_abs_action": action_max_abs,
                "mean_joint_error_l2": joint_error_sum / steps_run,
                "per_motion": per_motion,
            }
            if paper_eval:
                result["rollouts_per_motion"] = 5
                result["mastered_threshold"] = 0.8
                result["partition"] = {
                    role: [f"{name}.npz" for name in names]
                    for role, names in partition.items()
                }
            output_path = args.eval_output or (
                ROOT / "logs" / "extreme_rgmt_eval" /
                f"paper_partition_{args.eval_checkpoint.stem if args.eval_checkpoint else 'zero_action'}_seed{args.seed}.json"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2) + "\n")
            print(f"STAGE1_EVAL_OUTPUT {output_path.resolve()}", flush=True)
    else:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        expected_timesteps = args.iterations * agent_cfg.num_steps_per_env * args.num_envs
        collected_timesteps = runner.tot_timesteps
        if collected_timesteps != expected_timesteps:
            raise RuntimeError(
                f"training gate collected {collected_timesteps} timesteps; expected {expected_timesteps}"
            )
        expected_iteration = args.iterations - 1
        if runner.current_learning_iteration != expected_iteration:
            raise RuntimeError(
                f"training gate ended at iteration {runner.current_learning_iteration}; "
                f"expected {expected_iteration}"
            )
        final_checkpoint = run_dir / f"model_{runner.current_learning_iteration}.pt"
        if not final_checkpoint.is_file():
            raise RuntimeError(f"training gate did not write final checkpoint: {final_checkpoint}")
        print(
            f"STAGE1_GATE_OK iterations={args.iterations} timesteps={collected_timesteps} "
            f"checkpoint={final_checkpoint}",
            flush=True,
        )
except BaseException as exc:
    training_error = exc
    print(f"STAGE1_TRAINING_EXCEPTION {type(exc).__name__}: {exc!r}", flush=True)
    traceback.print_exc()
finally:
    env.close()
    try:
        # Isaac Lab 6's deprecated AppLauncher.close() exits through its ExitStack.
        app.close(exit_code=1 if training_error is not None else 0)
    except SystemExit as close_exit:
        if training_error is None and close_exit.code not in (None, 0):
            raise
if training_error is not None:
    raise training_error
