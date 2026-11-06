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
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--motion-dir", type=Path, required=True)
parser.add_argument("--run-name", default=None)
parser.add_argument("--headless", action="store_true", default=False)
if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    parser.print_help()
    raise SystemExit(0)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.motion_dir.exists():
    raise SystemExit(f"motion directory does not exist: {args.motion_dir}")
motion_files = sorted(args.motion_dir.rglob("*.npz")) if args.motion_dir.is_dir() else [args.motion_dir]
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
from isaaclab.managers import EventTermCfg as EventTerm, RewardTermCfg as RewTerm, SceneEntityCfg  # noqa: E402
from isaaclab.envs import mdp as isaac_mdp  # noqa: E402
from extreme_rgmt.mdp import (  # noqa: E402
    feet_slip, paper_noisy_command_window, paper_noisy_proprio, randomize_motor_strength,
)
from extreme_rgmt.configuration import configure_paper_rewards  # noqa: E402
from extreme_rgmt.domain_randomization import configure_paper_domain_randomization  # noqa: E402
from extreme_rgmt.models import register_with_motion_tracking_rl  # noqa: E402

registry.autoload()
register_with_motion_tracking_rl()
register_pmt_tasks()

task_id = "RGMT-G1-v0"
env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")

# Retain PMT's tested G1 flat tracking environment, adaptive multi-motion command,
# reward terms and simulator wrapper; use Extreme-RGMT's actor-critic and PPO values.
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = args.seed
if args.device is not None:
    env_cfg.sim.device = args.device
env_cfg.pmt_motion_paths = [str(path.resolve()) for path in motion_files]
env_cfg.__post_init__()
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
agent_cfg.seed = args.seed
agent_cfg.max_iterations = args.iterations
agent_cfg.experiment_name = "extreme_rgmt_stage1"
agent_cfg.run_name = args.run_name
agent_cfg.num_steps_per_env = 24
agent_cfg.save_interval = 100
agent_cfg.policy.class_name = "ExtremeRGMTActorCritic"
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

env = gym.make(task_id, cfg=env_cfg, render_mode=None)
from pmt_tasks.isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

env = RslRlVecEnvWrapper(env)
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
training_error = None
try:
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
