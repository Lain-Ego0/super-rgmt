#!/usr/bin/env python3
"""Train Extreme-RGMT Stage II from a Stage-I checkpoint and clip partition.

The split JSON must contain ``mastered`` and ``challenging`` arrays of NPZ
basenames produced by Stage-I clip evaluation and the 80% completion rule.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PMT_ROOT = Path(os.environ.get("PMT_ROOT", ROOT / "third_party" / "PMT-pinned")).expanduser().resolve()
if not (PMT_ROOT / "pmt_tasks" / "builder.py").is_file():
    raise SystemExit(f"PMT backend not found at {PMT_ROOT}")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PMT_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=10_000)
parser.add_argument("--seed", type=int, default=2)
parser.add_argument("--motion-dir", type=Path, required=True)
parser.add_argument("--split-json", type=Path, required=True)
parser.add_argument("--stage1-checkpoint", type=Path, required=True)
parser.add_argument("--run-name", default=None)
parser.add_argument("--headless", action="store_true", default=False)
if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    parser.print_help()
    raise SystemExit(0)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
for path in (args.motion_dir, args.split_json, args.stage1_checkpoint):
    if not path.exists():
        raise SystemExit(f"required input does not exist: {path}")
motion_files = sorted(args.motion_dir.rglob("*.npz"))
if not motion_files:
    raise SystemExit(f"no .npz motion clips under {args.motion_dir}")
from extreme_rgmt.evaluation import extract_validated_partition
try:
    split = extract_validated_partition(json.loads(args.split_json.read_text()))
except (TypeError, ValueError) as exc:
    raise SystemExit(str(exc)) from exc
by_name = {path.name: i for i, path in enumerate(motion_files)}
if len(by_name) != len(motion_files):
    raise SystemExit("motion basenames must be unique")
mastered_names, challenging_names = split["mastered"], split["challenging"]
if not mastered_names or not challenging_names:
    raise SystemExit("Stage-II split must contain at least one mastered and one challenging clip")
if set(mastered_names) & set(challenging_names):
    raise SystemExit("mastered and challenging lists overlap")
if set(mastered_names) | set(challenging_names) != set(by_name):
    raise SystemExit("split JSON must partition every NPZ in --motion-dir exactly once")

app = AppLauncher(args).app
from extreme_rgmt.configuration import configure_paper_rewards  # noqa: E402
from extreme_rgmt.domain_randomization import configure_paper_domain_randomization  # noqa: E402
from extreme_rgmt.mdp import (  # noqa: E402
    feet_slip, paper_noisy_command_window, paper_noisy_proprio, randomize_motor_strength,
)
from extreme_rgmt.models import register_with_motion_tracking_rl  # noqa: E402
from extreme_rgmt.pace_integration import PaceMotionGroups, PacePPOAdapter  # noqa: E402
from extreme_rgmt.stage2 import PACEProgress  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab.envs import mdp as isaac_mdp  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm, RewardTermCfg as RewTerm, SceneEntityCfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from motion_tracking_rl import registry  # noqa: E402
from pmt_tasks.isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from pmt_tasks.registry_gym import register_pmt_tasks  # noqa: E402

registry.autoload()
register_with_motion_tracking_rl()
register_pmt_tasks()
task_id = "RGMT-G1-v0"
env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = args.seed
if args.device is not None:
    env_cfg.sim.device = args.device
env_cfg.pmt_motion_paths = [str(path.resolve()) for path in motion_files]
env_cfg.__post_init__()
# Stage-II inputs are already clip-level motion files. Disable PMT's generic
# 4,000-frame chunking so clip IDs used by PACE remain stable and one-to-one.
env_cfg.commands.motion.chunk_length = 0
configure_paper_domain_randomization(
    env_cfg, event_term_cls=EventTerm, scene_entity_cfg_cls=SceneEntityCfg,
    mdp=isaac_mdp, motor_strength_func=randomize_motor_strength,
)
for group_name in ("policy", "proprio", "proprio_history"):
    getattr(env_cfg.observations, group_name).proprio.func = paper_noisy_proprio
env_cfg.observations.command_window.command_window.func = paper_noisy_command_window
configure_paper_rewards(
    env_cfg.rewards, reward_term_cls=RewTerm, scene_entity_cfg_cls=SceneEntityCfg,
    feet_slip_func=feet_slip,
)
agent_cfg.seed = args.seed
agent_cfg.max_iterations = args.iterations
agent_cfg.experiment_name = "extreme_rgmt_stage2"
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
for key in (
    "actor_obs_normalization", "critic_obs_normalization", "history_obs_normalization",
    "command_obs_normalization",
):
    setattr(agent_cfg.policy, key, False)
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

env = None
group_sampler = None
pace_adapter = None
training_error = None
try:
    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    unwrapped = env.unwrapped
    command = unwrapped.command_manager.get_term("motion")
    loaded_names = [Path(motion.source_file).name for motion in command.data_store.motions]
    if len(loaded_names) != len(set(loaded_names)):
        raise RuntimeError("PMT loaded duplicate motion basenames; use a unique split manifest")
    if set(loaded_names) != set(mastered_names) | set(challenging_names):
        raise RuntimeError(
            "Stage-II partition does not match PMT's loaded motion files: "
            f"loaded={len(loaded_names)}, listed={len(mastered_names) + len(challenging_names)}"
        )
    loaded_index = {name: index for index, name in enumerate(loaded_names)}
    mastered_ids = [loaded_index[name] for name in mastered_names]
    challenging_ids = [loaded_index[name] for name in challenging_names]
    group_sampler = PaceMotionGroups(
        command,
        num_envs=args.num_envs,
        acquisition_fraction=0.8,
        mastered_motion_ids=mastered_ids,
        challenging_motion_ids=challenging_ids,
    )

    runner_class = registry.get_runner(agent_cfg.class_name)
    runner_cfg = agent_cfg.to_dict()
    for key in (
        "anchor_estimator_detach", "anchor_estimator_hidden_dims", "anchor_estimator_latent_inputs",
        "anchor_estimator_output_dim", "anchor_gt_normalization", "forward_hidden_dims",
        "forward_obs_normalization", "use_anchor_estimator", "use_vel_estimator",
        "vel_estimator_detach", "vel_estimator_hidden_dims", "vel_estimator_output_dim",
        "vel_gt_normalization",
    ):
        runner_cfg["policy"].pop(key, None)
    run_dir = ROOT / "logs" / "extreme_rgmt_stage2" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    runner = runner_class(env, runner_cfg, log_dir=str(run_dir), device=agent_cfg.device)
    runner.load(str(args.stage1_checkpoint), load_optimizer=False, map_location=agent_cfg.device)
    # Stage II starts a fresh PPO optimizer and its own iteration clock.
    runner.current_learning_iteration = 0
    runner.alg.set_learning_iteration(0)
    pace_adapter = PacePPOAdapter(
        runner.alg,
        group_sampler,
        num_steps=agent_cfg.num_steps_per_env,
        num_mini_batches=agent_cfg.algorithm.num_mini_batches,
        num_epochs=agent_cfg.algorithm.num_learning_epochs,
        top_fraction=0.05,
        resample_ratio=0.25,
        progress=PACEProgress(rho_ref=0.6, beta=0.99, lambda_base=0.3, kappa=5.0),
    )
    runner.add_git_repo_to_log(__file__)
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
    expected_steps = args.iterations * agent_cfg.num_steps_per_env * args.num_envs
    final_checkpoint = run_dir / f"model_{runner.current_learning_iteration}.pt"
    if runner.tot_timesteps != expected_steps:
        raise RuntimeError(f"collected {runner.tot_timesteps} timesteps; expected {expected_steps}")
    if runner.current_learning_iteration != args.iterations - 1 or not final_checkpoint.is_file():
        raise RuntimeError(f"Stage-II final checkpoint was not written: {final_checkpoint}")
    (run_dir / "pace_state.json").write_text(json.dumps({
        "progress": pace_adapter.progress.state_dict(),
        "mastered": mastered_names,
        "challenging": challenging_names,
        "stage1_checkpoint": str(args.stage1_checkpoint.resolve()),
        "seed": args.seed,
        "num_envs": args.num_envs,
        "iterations": args.iterations,
    }, indent=2) + "\n")
    print(
        f"STAGE2_GATE_OK iterations={args.iterations} timesteps={runner.tot_timesteps} "
        f"mastered={len(mastered_names)} challenging={len(challenging_names)} "
        f"checkpoint={final_checkpoint}",
        flush=True,
    )
except BaseException as exc:
    training_error = exc
    print(f"STAGE2_TRAINING_EXCEPTION {type(exc).__name__}: {exc!r}", flush=True)
    traceback.print_exc()
finally:
    if pace_adapter is not None:
        pace_adapter.close()
    if group_sampler is not None:
        group_sampler.close()
    if env is not None:
        env.close()
    try:
        app.close(exit_code=1 if training_error is not None else 0)
    except SystemExit as close_exit:
        if training_error is None and close_exit.code not in (None, 0):
            raise
if training_error is not None:
    raise training_error
