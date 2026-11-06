from types import SimpleNamespace

import pytest
import torch

from extreme_rgmt.evaluation import (
    configure_paper_evaluation_terminations, fixed_motion_schedule, unwrap_manager_env,
    validate_fixed_evaluation_env, randomized_rollout_tasks,
    classify_motion_success, build_motion_partition, extract_validated_partition,
)


def test_fixed_schedule_round_robins_motion_ids_and_holds_frame():
    motions, frames = fixed_motion_schedule(7, 3, frame_id=12)
    assert motions.tolist() == [0, 1, 2, 0, 1, 2, 0]
    assert frames.tolist() == [12] * 7


def test_fixed_schedule_spreads_envs_over_larger_motion_set():
    motions, frames = fixed_motion_schedule(4, 10, frame_id=7)
    assert motions.tolist() == [0, 3, 6, 9]
    assert frames.tolist() == [7] * 4


def test_unwrap_resolves_raw_manager_env_through_rsl_and_gym_wrappers():
    base = SimpleNamespace(command_manager=object(), event_manager=object())
    gym_wrapper = SimpleNamespace(unwrapped=base)
    rsl_wrapper = SimpleNamespace(unwrapped=gym_wrapper)
    assert unwrap_manager_env(rsl_wrapper) is base


def _make_env():
    clean_proprio = lambda: None
    clean_command = lambda: None
    obs = SimpleNamespace(**{
        name: SimpleNamespace(enable_corruption=False, proprio=SimpleNamespace(func=clean_proprio))
        for name in ("policy", "proprio", "proprio_history")
    })
    obs.command_window = SimpleNamespace(
        enable_corruption=False, command_window=SimpleNamespace(func=clean_command)
    )
    expected_motion, expected_frame = fixed_motion_schedule(3, 2, 4)
    command = SimpleNamespace(
        motion_ids=expected_motion.clone(), frame_ids=expected_frame.clone(),
        data_store=SimpleNamespace(motions=[
            SimpleNamespace(source_file=f"clip{i}.npz") for i in range(2)
        ]),
    )
    observation_manager = SimpleNamespace(
        _group_obs_term_names={
            "policy": ["proprio"], "proprio": ["proprio"],
            "proprio_history": ["proprio"], "command_window": ["command_window"],
        },
        _group_obs_term_cfgs={
            "policy": [SimpleNamespace(func=clean_proprio, noise=None)],
            "proprio": [SimpleNamespace(func=clean_proprio, noise=None)],
            "proprio_history": [SimpleNamespace(func=clean_proprio, noise=None)],
            "command_window": [SimpleNamespace(func=clean_command, noise=None)],
        },
    )
    base = SimpleNamespace(
        observation_manager=observation_manager,
        cfg=SimpleNamespace(events=SimpleNamespace(), observations=obs),
        event_manager=SimpleNamespace(active_terms=[]),
        command_manager=SimpleNamespace(get_term=lambda _: command),
    )
    base.cfg.events = SimpleNamespace(**{name: None for name in (
        "physics_material", "add_joint_default_pos", "base_com", "add_base_mass",
        "randomize_motor_strength", "randomize_pd_gains", "randomize_armature", "push_robot",
    )})
    wrapped = SimpleNamespace(unwrapped=SimpleNamespace(unwrapped=base))
    return wrapped, expected_motion, expected_frame, clean_proprio, clean_command


def test_runtime_eval_validation_checks_underlying_env_and_exact_schedule():
    env, motion, frame, clean_proprio, clean_command = _make_env()
    assert validate_fixed_evaluation_env(
        env, expected_motion_ids=motion, expected_frame_ids=frame, expected_motion_files=["clip0.npz", "clip1.npz"],
        clean_proprio_fn=clean_proprio, clean_command_fn=clean_command,
    ) is env.unwrapped.unwrapped


def test_runtime_eval_validation_rejects_stale_motion_schedule():
    env, motion, frame, clean_proprio, clean_command = _make_env()
    env.unwrapped.unwrapped.command_manager.get_term("motion").frame_ids[0] = 99
    with pytest.raises(AssertionError, match="frame IDs"):
        validate_fixed_evaluation_env(
            env, expected_motion_ids=motion, expected_frame_ids=frame, expected_motion_files=["clip0.npz", "clip1.npz"],
            clean_proprio_fn=clean_proprio, clean_command_fn=clean_command,
        )


def test_runtime_eval_validation_rejects_custom_observation_noise():
    env, motion, frame, clean_proprio, clean_command = _make_env()
    env.unwrapped.unwrapped.cfg.observations.proprio_history.proprio.func = lambda: None
    with pytest.raises(AssertionError, match="noisy proprio"):
        validate_fixed_evaluation_env(
            env, expected_motion_ids=motion, expected_frame_ids=frame, expected_motion_files=["clip0.npz", "clip1.npz"],
            clean_proprio_fn=clean_proprio, clean_command_fn=clean_command,
        )


def test_runtime_eval_validation_rejects_noise_in_built_observation_manager():
    env, motion, frame, clean_proprio, clean_command = _make_env()
    env.unwrapped.unwrapped.observation_manager._group_obs_term_cfgs["proprio_history"][0].func = lambda: None
    with pytest.raises(AssertionError, match="live observation manager"):
        validate_fixed_evaluation_env(
            env, expected_motion_ids=motion, expected_frame_ids=frame, expected_motion_files=["clip0.npz", "clip1.npz"],
            clean_proprio_fn=clean_proprio, clean_command_fn=clean_command,
        )


def test_runtime_eval_validation_rejects_noise_in_built_observation_manager():
    env, motion, frame, clean_proprio, clean_command = _make_env()
    env.unwrapped.unwrapped.observation_manager._group_obs_term_cfgs["proprio_history"][0].func = lambda: None
    with pytest.raises(AssertionError, match="live observation manager"):
        validate_fixed_evaluation_env(
            env, expected_motion_ids=motion, expected_frame_ids=frame, expected_motion_files=["clip0.npz", "clip1.npz"],
            clean_proprio_fn=clean_proprio, clean_command_fn=clean_command,
        )


def test_runtime_eval_validation_rejects_live_randomization():
    env, motion, frame, clean_proprio, clean_command = _make_env()
    env.unwrapped.unwrapped.event_manager.active_terms = {"interval": ["push_robot"]}
    with pytest.raises(AssertionError, match="live event manager"):
        validate_fixed_evaluation_env(
            env, expected_motion_ids=motion, expected_frame_ids=frame, expected_motion_files=["clip0.npz", "clip1.npz"],
            clean_proprio_fn=clean_proprio, clean_command_fn=clean_command,
        )


def test_paper_evaluation_uses_root_height_threshold_and_disables_extra_failures():
    cfg = SimpleNamespace(
        anchor_pos=SimpleNamespace(params={"threshold": 0.4}),
        anchor_ori=object(),
        ee_body_pos=object(),
        time_out=object(),
    )
    configure_paper_evaluation_terminations(cfg)
    assert cfg.anchor_pos.params["threshold"] == 0.2
    assert cfg.anchor_ori is None
    assert cfg.ee_body_pos is None
    assert cfg.time_out is not None


def test_paper_evaluation_rejects_missing_height_termination():
    with pytest.raises(ValueError, match="anchor_pos"):
        configure_paper_evaluation_terminations(SimpleNamespace(anchor_pos=None))


def test_rollout_tasks_start_at_first_frame_with_exact_seeded_trial_counts():
    tasks = randomized_rollout_tasks([1, 4, 11], rollouts_per_motion=5, seed=17)
    assert len(tasks) == 15
    assert [sum(motion_id == i for motion_id, _ in tasks) for i in range(3)] == [5, 5, 5]
    assert all(frame == 0 for _, frame in tasks)
    assert tasks == randomized_rollout_tasks([1, 4, 11], rollouts_per_motion=5, seed=17)
    assert tasks != randomized_rollout_tasks([1, 4, 11], rollouts_per_motion=5, seed=18)


def test_paper_partition_requires_five_trials_and_four_successes():
    split = build_motion_partition(
        ["a", "b", "c"], {"a": (4, 5), "b": (3, 5), "c": (0, 5)}
    )
    assert split == {"mastered": ["a"], "challenging": ["b", "c"]}
    assert classify_motion_success(4, 5) == "mastered"
    assert classify_motion_success(3, 5) == "challenging"
    with pytest.raises(ValueError, match="exactly 5"):
        build_motion_partition(["a"], {"a": (4, 4)})


def test_stage2_accepts_evaluator_json_only_when_partition_matches_five_trials():
    report = {
        "rollouts_per_motion": 5, "mastered_threshold": 0.8,
        "per_motion": {
            "a": {"rollout_trials": [{"success": True}] * 4 + [{"success": False}]},
            "b": {"rollout_trials": [{"success": True}] * 3 + [{"success": False}] * 2},
        },
        "partition": {"mastered": ["a.npz"], "challenging": ["b.npz"]},
    }
    assert extract_validated_partition(report) == report["partition"]
    report["partition"]["mastered"] = []
    with pytest.raises(ValueError, match="does not match"):
        extract_validated_partition(report)
