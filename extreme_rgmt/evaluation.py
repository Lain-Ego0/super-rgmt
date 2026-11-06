"""Deterministic Stage-I evaluation helpers and runtime assertions."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import random

import torch


RANDOMIZED_EVENT_NAMES = (
    "physics_material",
    "add_joint_default_pos",
    "base_com",
    "add_base_mass",
    "randomize_motor_strength",
    "randomize_pd_gains",
    "randomize_armature",
    "push_robot",
)


def configure_paper_evaluation_terminations(termination_cfg: Any, height_threshold: float = 0.2) -> None:
    """Use the paper's rollout failure criterion: root-height error above 0.2 m.

    PMT adds anchor-orientation and selected end-effector position terminations;
    those are not part of the paper's stated evaluation success definition.
    Training termination configuration is left untouched by this helper.
    """
    if height_threshold <= 0.0:
        raise ValueError("height_threshold must be positive")
    anchor_pos = getattr(termination_cfg, "anchor_pos", None)
    if anchor_pos is None or not isinstance(getattr(anchor_pos, "params", None), dict):
        raise ValueError("paper evaluation requires an anchor_pos termination with mutable params")
    anchor_pos.params["threshold"] = float(height_threshold)
    for name in ("anchor_ori", "ee_body_pos"):
        if hasattr(termination_cfg, name):
            setattr(termination_cfg, name, None)


def fixed_motion_schedule(num_envs: int, num_motions: int, frame_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign envs round-robin over motions at one explicit reference frame."""
    if num_envs < 1 or num_motions < 1 or frame_id < 0:
        raise ValueError("num_envs and num_motions must be positive and frame_id nonnegative")
    if num_envs < num_motions:
        motion_ids = torch.linspace(0, num_motions - 1, steps=num_envs).round().to(torch.long)
    else:
        motion_ids = torch.arange(num_envs, dtype=torch.long).remainder(num_motions)
    frame_ids = torch.full((num_envs,), frame_id, dtype=torch.long)
    return motion_ids, frame_ids


def unwrap_manager_env(env: Any) -> Any:
    """Resolve the underlying ManagerBasedRLEnv beneath Gym/RSL wrappers."""
    candidate = env
    seen: set[int] = set()
    while id(candidate) not in seen:
        seen.add(id(candidate))
        if hasattr(candidate, "command_manager") and hasattr(candidate, "event_manager"):
            return candidate
        next_candidate = getattr(candidate, "unwrapped", None)
        if next_candidate is None or next_candidate is candidate:
            break
        candidate = next_candidate
    raise AssertionError(f"could not resolve ManagerBasedRLEnv beneath {type(env).__name__}")


def validate_fixed_evaluation_env(
    env: Any,
    *,
    expected_motion_ids: torch.Tensor,
    expected_frame_ids: torch.Tensor,
    expected_motion_files: list[str] | None = None,
    clean_proprio_fn: Any,
    clean_command_fn: Any,
) -> Any:
    """Fail closed unless the live base env uses clean inputs and exact reset IDs."""
    base = unwrap_manager_env(env)
    cfg = base.cfg
    for event_name in RANDOMIZED_EVENT_NAMES:
        if getattr(cfg.events, event_name, None) is not None:
            raise AssertionError(f"evaluation config still enables randomized event {event_name}")
        active_event_terms = getattr(base.event_manager, "active_terms", ())
        if isinstance(active_event_terms, dict):
            active_event_names = {term for terms in active_event_terms.values() for term in terms}
        else:
            active_event_names = set(active_event_terms)
        if event_name in active_event_names:
            raise AssertionError(f"live event manager still enables randomized event {event_name}")
    for group_name in ("policy", "proprio", "proprio_history"):
        group = getattr(cfg.observations, group_name)
        if group.enable_corruption:
            raise AssertionError(f"evaluation observation group {group_name} still enables corruption")
        if group.proprio.func is not clean_proprio_fn:
            raise AssertionError(f"evaluation observation group {group_name} still uses a noisy proprio function")
    command_group = cfg.observations.command_window
    if command_group.enable_corruption or command_group.command_window.func is not clean_command_fn:
        raise AssertionError("evaluation command window is not clean")

    # Confirm the already-built observation manager retained those exact callables,
    # rather than merely checking a config object that was not used to construct it.
    observation_manager = base.observation_manager
    for group_name in ("policy", "proprio", "proprio_history"):
        names = observation_manager._group_obs_term_names[group_name]
        term_cfgs = observation_manager._group_obs_term_cfgs[group_name]
        index = names.index("proprio")
        if term_cfgs[index].func is not clean_proprio_fn or term_cfgs[index].noise is not None:
            raise AssertionError(f"live observation manager group {group_name} is not clean")
    names = observation_manager._group_obs_term_names["command_window"]
    term_cfgs = observation_manager._group_obs_term_cfgs["command_window"]
    index = names.index("command_window")
    if term_cfgs[index].func is not clean_command_fn or term_cfgs[index].noise is not None:
        raise AssertionError("live observation manager command window is not clean")

    command = base.command_manager.get_term("motion")
    expected_motion_ids = expected_motion_ids.to(command.motion_ids.device)
    expected_frame_ids = expected_frame_ids.to(command.frame_ids.device)
    if not torch.equal(command.motion_ids, expected_motion_ids):
        raise AssertionError(
            f"live motion IDs differ from fixed schedule: actual={command.motion_ids.tolist()} "
            f"expected={expected_motion_ids.tolist()}"
        )
    if not torch.equal(command.frame_ids, expected_frame_ids):
        raise AssertionError(
            f"live frame IDs differ from fixed schedule: actual={command.frame_ids.tolist()} "
            f"expected={expected_frame_ids.tolist()}"
        )
    if expected_motion_files is not None:
        loaded_files = [Path(motion.source_file).name for motion in command.data_store.motions]
        if loaded_files != expected_motion_files:
            raise AssertionError(
                f"live motion-file order differs from evaluation manifest: "
                f"loaded={loaded_files[:5]}... ({len(loaded_files)}) "
                f"expected={expected_motion_files[:5]}... ({len(expected_motion_files)})"
            )
    return base


def randomized_rollout_tasks(
    motion_lengths: Iterable[int], *, rollouts_per_motion: int = 5, seed: int = 0
) -> list[tuple[int, int]]:
    """Return five seeded/shuffled first-frame rollout tasks for every motion."""
    lengths = [int(length) for length in motion_lengths]
    if not lengths or any(length < 1 for length in lengths):
        raise ValueError("motion_lengths must contain positive lengths")
    if rollouts_per_motion < 1:
        raise ValueError("rollouts_per_motion must be positive")
    rng = random.Random(seed)
    tasks = [
        (motion_id, 0)
        for motion_id, _length in enumerate(lengths)
        for _ in range(rollouts_per_motion)
    ]
    rng.shuffle(tasks)
    return tasks


def classify_motion_success(
    successful_rollouts: int, total_rollouts: int, *, threshold: float = 0.8
) -> str:
    """Classify a fully evaluated clip; reject incomplete or invalid trial counts."""
    if total_rollouts < 1 or not 0 <= successful_rollouts <= total_rollouts:
        raise ValueError("rollout counts must satisfy 0 <= successful <= total and total > 0")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    return "mastered" if successful_rollouts / total_rollouts >= threshold else "challenging"


def build_motion_partition(
    motion_names: list[str],
    success_counts: dict[str, tuple[int, int]],
    *,
    expected_rollouts: int = 5,
    threshold: float = 0.8,
) -> dict[str, list[str]]:
    """Build a disjoint, exhaustive split and fail if any clip lacks exact trials."""
    if len(set(motion_names)) != len(motion_names):
        raise ValueError("motion names must be unique")
    if set(success_counts) != set(motion_names):
        raise ValueError("success counts must cover exactly the requested motion names")
    result = {"mastered": [], "challenging": []}
    for name in motion_names:
        successes, total = success_counts[name]
        if total != expected_rollouts:
            raise ValueError(f"{name}: expected exactly {expected_rollouts} rollouts, got {total}")
        result[classify_motion_success(successes, total, threshold=threshold)].append(name)
    return result


def extract_validated_partition(payload: dict[str, Any]) -> dict[str, list[str]]:
    """Read a split-only JSON or validate an evaluator report before Stage II."""
    if set(payload) == {"mastered", "challenging"}:
        partition = payload
    else:
        if payload.get("rollouts_per_motion") != 5 or payload.get("mastered_threshold") != 0.8:
            raise ValueError("evaluation report must record five rollouts and the 0.8 mastered threshold")
        metrics = payload.get("per_motion")
        partition = payload.get("partition")
        if not isinstance(metrics, dict) or not isinstance(partition, dict):
            raise ValueError("evaluation report must include per_motion metrics and a partition")
        expected: dict[str, list[str]] = {"mastered": [], "challenging": []}
        for name, result in metrics.items():
            trials = result.get("rollout_trials", [])
            if len(trials) != 5:
                raise ValueError(f"{name}: expected exactly five recorded rollout trials")
            successes = sum(bool(trial.get("success")) for trial in trials)
            role = "mastered" if successes / 5 >= 0.8 else "challenging"
            expected[role].append(f"{name}.npz")
        for role in expected:
            expected[role].sort()
        normalized = {role: sorted(partition.get(role, [])) for role in expected}
        if normalized != expected:
            raise ValueError("stored partition does not match the five-rollout outcomes and 0.8 threshold")
        partition = normalized
    if set(partition) != {"mastered", "challenging"}:
        raise ValueError("split JSON must contain exactly 'mastered' and 'challenging' arrays")
    if not all(
        isinstance(partition[role], list)
        and all(isinstance(name, str) for name in partition[role])
        and len(partition[role]) == len(set(partition[role]))
        for role in partition
    ):
        raise ValueError("partition roles must contain unique NPZ basename strings")
    if set(partition["mastered"]) & set(partition["challenging"]):
        raise ValueError("mastered and challenging motion sets overlap")
    return partition
