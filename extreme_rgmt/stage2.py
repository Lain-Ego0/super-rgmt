"""Pure PACE/STAR optimization helpers from Extreme-RGMT Sections V-A/B.

Inputs are flattened acquisition/consolidation rollout transitions. The caller
must provide per-transition motion-bin and episode-fragment identifiers.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from types import MethodType
from dataclasses import dataclass

import torch
from torch import Tensor


def allocate_pace_environments(num_envs: int, acquisition_fraction: float = 0.8) -> tuple[int, int]:
    """Return fixed acquisition/consolidation environment counts (PACE Alg. 1).

    The paper specifies a fraction but not integer rounding. We use nearest
    integer and require at least one environment per role when possible.
    """
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    if not 0.0 <= acquisition_fraction <= 1.0:
        raise ValueError("acquisition_fraction must be in [0, 1]")
    acquisition = round(num_envs * acquisition_fraction)
    if num_envs >= 2 and 0.0 < acquisition_fraction < 1.0:
        acquisition = min(max(acquisition, 1), num_envs - 1)
    return acquisition, num_envs - acquisition


@dataclass
class PACEProgress:
    """EMA state and progress-adaptive consolidation coefficient (Eq. 17–19)."""

    rho_ref: float = 0.6
    beta: float = 0.99
    lambda_base: float = 0.3
    kappa: float = 5.0
    rho_ema: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.rho_ref <= 1.0:
            raise ValueError("rho_ref must be in [0, 1]")
        if not 0.0 <= self.beta < 1.0:
            raise ValueError("beta must be in [0, 1)")
        if not 0.0 <= self.lambda_base <= 1.0 or self.kappa < 0.0:
            raise ValueError("invalid consolidation schedule")
        if self.rho_ema is None:
            self.rho_ema = self.rho_ref

    def state_dict(self) -> dict[str, float]:
        """Serialize schedule state so resumed runs preserve the EMA."""
        return {
            "rho_ref": self.rho_ref,
            "beta": self.beta,
            "lambda_base": self.lambda_base,
            "kappa": self.kappa,
            "rho_ema": float(self.rho_ema),
        }

    def load_state_dict(self, state: dict[str, float]) -> None:
        expected = {"rho_ref", "beta", "lambda_base", "kappa", "rho_ema"}
        if set(state) != expected:
            raise ValueError(f"PACE state must contain exactly {sorted(expected)}")
        restored = PACEProgress(**state)
        self.rho_ref = restored.rho_ref
        self.beta = restored.beta
        self.lambda_base = restored.lambda_base
        self.kappa = restored.kappa
        self.rho_ema = restored.rho_ema

    def update(self, valid_acquisition: int, valid_consolidation: int) -> float:
        total = valid_acquisition + valid_consolidation
        if valid_acquisition < 0 or valid_consolidation < 0 or total <= 0:
            raise ValueError("valid sample counts must be nonnegative with positive total")
        rho = valid_acquisition / total
        self.rho_ema = self.beta * float(self.rho_ema) + (1.0 - self.beta) * rho
        return min(1.0, self.lambda_base + self.kappa * max(0.0, self.rho_ema - self.rho_ref))


def rollout_fragment_ids(dones: Tensor) -> tuple[Tensor, Tensor]:
    """Flatten ``[time, env]`` terminal flags into STAR env/fragment IDs.

    Each maximal contiguous rollout segment within an environment receives its
    own fragment ID. A transition marked done belongs to the fragment it ends;
    the following transition starts a new fragment. Returned vectors follow
    time-major flattening, matching ``storage.tensor.flatten(0, 1)``.
    """
    if dones.ndim != 2 or dones.shape[0] == 0 or dones.shape[1] == 0:
        raise ValueError("dones must be a nonempty [time, env] tensor")
    terminal = dones.to(dtype=torch.bool)
    time_steps, num_envs = terminal.shape
    env_ids = torch.arange(num_envs, device=dones.device).expand(time_steps, -1)
    starts = torch.zeros_like(terminal, dtype=torch.long)
    if time_steps > 1:
        starts[1:] = terminal[:-1].long()
    fragment_ids = starts.cumsum(dim=0)
    return env_ids.reshape(-1), fragment_ids.reshape(-1)


class Stage2RolloutMetadata:
    """Sidecar for PPO storage fields absent from the upstream rollout schema.

    Append one vector per simulator step. ``role_ids`` uses 0 for acquisition
    and 1 for consolidation; the role assignment must stay fixed for each env
    throughout a rollout. ``bin_ids`` are the adaptive sampler's reference-bin
    IDs at the time the transition was collected.
    """

    def __init__(self, num_steps: int, num_envs: int, device: torch.device | str):
        if num_steps < 1 or num_envs < 1:
            raise ValueError("num_steps and num_envs must be positive")
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = torch.device(device)
        self._bins: list[Tensor] = []
        self._roles: list[Tensor] = []
        self._dones: list[Tensor] = []
        self._role_assignment: Tensor | None = None

    def append(self, bin_ids: Tensor, role_ids: Tensor, dones: Tensor) -> None:
        if len(self._bins) >= self.num_steps:
            raise RuntimeError("rollout metadata is already full")
        values = []
        for name, tensor in (("bin_ids", bin_ids), ("role_ids", role_ids), ("dones", dones)):
            tensor = tensor.to(device=self.device).reshape(-1)
            if tensor.numel() != self.num_envs:
                raise ValueError(f"{name} must contain one value per environment")
            values.append(tensor)
        bins, roles, terminal = values
        if torch.any(bins < 0):
            raise ValueError("bin_ids must be nonnegative")
        if torch.any((roles != 0) & (roles != 1)):
            raise ValueError("role_ids must be 0 (acquisition) or 1 (consolidation)")
        if self._role_assignment is None:
            self._role_assignment = roles.clone()
        elif not torch.equal(roles, self._role_assignment):
            raise ValueError("PACE environment roles must remain fixed during a rollout")
        self._bins.append(bins.long())
        self._roles.append(roles.long())
        self._dones.append(terminal.bool())

    def flatten(self) -> dict[str, Tensor]:
        if len(self._bins) != self.num_steps:
            raise RuntimeError(f"expected {self.num_steps} steps, received {len(self._bins)}")
        bins = torch.stack(self._bins)
        roles = torch.stack(self._roles)
        dones = torch.stack(self._dones)
        env_ids, fragment_ids = rollout_fragment_ids(dones)
        return {
            "bin_ids": bins.reshape(-1),
            "role_ids": roles.reshape(-1),
            "dones": dones.reshape(-1),
            "env_ids": env_ids,
            "fragment_ids": fragment_ids,
        }


def annotate_stage2_observations(
    observations,
    *,
    role_ids: Tensor,
    bin_ids: Tensor,
    role_key: str = "extreme_rgmt_role",
    bin_key: str = "extreme_rgmt_bin",
):
    """Attach rollout metadata to a TensorDict before ``PPO.act`` stores it.

    PMT's PPO stores the observation object during ``act`` and copies all its
    leaves into rollout storage in ``process_env_step``. Extra metadata leaves
    therefore survive flattening without editing the pinned storage code.
    Policy observation groups must continue to select only model input keys.
    """
    if role_key in observations or bin_key in observations:
        raise ValueError("Stage-II metadata keys already exist in observations")
    if role_ids.ndim != 1 or bin_ids.shape != role_ids.shape:
        raise ValueError("role_ids and bin_ids must be equal-length vectors")
    if torch.any((role_ids != 0) & (role_ids != 1)) or torch.any(bin_ids < 0):
        raise ValueError("invalid role or bin metadata")
    if observations.batch_size and observations.batch_size[-1] != role_ids.numel():
        raise ValueError("metadata length must match observation environment batch")
    # TensorDict.set mutates and returns the same object, matching PPO's rollout
    # observation reference semantics.
    observations.set(role_key, role_ids.to(device=observations.device, dtype=torch.long).unsqueeze(-1))
    observations.set(bin_key, bin_ids.to(device=observations.device, dtype=torch.long).unsqueeze(-1))
    return observations


@contextmanager
def star_pace_update(
    storage,
    metadata: dict[str, Tensor],
    bin_probabilities: Tensor,
    *,
    top_fraction: float = 0.05,
    resample_ratio: float = 0.25,
    num_mini_batches: int,
    num_epochs: int,
    pace_policy=None,
):
    """Temporarily install paper-aligned PACE/STAR data handling on PPO storage.

    PPO minibatches contain acquisition transitions only, so the PPO surrogate,
    critic loss, and entropy regularizer are all computed on the acquisition
    distribution. A separate mastered-state minibatch is exposed to the policy's
    auxiliary-loss hook for the PACE action-MSE term. All temporary state and the
    original generator are restored on context exit.
    """
    required = {"bin_ids", "role_ids", "env_ids", "fragment_ids"}
    if not required.issubset(metadata):
        raise ValueError(f"metadata is missing {sorted(required - set(metadata))}")
    if storage.training_type != "rl":
        raise ValueError("STAR/PACE update requires reinforcement-learning storage")
    if not hasattr(storage, "mini_batch_generator"):
        raise ValueError("storage must expose mini_batch_generator")

    n_steps, n_envs = storage.num_transitions_per_env, storage.num_envs
    count = n_steps * n_envs
    flat_meta = {key: metadata[key].to(storage.device).reshape(-1) for key in required}
    if any(value.numel() != count for value in flat_meta.values()):
        raise ValueError("each metadata vector must match flattened rollout size")
    roles = flat_meta["role_ids"]
    bins = flat_meta["bin_ids"]
    if torch.any((roles != 0) & (roles != 1)):
        raise ValueError("role_ids must be 0 (acquisition) or 1 (consolidation)")

    original_generator = storage.mini_batch_generator
    raw_advantages = (storage.returns - storage.values).reshape(-1)
    acquisition_flat_indices = torch.nonzero(roles == 0, as_tuple=False).flatten()
    consolidation_flat_indices = torch.nonzero(roles == 1, as_tuple=False).flatten()
    acquisition_advantages = raw_advantages[acquisition_flat_indices]
    acquisition_bins = bins[acquisition_flat_indices]
    normalized_acquisition_advantages, _ = difficulty_conditioned_advantages(
        acquisition_advantages,
        acquisition_bins,
        bin_probabilities.to(storage.device),
    )
    pool, pool_probs = star_fragment_pool(
        acquisition_advantages,
        acquisition_bins,
        bin_probabilities.to(storage.device),
        flat_meta["env_ids"][acquisition_flat_indices],
        flat_meta["fragment_ids"][acquisition_flat_indices],
        top_fraction,
    )
    if acquisition_flat_indices.numel() == 0 or consolidation_flat_indices.numel() == 0:
        raise ValueError("PACE requires at least one sample from each environment role")

    def _star_generator(self, requested_batches: int, requested_epochs: int = 8):
        if self.training_type != "rl":
            raise ValueError("STAR minibatch generation requires RL storage")
        if requested_batches != num_mini_batches or requested_epochs != num_epochs:
            raise ValueError("update minibatch settings differ from STAR context settings")
        if requested_batches > acquisition_flat_indices.numel() or requested_batches > consolidation_flat_indices.numel():
            raise ValueError("STAR needs at least one acquisition and consolidation transition per minibatch")

        obs = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_log_prob = self.actions_log_prob.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for _ in range(requested_epochs):
            shuffled_acquisition = acquisition_flat_indices[
                torch.randperm(acquisition_flat_indices.numel(), device=self.device)
            ]
            shuffled_consolidation = consolidation_flat_indices[
                torch.randperm(consolidation_flat_indices.numel(), device=self.device)
            ]
            acquisition_batches = torch.tensor_split(shuffled_acquisition, requested_batches)
            consolidation_batches = torch.tensor_split(shuffled_consolidation, requested_batches)
            for acquisition_regular_ids, consolidation_ids in zip(acquisition_batches, consolidation_batches):
                acq_count = acquisition_regular_ids.numel()
                if acq_count == 0 or consolidation_ids.numel() == 0:
                    raise ValueError("PACE minibatches cannot have an empty role")
                # STAR replaces a fraction of the acquisition minibatch with
                # weighted draws from its selected high-value fragment pool.
                local_ids = star_mixed_minibatch_indices(
                    pool,
                    pool_probs,
                    acquisition_flat_indices.numel(),
                    acq_count,
                    resample_ratio,
                )
                acquisition_ids = acquisition_flat_indices[local_ids]
                if pace_policy is not None:
                    pace_policy._pace_consolidation_obs = obs[consolidation_ids]
                try:
                    yield (
                        obs[acquisition_ids], actions[acquisition_ids], values[acquisition_ids],
                        normalized_acquisition_advantages[local_ids].unsqueeze(-1),
                        returns[acquisition_ids], old_log_prob[acquisition_ids],
                        old_mu[acquisition_ids], old_sigma[acquisition_ids], (None, None), None,
                    )
                finally:
                    if pace_policy is not None:
                        pace_policy._pace_consolidation_obs = None

    try:
        storage.mini_batch_generator = MethodType(_star_generator, storage)
        yield {"star_pool_size": int(pool.numel()), "acquisition_samples": int((roles == 0).sum())}
    finally:
        storage.mini_batch_generator = original_generator
        if pace_policy is not None:
            pace_policy._pace_consolidation_obs = None


def pace_objective(acquisition_loss: Tensor, current_actions: Tensor, reference_actions: Tensor, weight: float) -> Tensor:
    """Combine acquisition loss with mastered-motion policy MSE (Eq. 14–16)."""
    if current_actions.shape != reference_actions.shape:
        raise ValueError("current and reference action tensors must have identical shapes")
    if current_actions.numel() == 0:
        raise ValueError("consolidation action batch must not be empty")
    if weight < 0.0:
        raise ValueError("consolidation weight must be nonnegative")
    consolidation = torch.square(current_actions - reference_actions).mean()
    return acquisition_loss + weight * consolidation


def pace_role_objective(
    acquisition_ppo_loss: Tensor,
    role_ids: Tensor,
    current_action_means: Tensor,
    reference_action_means: Tensor,
    consolidation_weight: float,
) -> tuple[Tensor, Tensor]:
    """Combine acquisition PPO with mastered-only PACE policy alignment.

    ``acquisition_ppo_loss`` must already be computed from acquisition samples
    only. Consolidation samples contribute only the frozen Stage-I action MSE,
    matching Eq. (14)-(16); the returned pair is total and consolidation loss.
    """
    if role_ids.ndim != 1 or current_action_means.shape != reference_action_means.shape:
        raise ValueError("role_ids must be a vector and action tensors must match")
    if current_action_means.shape[0] != role_ids.numel():
        raise ValueError("one role ID is required for each action row")
    if torch.any((role_ids != 0) & (role_ids != 1)):
        raise ValueError("role_ids must be 0 (acquisition) or 1 (consolidation)")
    if consolidation_weight < 0.0:
        raise ValueError("consolidation_weight must be nonnegative")
    mastered = role_ids == 1
    if torch.any(mastered):
        consolidation_loss = (current_action_means[mastered] - reference_action_means[mastered]).square().mean()
    else:
        consolidation_loss = current_action_means.sum() * 0.0
    return acquisition_ppo_loss + consolidation_weight * consolidation_loss, consolidation_loss


@contextmanager
def pace_reference_constraint(
    algorithm,
    reference_policy,
    consolidation_weight: float,
    *,
    role_key: str = "extreme_rgmt_role",
):
    """Add the frozen Stage-I action-MSE term to PPO's actor surrogate.

    The rollout storage already carries ``role_key`` in its observation
    TensorDict. Acquisition advantages are supplied by :func:`star_pace_update`;
    this context adds Eq. (15) on consolidation rows after the current policy
    has evaluated each PPO minibatch. Both monkey-patched methods are restored
    even if an update raises.
    """
    if consolidation_weight < 0.0:
        raise ValueError("consolidation_weight must be nonnegative")
    policy = algorithm.policy
    if policy.is_recurrent:
        raise ValueError("PACE reference constraint currently requires a feed-forward policy")
    reference_policy.eval()
    for parameter in reference_policy.parameters():
        parameter.requires_grad_(False)

    original_act = policy.act
    original_compute = algorithm._compute_surrogate_loss
    state: dict[str, Tensor] = {}

    def act_with_reference(obs_batch, *args, **kwargs):
        result = original_act(obs_batch, *args, **kwargs)
        roles = obs_batch.get(role_key)
        if roles is None:
            raise KeyError(f"PPO minibatch is missing PACE role metadata {role_key!r}")
        roles = roles.reshape(-1).to(dtype=torch.long)
        current = policy.action_mean[: roles.numel()]
        with torch.no_grad():
            reference = reference_policy.act_inference(obs_batch)
        if reference.shape != current.shape:
            raise ValueError(
                f"reference action shape {tuple(reference.shape)} does not match "
                f"current policy shape {tuple(current.shape)}"
            )
        _, state["loss"] = pace_role_objective(
            current.sum() * 0.0,
            roles,
            current,
            reference,
            1.0,
        )
        return result

    def compute_with_reference(*args, **kwargs):
        result = original_compute(*args, **kwargs)
        if "loss" not in state:
            raise RuntimeError("PACE loss was not prepared by the policy minibatch evaluation")
        return (result[0] + consolidation_weight * state["loss"], *result[1:])

    policy.act = act_with_reference
    algorithm._compute_surrogate_loss = compute_with_reference
    try:
        yield state
    finally:
        policy.act = original_act
        algorithm._compute_surrogate_loss = original_compute


def difficulty_conditioned_advantages(
    raw_advantages: Tensor,
    bin_ids: Tensor,
    bin_probabilities: Tensor,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Return normalized advantages and Eq. 20 difficulty weights."""
    if raw_advantages.ndim != 1 or bin_ids.shape != raw_advantages.shape:
        raise ValueError("raw_advantages and bin_ids must be equal-length vectors")
    if bin_probabilities.ndim != 1 or bin_probabilities.numel() == 0:
        raise ValueError("bin_probabilities must be a nonempty vector")
    if torch.any(bin_ids < 0) or torch.any(bin_ids >= bin_probabilities.numel()):
        raise ValueError("bin_ids contain an out-of-range bin")
    if torch.any(bin_probabilities < 0) or not torch.isclose(
        bin_probabilities.sum(), bin_probabilities.new_tensor(1.0), atol=1e-5, rtol=1e-5
    ):
        raise ValueError("bin_probabilities must be nonnegative and sum to one")

    bin_count = bin_probabilities.numel()
    difficulty = bin_count * bin_probabilities[bin_ids]
    normalized = torch.empty_like(raw_advantages)
    for mask in (difficulty > 1.0, difficulty <= 1.0):
        if not torch.any(mask):
            continue
        values = raw_advantages[mask]
        std = values.std(unbiased=False)
        normalized[mask] = (values - values.mean()) / (std + eps)
    return normalized, difficulty


def star_fragment_pool(
    raw_advantages: Tensor,
    bin_ids: Tensor,
    bin_probabilities: Tensor,
    env_ids: Tensor,
    fragment_ids: Tensor,
    top_fraction: float = 0.05,
) -> tuple[Tensor, Tensor]:
    """Select STAR's top bin-fragment pairs and return pool indices/probabilities.

    A fragment is identified by `(env_id, fragment_id)`; callers must ensure
    fragment IDs do not merge disjoint episodes within an environment.
    """
    vectors = (bin_ids, env_ids, fragment_ids)
    if raw_advantages.ndim != 1 or any(x.shape != raw_advantages.shape for x in vectors):
        raise ValueError("advantages and identifiers must be equal-length vectors")
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    if bin_probabilities.ndim != 1 or bin_probabilities.numel() == 0:
        raise ValueError("bin_probabilities must be a nonempty vector")
    n_bins = bin_probabilities.numel()
    if torch.any(bin_ids < 0) or torch.any(bin_ids >= n_bins):
        raise ValueError("bin_ids contain an out-of-range bin")
    if torch.any(bin_probabilities < 0) or not torch.isclose(
        bin_probabilities.sum(), bin_probabilities.new_tensor(1.0), atol=1e-5, rtol=1e-5
    ):
        raise ValueError("bin_probabilities must be nonnegative and sum to one")

    difficulty = n_bins * bin_probabilities[bin_ids]
    high = difficulty > 1.0
    if not torch.any(high):
        empty = torch.empty(0, dtype=torch.long, device=raw_advantages.device)
        return empty, raw_advantages.new_empty(0)

    # Score each valid (high-difficulty bin, rollout fragment) pair by mean raw advantage.
    pair_scores: dict[tuple[int, int, int], list[float]] = {}
    for i in torch.nonzero(high, as_tuple=False).flatten().tolist():
        key = (int(bin_ids[i]), int(env_ids[i]), int(fragment_ids[i]))
        pair_scores.setdefault(key, []).append(float(raw_advantages[i]))

    selected_fragments: set[tuple[int, int]] = set()
    for b in torch.unique(bin_ids[high]).tolist():
        candidates = [(key, sum(vals) / len(vals)) for key, vals in pair_scores.items() if key[0] == int(b)]
        candidates.sort(key=lambda item: item[1], reverse=True)
        keep = max(math.ceil(top_fraction * len(candidates)), 1)
        selected_fragments.update((key[1], key[2]) for key, _ in candidates[:keep])

    selected_mask = torch.tensor(
        [(int(e), int(f)) in selected_fragments for e, f in zip(env_ids.tolist(), fragment_ids.tolist())],
        dtype=torch.bool,
        device=raw_advantages.device,
    )
    indices = torch.nonzero(selected_mask, as_tuple=False).flatten()
    if indices.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=raw_advantages.device)
        return empty, raw_advantages.new_empty(0)

    # Eq. 29: each selected transition receives its fragment's mean difficulty weight.
    fragment_weight: dict[tuple[int, int], float] = {}
    for e, f in selected_fragments:
        mask = (env_ids == e) & (fragment_ids == f)
        fragment_weight[(e, f)] = float(difficulty[mask].mean())
    transition_weights = torch.tensor(
        [fragment_weight[(int(env_ids[i]), int(fragment_ids[i]))] for i in indices.tolist()],
        device=raw_advantages.device,
        dtype=raw_advantages.dtype,
    )
    probabilities = transition_weights / transition_weights.sum()
    return indices, probabilities


def star_mixed_minibatch_indices(
    pool_indices: Tensor,
    pool_probabilities: Tensor,
    buffer_size: int,
    minibatch_size: int,
    resample_ratio: float = 0.25,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw STAR's weighted high-value transitions plus ordinary buffer samples."""
    if buffer_size < 1 or minibatch_size < 1 or minibatch_size > buffer_size:
        raise ValueError("require 1 <= minibatch_size <= buffer_size")
    if not 0.0 <= resample_ratio <= 1.0:
        raise ValueError("resample_ratio must be in [0, 1]")
    if pool_indices.numel() == 0:
        return torch.randperm(buffer_size, device=pool_indices.device, generator=generator)[:minibatch_size]
    if pool_indices.shape != pool_probabilities.shape:
        raise ValueError("pool indices and probabilities must have identical shapes")
    n_star = min(minibatch_size, max(math.floor(resample_ratio * minibatch_size), 1))
    n_regular = minibatch_size - n_star
    weighted = pool_indices[torch.multinomial(pool_probabilities, n_star, replacement=True, generator=generator)]
    if n_regular:
        ordinary = torch.randperm(buffer_size, device=pool_indices.device, generator=generator)[:n_regular]
        result = torch.cat((weighted, ordinary))
    else:
        result = weighted
    return result[torch.randperm(result.numel(), device=result.device, generator=generator)]
