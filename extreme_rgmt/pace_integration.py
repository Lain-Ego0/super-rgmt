"""Runtime adapters that connect the pure PACE/STAR logic to PMT RGMT.

This module is imported only after Isaac Lab has launched. It deliberately
uses duck typing so the paper-specific code stays outside the pinned PMT tree.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import Tensor

from .stage2 import (
    PACEProgress,
    Stage2RolloutMetadata,
    allocate_pace_environments,
    annotate_stage2_observations,
    star_pace_update,
)


class PaceMotionGroups:
    """Route a shared PMT motion command through PACE's two fixed env roles.

    Acquisition envs sample the challenging clips with PMT's temporal
    bin-adaptive sampler. Consolidation envs sample mastered clips uniformly.
    Motion IDs exposed by the command remain global IDs, while the adaptive
    sampler keeps local IDs for its challenging subset.
    """

    def __init__(
        self,
        command,
        *,
        num_envs: int,
        acquisition_fraction: float,
        mastered_motion_ids: list[int],
        challenging_motion_ids: list[int],
    ) -> None:
        total_motions = len(command.data_store.motions)
        mastered = torch.as_tensor(mastered_motion_ids, dtype=torch.long, device=command.device)
        challenging = torch.as_tensor(challenging_motion_ids, dtype=torch.long, device=command.device)
        if mastered.numel() == 0 or challenging.numel() == 0:
            raise ValueError("PACE requires nonempty mastered and challenging motion sets")
        all_ids = torch.cat((mastered, challenging))
        unique_count = int(torch.unique(all_ids).numel())
        if all_ids.numel() != total_motions or unique_count != total_motions:
            raise ValueError(
                "mastered/challenging IDs must partition every loaded motion exactly once "
                f"(mastered={mastered.numel()}, challenging={challenging.numel()}, "
                f"unique={unique_count}, loaded={total_motions})"
            )
        if int(all_ids.min()) != 0 or int(all_ids.max()) != total_motions - 1:
            raise ValueError("motion IDs must cover the range [0, num_motions)")
        if num_envs != command.num_envs:
            raise ValueError("num_envs must match the motion command environment count")

        acquisition_count, consolidation_count = allocate_pace_environments(
            num_envs, acquisition_fraction
        )
        if acquisition_count == 0 or consolidation_count == 0:
            raise ValueError("PACE requires at least one environment for each role")
        self.command = command
        self.mastered_ids = mastered
        self.challenging_ids = challenging
        self.num_envs = num_envs
        self.device = torch.device(command.device)
        self.role_ids = torch.ones(num_envs, dtype=torch.long, device=self.device)
        self.role_ids[:acquisition_count] = 0
        self.acquisition_count = acquisition_count
        self.consolidation_count = consolidation_count
        self.motion_lengths = command.data_store.motion_lengths.to(self.device)
        self.acquisition_inverse = torch.full(
            (total_motions,), -1, dtype=torch.long, device=self.device
        )
        self.acquisition_inverse[challenging] = torch.arange(
            challenging.numel(), device=self.device
        )
        self.bin_offsets: Tensor | None = None

        template = command.sampler
        if not hasattr(template, "failed_bin_count") or not hasattr(template, "motion_num_bins"):
            raise TypeError("PACE acquisition requires PMT's bin-adaptive motion sampler")
        cfg = command.cfg
        self.acquisition_sampler = type(template)(
            num_motions=int(challenging.numel()),
            motion_lengths=self.motion_lengths[challenging],
            device=self.device,
            motion_fps=float(cfg.motion_fps),
            bin_duration=float(cfg.bin_duration),
            beta=float(cfg.adaptive_beta),
            alpha=float(cfg.adaptive_alpha),
            uniform_ratio=float(cfg.adaptive_uniform_ratio),
            update_interval=int(cfg.adaptive_update_interval),
            kernel_size=int(cfg.adaptive_kernel_size),
            kernel_lambda=float(cfg.adaptive_kernel_lambda),
        )
        self.command.sampler = self.acquisition_sampler

        self._original_draw = command._draw_samples
        self._original_record = command._record_outcomes
        self._original_sampler = template
        command._draw_samples = lambda env_ids: self._draw_samples(command, env_ids)
        command._record_outcomes = lambda env_ids, terminated: self._record_outcomes(
            command, env_ids, terminated
        )

        # The initial PMT constructor has already reset the scene. Re-sample once
        # with PACE's fixed groups without counting that reset as an outcome.
        command._is_first_reset = True
        command._resample_command(torch.arange(num_envs, device=self.device))

    def close(self) -> None:
        self.command._draw_samples = self._original_draw
        self.command._record_outcomes = self._original_record
        self.command.sampler = self._original_sampler

    def _draw_samples(self, command, env_ids: Tensor):
        roles = self.role_ids[env_ids]
        motion_ids = torch.empty(env_ids.numel(), dtype=torch.long, device=self.device)
        frame_ids = torch.empty_like(motion_ids)
        acquisition = roles == 0
        if acquisition.any():
            local = self.acquisition_sampler.sample(int(acquisition.sum()))
            motion_ids[acquisition] = self.challenging_ids[local.motion_ids]
            frame_ids[acquisition] = local.frame_ids
        consolidation = ~acquisition
        if consolidation.any():
            local_ids = torch.randint(
                self.mastered_ids.numel(), (int(consolidation.sum()),), device=self.device
            )
            selected_motions = self.mastered_ids[local_ids]
            lengths = self.motion_lengths[selected_motions]
            frames = (torch.rand(selected_motions.numel(), device=self.device) * lengths).long()
            frame_ids[consolidation] = torch.minimum(frames, lengths - 1)
            motion_ids[consolidation] = selected_motions
        return SimpleNamespace(motion_ids=motion_ids, frame_ids=frame_ids)

    def _record_outcomes(self, command, env_ids: Tensor, terminated: Tensor) -> None:
        acquisition = self.role_ids[env_ids] == 0
        if acquisition.any():
            selected_envs = env_ids[acquisition]
            global_ids = command.start_motion_ids[selected_envs]
            local_ids = self.acquisition_inverse[global_ids]
            if torch.any(local_ids < 0):
                raise RuntimeError("acquisition environment sampled a non-challenging motion")
            self.acquisition_sampler.update(
                local_ids,
                command.start_frame_ids[selected_envs],
                terminated[acquisition],
            )

    def bin_probabilities(self) -> Tensor:
        """Joint adaptive probabilities over the flattened challenging bins."""
        sampler = self.acquisition_sampler
        motion_probs = sampler._get_motion_probs()
        per_motion: list[Tensor] = []
        for motion_id in range(sampler.num_motions):
            valid = sampler.valid_bins_mask[motion_id : motion_id + 1]
            pf = sampler._compute_pf_selected_smoothed(
                torch.tensor([motion_id], device=self.device), valid
            ).clone()
            pf = pf.clamp(min=1e-6)
            pf[~valid] = 0.0
            weighted = pf.pow(sampler.beta)
            weighted = weighted / (weighted.sum(dim=1, keepdim=True) + 1e-8)
            uniform = valid.float() / sampler.motion_num_bins[motion_id].clamp(min=1)
            conditional = weighted * (1.0 - sampler.uniform_ratio) + uniform * sampler.uniform_ratio
            conditional[~valid] = 0.0
            conditional = conditional / conditional.sum(dim=1, keepdim=True).clamp(min=1e-8)
            count = int(sampler.motion_num_bins[motion_id])
            per_motion.append(motion_probs[motion_id] * conditional[0, :count])
        probabilities = torch.cat(per_motion)
        return probabilities / probabilities.sum().clamp(min=1e-8)

    def transition_bin_ids(self) -> Tensor:
        """Map live global motion/frame references to STAR's flattened bin IDs."""
        sampler = self.acquisition_sampler
        bin_counts = sampler.motion_num_bins
        offsets = torch.cat((bin_counts.new_zeros(1), bin_counts.cumsum(0)[:-1]))
        self.bin_offsets = offsets
        command = self.command
        bins = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        acquisition = self.role_ids == 0
        local_ids = self.acquisition_inverse[command.motion_ids[acquisition]]
        if torch.any(local_ids < 0):
            raise RuntimeError("acquisition environment currently references a mastered motion")
        local_bins = torch.div(
            command.frame_ids[acquisition], sampler.bin_size, rounding_mode="floor"
        ).clamp(min=0)
        local_bins = torch.minimum(local_bins, bin_counts[local_ids] - 1)
        bins[acquisition] = offsets[local_ids] + local_bins
        return bins


class PacePPOAdapter:
    """Attach Stage-II metadata, STAR minibatches, and PACE loss to a PMT PPO."""

    def __init__(
        self,
        algorithm,
        motion_groups: PaceMotionGroups,
        *,
        num_steps: int,
        num_mini_batches: int,
        num_epochs: int,
        top_fraction: float = 0.05,
        resample_ratio: float = 0.25,
        progress: PACEProgress | None = None,
    ) -> None:
        self.algorithm = algorithm
        self.groups = motion_groups
        if not hasattr(algorithm.policy, "initialize_pace_reference") or not hasattr(
            algorithm.policy, "set_pace_consolidation_weight"
        ):
            raise TypeError("Stage-II policy must expose the Extreme-RGMT PACE reference hooks")
        algorithm.policy.initialize_pace_reference()
        self.num_steps = num_steps
        self.num_mini_batches = num_mini_batches
        self.num_epochs = num_epochs
        self.top_fraction = top_fraction
        self.resample_ratio = resample_ratio
        self.progress = progress or PACEProgress()
        self.metadata = Stage2RolloutMetadata(num_steps, motion_groups.num_envs, motion_groups.device)
        self.pending_bins: Tensor | None = None

        # RolloutStorage is allocated before this adapter is attached. Register
        # the sidecar leaves now so its TensorDict schema accepts annotated
        # observations on the first transition.
        storage = algorithm.storage
        storage.observations.set(
            "extreme_rgmt_role",
            torch.zeros(
                num_steps, motion_groups.num_envs, 1,
                dtype=torch.long, device=storage.device,
            ),
        )
        storage.observations.set(
            "extreme_rgmt_bin",
            torch.zeros(
                num_steps, motion_groups.num_envs, 1,
                dtype=torch.long, device=storage.device,
            ),
        )

        self._original_act = algorithm.act
        self._original_process = algorithm.process_env_step
        self._original_update = algorithm.update
        algorithm.act = self._act
        algorithm.process_env_step = self._process_env_step
        algorithm.update = self._update

    def _act(self, observations):
        roles = self.groups.role_ids
        bins = self.groups.transition_bin_ids()
        self.pending_bins = bins.clone()
        annotated = annotate_stage2_observations(
            observations, role_ids=roles, bin_ids=bins
        )
        return self._original_act(annotated)

    def _process_env_step(self, observations, rewards, dones, extras):
        if self.pending_bins is None:
            raise RuntimeError("PPO env step occurred without a matching action")
        self.metadata.append(self.pending_bins, self.groups.role_ids, dones)
        self.pending_bins = None
        return self._original_process(observations, rewards, dones, extras)

    def _update(self):
        metadata = self.metadata.flatten()
        probabilities = self.groups.bin_probabilities()
        valid = ~metadata["dones"].bool()
        roles = metadata["role_ids"]
        valid_acquisition = int((valid & (roles == 0)).sum())
        valid_consolidation = int((valid & (roles == 1)).sum())
        if valid_acquisition + valid_consolidation == 0:
            # Every transition terminated. Keep PACE well-defined for this rare
            # rollout by using the configured environment allocation as progress.
            valid_acquisition = self.groups.acquisition_count
            valid_consolidation = self.groups.consolidation_count
        weight = self.progress.update(valid_acquisition, valid_consolidation)
        with star_pace_update(
            self.algorithm.storage,
            metadata,
            probabilities,
            top_fraction=self.top_fraction,
            resample_ratio=self.resample_ratio,
            num_mini_batches=self.num_mini_batches,
            num_epochs=self.num_epochs,
            pace_policy=self.algorithm.policy,
        ) as star_stats:
            self.algorithm.policy.set_pace_consolidation_weight(weight)
            result = self._original_update()
        result.update({
            "pace/consolidation_weight": weight,
            "pace/acquisition_valid": float(valid_acquisition),
            "pace/consolidation_valid": float(valid_consolidation),
            "pace/star_pool_size": float(star_stats["star_pool_size"]),
        })
        self.metadata = Stage2RolloutMetadata(
            self.num_steps, self.groups.num_envs, self.groups.device
        )
        return result

    def close(self) -> None:
        self.algorithm.act = self._original_act
        self.algorithm.process_env_step = self._original_process
        self.algorithm.update = self._original_update
