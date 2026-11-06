"""Isaac Lab action terms required by Extreme-RGMT's residual action equation."""
from __future__ import annotations

import torch
from torch import Tensor

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction


def compose_reference_residual(reference: Tensor, residual: Tensor, scale: float | Tensor) -> Tensor:
    """Compose position targets using the paper's per-environment reference pose."""
    if reference.shape != residual.shape:
        raise ValueError(f"reference and residual shapes differ: {reference.shape} vs {residual.shape}")
    return reference + residual * scale


class ReferenceResidualJointPositionAction(JointPositionAction):
    """Apply ``q_target = q_ref + scale * action`` for each environment.

    The stock action term's offset represents a static/default pose. The motion
    reference varies by environment and control step, so compose it directly
    from the live command rather than relying on mutable action offsets.
    """

    def process_actions(self, actions: Tensor):
        self._raw_actions[:] = actions
        motion = self._env.command_manager.get_term("motion")
        self._processed_actions = compose_reference_residual(
            motion.joint_pos, self._raw_actions, self._scale
        )
        if self.cfg.clip is not None:
            self._processed_actions.clamp_(min=self._clip[:, :, 0], max=self._clip[:, :, 1])
