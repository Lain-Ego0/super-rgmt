"""Paper-derived Extreme-RGMT Stage-I actor-critic components.

The model is deliberately independent of a simulator package.  The only
runtime dependency is PyTorch; ``register_with_motion_tracking_rl`` connects it
to the optional PMT/RSL-RL registry used by the training launcher.
"""
from __future__ import annotations

import math
import copy
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.distributions import Normal


class MLP(nn.Sequential):
    def __init__(self, input_dim: int, dims: Sequence[int], activation: str = "elu") -> None:
        layers: list[nn.Module] = []
        prev = input_dim
        for width in dims:
            layers.extend((nn.Linear(prev, int(width)), _activation(activation)))
            prev = int(width)
        super().__init__(*layers)


def _activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "elu":
        return nn.ELU()
    if key == "relu":
        return nn.ReLU()
    if key == "tanh":
        return nn.Tanh()
    if key == "gelu":
        return nn.GELU()
    raise ValueError(f"unsupported activation {name!r}")


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, width: int, max_length: int) -> None:
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10_000.0) / width))
        encoding = torch.zeros(max_length, width)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] > self.encoding.shape[0]:
            raise ValueError(f"sequence length {x.shape[1]} exceeds positional capacity {self.encoding.shape[0]}")
        return x + self.encoding[: x.shape[1]].to(device=x.device, dtype=x.dtype).unsqueeze(0)


class AttentionBlock(nn.Module):
    """Pre-norm attention + residual MLP block, shared by history and command paths."""

    def __init__(self, width: int, heads: int, mlp_ratio: int = 4, activation: str = "elu") -> None:
        super().__init__()
        if width % heads:
            raise ValueError(f"attention width {width} must be divisible by {heads} heads")
        self.norm_attention = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(width)
        self.mlp = MLP(width, (mlp_ratio * width, width), activation)
        self.output_norm = nn.LayerNorm(width)

    def self_attention(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        normalized = self.norm_attention(x)
        attended, _ = self.attention(normalized, normalized, normalized, attn_mask=mask, need_weights=False)
        x = x + attended
        return self.output_norm(x + self.mlp(self.norm_mlp(x)))

    def cross_attention(self, query: Tensor, key_value: Tensor) -> Tensor:
        attended, _ = self.attention(self.norm_attention(query), key_value, key_value, need_weights=False)
        query = query + attended
        return self.output_norm(query + self.mlp(self.norm_mlp(query)))


class FiniteScalarQuantizer(nn.Module):
    """Straight-through scalar quantization for an even-dimensional latent.

    The paper specifies two 32-D tokens but does not report their per-dimension
    level counts.  The default of five odd levels per scalar is therefore an
    explicit reproduction assumption, configurable by ``fsq_levels``.
    """

    def __init__(self, token_width: int = 32, levels: Sequence[int] = (5,) * 32) -> None:
        super().__init__()
        levels_t = torch.as_tensor(tuple(levels), dtype=torch.int64)
        if levels_t.ndim != 1 or levels_t.numel() != token_width:
            raise ValueError(f"fsq_levels must contain exactly {token_width} entries")
        if torch.any(levels_t < 3) or torch.any(levels_t.remainder(2) == 0):
            raise ValueError("FSQ currently requires odd level counts >= 3 for every scalar")
        self.token_width = token_width
        self.register_buffer("levels", levels_t, persistent=False)
        self.register_buffer("half_levels", (levels_t - 1).div(2, rounding_mode="floor"), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != 2 * self.token_width:
            raise ValueError(f"expected latent width {2 * self.token_width}, got {x.shape[-1]}")
        tokens = x.reshape(*x.shape[:-1], 2, self.token_width)
        half = self.half_levels.to(device=x.device, dtype=x.dtype)
        bounded = torch.tanh(tokens)
        scaled = bounded * half
        rounded = scaled + (scaled.round() - scaled).detach()
        quantized = rounded / half
        return quantized.reshape_as(x)


class ExtremeRGMTActorCritic(nn.Module):
    """Extreme-RGMT PPO policy for the paper's 29-DoF G1 observation contract.

    Expected groups follow the RGMT task convention:
      * ``policy``: [B, 93] = current proprioception (64) + previous action (29)
      * ``policy_history``: [B, 10, 93] = ten aligned state/action history frames
      * ``command_window``: [B, 21, 38] = local reference window
      * ``critic``: [B, C] = asymmetric privileged critic features

    Methods ``act``, ``act_inference``, ``evaluate``, ``get_actions_log_prob``,
    ``action_mean``, ``action_std`` and ``entropy`` match the common RSL-RL
    actor-critic interface.
    """

    is_recurrent = False
    proprio_dim = 64
    action_dim = 29
    reference_dim = 38
    latent_dim = 64

    def __init__(
        self,
        obs: Mapping[str, Tensor] | Any,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_hidden_dims: Sequence[int] = (1024, 1024, 512, 256),
        critic_hidden_dims: Sequence[int] = (1024, 1024, 512, 512),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        history_len: int = 10,
        command_len: int = 21,
        attention_heads: int = 4,
        attention_mlp_ratio: int = 4,
        fsq_levels: Sequence[int] = (5,) * 32,
        n_embd: int | None = None,
        n_heads: int | None = None,
        mlp_ratio: int | None = None,
        cmd_len: int | None = None,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        history_obs_normalization: bool = False,
        command_obs_normalization: bool = False,
        state_dependent_std: bool = False,
        noise_std_type: str = "scalar",
        log_std_bounds: tuple[float, float] = (-5.0, 2.0),
        min_std: float = 1e-6,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del actor_obs_normalization, critic_obs_normalization, history_obs_normalization, command_obs_normalization
        del min_std, validate_args
        if state_dependent_std:
            raise ValueError("Extreme-RGMT reproduction uses a state-independent Gaussian policy")
        if noise_std_type not in ("scalar", "log"):
            raise ValueError(f"unsupported noise_std_type {noise_std_type!r}")
        if n_embd is not None and n_embd != 64:
            raise ValueError("Extreme-RGMT encoder width is 64; n_embd override is incompatible")
        if n_heads is not None:
            attention_heads = n_heads
        if mlp_ratio is not None:
            attention_mlp_ratio = mlp_ratio
        if cmd_len is not None:
            command_len = cmd_len
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported ExtremeRGMTActorCritic options: {unknown}")
        if num_actions != self.action_dim:
            raise ValueError(f"Extreme-RGMT G1 policy requires 29 actions, got {num_actions}")
        if history_len < 1 or command_len < 1:
            raise ValueError("history_len and command_len must be positive")
        self.obs_groups = obs_groups
        self.pace_reference_policy: ExtremeRGMTActorCritic | None = None
        self.pace_role_key = "extreme_rgmt_role"
        self.pace_aux_loss_name = "pace_consolidation"
        self.pace_consolidation_weight = 0.3
        self.history_len = int(history_len)
        self.command_len = int(command_len)
        self.num_actions = int(num_actions)

        self.state_encoder = MLP(self.proprio_dim, (64, 128, 64), activation)
        self.action_encoder = MLP(self.action_dim, (29, 64, 64), activation)
        self.command_encoder = MLP(self.reference_dim, (38, 128, 64), activation)
        self.state_norm = nn.LayerNorm(64)
        self.action_norm = nn.LayerNorm(64)
        self.command_norm = nn.LayerNorm(64)

        self.history_position = SinusoidalPositionEncoding(64, max_length=2 * self.history_len)
        self.history_block = AttentionBlock(64, attention_heads, attention_mlp_ratio, activation)
        causal_mask = torch.triu(torch.ones(2 * self.history_len, 2 * self.history_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        self.command_position = SinusoidalPositionEncoding(64, max_length=self.command_len)
        self.command_query = nn.Sequential(nn.Linear(64, 64), _activation(activation))
        self.command_block = AttentionBlock(64, attention_heads, attention_mlp_ratio, activation)
        self.quantizer = FiniteScalarQuantizer(token_width=32, levels=fsq_levels)

        self.actor = MLP(self.proprio_dim + self.action_dim + self.latent_dim, actor_hidden_dims, activation)
        self.action_mean_head = nn.Linear(int(actor_hidden_dims[-1]), self.num_actions)
        self.critic = MLP(self._group_dim(obs, obs_groups, "critic"), critic_hidden_dims, activation)
        self.value_head = nn.Linear(int(critic_hidden_dims[-1]), 1)
        initial_log_std = math.log(float(init_noise_std))
        self.log_std = nn.Parameter(torch.full((self.num_actions,), initial_log_std))
        self.log_std_bounds = tuple(float(x) for x in log_std_bounds)
        self.distribution: Normal | None = None

    @staticmethod
    def _group_dim(obs: Mapping[str, Tensor] | Any, obs_groups: dict[str, list[str]], name: str) -> int:
        keys = obs_groups.get(name, [])
        if not keys:
            raise ValueError(f"obs_groups[{name!r}] must name at least one observation group")
        return sum(int(obs[key].shape[-1]) for key in keys)

    def _group(self, obs: Mapping[str, Tensor] | Any, name: str) -> Tensor:
        keys = self.obs_groups.get(name, [])
        if not keys:
            raise KeyError(f"obs_groups[{name!r}] is empty")
        tensors = [obs[key] for key in keys]
        if any(not isinstance(value, Tensor) for value in tensors):
            raise TypeError(f"all observations in group {name!r} must be torch tensors")
        return tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=-1)

    def encode_history(self, history: Tensor) -> Tensor:
        if history.ndim == 2:
            if history.shape[-1] != self.history_len * (self.proprio_dim + self.action_dim):
                raise ValueError(f"flat history must have {self.history_len * 93} features")
            history = history.reshape(history.shape[0], self.history_len, self.proprio_dim + self.action_dim)
        if history.ndim != 3 or history.shape[1:] != (self.history_len, self.proprio_dim + self.action_dim):
            raise ValueError(f"history must have shape [batch,{self.history_len},93], got {tuple(history.shape)}")
        prop, action = history[..., : self.proprio_dim], history[..., self.proprio_dim :]
        state_tokens = self.state_norm(self.state_encoder(prop))
        action_tokens = self.action_norm(self.action_encoder(action))
        # Paper Eq. (6): a_{t-H-1}, o_{t-H}, ..., a_{t-1}, o_t.
        tokens = torch.stack((action_tokens, state_tokens), dim=2).flatten(1, 2)
        tokens = self.history_position(tokens)
        mask = self.causal_mask[: tokens.shape[1], : tokens.shape[1]].to(device=tokens.device)
        return self.history_block.self_attention(tokens, mask=mask)[:, -1]

    def encode_command(self, history_latent: Tensor, command_window: Tensor) -> Tensor:
        if command_window.ndim == 2:
            if command_window.shape[-1] != self.command_len * self.reference_dim:
                raise ValueError(f"flat command window must have {self.command_len * self.reference_dim} features")
            command_window = command_window.reshape(command_window.shape[0], self.command_len, self.reference_dim)
        if command_window.ndim != 3 or command_window.shape[1:] != (self.command_len, self.reference_dim):
            raise ValueError(
                f"command window must have shape [batch,{self.command_len},{self.reference_dim}], "
                f"got {tuple(command_window.shape)}"
            )
        command_tokens = self.command_norm(self.command_encoder(command_window))
        command_tokens = self.command_position(command_tokens)
        query = self.command_query(history_latent).unsqueeze(1)
        attended = self.command_block.cross_attention(query, command_tokens).squeeze(1)
        return self.quantizer(attended)

    def _update_distribution(self, obs: Mapping[str, Tensor] | Any) -> None:
        policy = self._group(obs, "policy")
        if policy.shape[-1] != self.proprio_dim + self.action_dim:
            raise ValueError(f"policy observation must be 93-D (64 proprio + 29 action), got {policy.shape[-1]}")
        history = self._group(obs, "policy_history")
        command = self._group(obs, "command_window")
        h = self.encode_history(history)
        u = self.encode_command(h, command)
        actor_input = torch.cat((policy[..., : self.proprio_dim], policy[..., self.proprio_dim :], u), dim=-1)
        mean = self.action_mean_head(self.actor(actor_input))
        std = self.log_std.clamp(min=self.log_std_bounds[0], max=self.log_std_bounds[1]).exp().expand_as(mean)
        self.distribution = Normal(mean, std)

    def forward(self) -> None:
        raise NotImplementedError("call act(), act_inference(), or evaluate()")

    def reset(self, dones: Tensor | None = None) -> None:
        del dones

    def update_normalization(self, obs: Mapping[str, Tensor] | Any) -> None:
        """Match PMT PPO's policy hook; this policy deliberately uses no running normalizers."""
        del obs

    @property
    def action_mean(self) -> Tensor:
        if self.distribution is None:
            raise RuntimeError("action distribution is not initialized; call act() first")
        return self.distribution.mean

    @property
    def action_std(self) -> Tensor:
        if self.distribution is None:
            raise RuntimeError("action distribution is not initialized; call act() first")
        return self.distribution.stddev

    @property
    def entropy(self) -> Tensor:
        if self.distribution is None:
            raise RuntimeError("action distribution is not initialized; call act() first")
        return self.distribution.entropy().sum(dim=-1)

    def act(self, obs: Mapping[str, Tensor] | Any, **kwargs: Any) -> Tensor:
        del kwargs
        self._update_distribution(obs)
        assert self.distribution is not None
        return self.distribution.sample()

    def act_inference(self, obs: Mapping[str, Tensor] | Any) -> Tensor:
        self._update_distribution(obs)
        assert self.distribution is not None
        return self.distribution.mean

    def set_pace_consolidation_weight(self, weight: float) -> None:
        """Set the current Eq. (19) coefficient used by PPO's aux-loss hook."""
        if not 0.0 <= weight <= 1.0:
            raise ValueError("PACE consolidation weight must be in [0, 1]")
        self.pace_consolidation_weight = float(weight)

    def initialize_pace_reference(self) -> None:
        """Freeze a Stage-I snapshot as PACE's consolidation target.

        Call immediately after loading the Stage-I checkpoint and before Stage-II
        optimization starts. The copied policy stays in eval mode and is excluded
        from gradients; only its action means supervise mastered-role samples.
        """
        reference = copy.deepcopy(self)
        reference.pace_reference_policy = None
        reference.eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        self.pace_reference_policy = reference

    def compute_sonic_aux_losses(self, obs: Mapping[str, Tensor] | Any) -> dict[str, Any] | None:
        """Expose PACE action alignment through PMT PPO's auxiliary-loss hook."""
        consolidation_obs = getattr(self, "_pace_consolidation_obs", None)
        role = obs.get(self.pace_role_key) if hasattr(obs, "get") else None
        if consolidation_obs is not None:
            # STAR's generator keeps mastered data out of every PPO loss and
            # supplies it separately for the PACE action-MSE objective only.
            current_mean = self.act_inference(consolidation_obs)
            if self.pace_reference_policy is None:
                raise RuntimeError("initialize_pace_reference() before PACE consolidation updates")
            with torch.no_grad():
                reference_mean = self.pace_reference_policy.act_inference(consolidation_obs)
            loss = (current_mean - reference_mean).square().mean()
        elif role is not None:
            # Retain the mixed-batch path for direct objective use and diagnostics.
            role = role.reshape(-1).to(dtype=torch.long)
            consolidation = role == 1
            if not torch.any(consolidation):
                zero = self.action_mean.sum() * 0.0
                return {
                    "aux_losses": {self.pace_aux_loss_name: zero},
                    "aux_loss_coef": {self.pace_aux_loss_name: self.pace_consolidation_weight},
                }
            if self.pace_reference_policy is None:
                raise RuntimeError("initialize_pace_reference() before PACE consolidation updates")
            current_mean = self.action_mean
            if current_mean.shape[0] != role.numel():
                raise ValueError("PACE role labels must align with the PPO minibatch")
            with torch.no_grad():
                reference_mean = self.pace_reference_policy.act_inference(obs)
            loss = (current_mean[consolidation] - reference_mean[consolidation]).square().mean()
        else:
            return None
        return {
            "aux_losses": {self.pace_aux_loss_name: loss},
            "aux_loss_coef": {self.pace_aux_loss_name: self.pace_consolidation_weight},
        }

    def evaluate(self, obs: Mapping[str, Tensor] | Any, **kwargs: Any) -> Tensor:
        del kwargs
        return self.value_head(self.critic(self._group(obs, "critic")))

    def get_actions_log_prob(self, actions: Tensor) -> Tensor:
        if self.distribution is None:
            raise RuntimeError("action distribution is not initialized; call act() first")
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "policy_class": type(self).__name__,
            "history_len": self.history_len,
            "command_len": self.command_len,
            "proprio_dim": self.proprio_dim,
            "action_dim": self.action_dim,
            "reference_dim": self.reference_dim,
            "fsq_levels": self.quantizer.levels.detach().cpu().tolist(),
        }


def register_with_motion_tracking_rl() -> None:
    """Register this network in PMT's class-name dispatch without patching PMT."""
    try:
        from motion_tracking_rl.registry import register_network
    except ImportError as exc:  # pragma: no cover - exercised by the training launcher
        raise RuntimeError("install the pinned motion_tracking_rl backend before registration") from exc
    register_network("ExtremeRGMTActorCritic", compat_name="transformer")(ExtremeRGMTActorCritic)
