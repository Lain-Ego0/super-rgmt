import pytest
import torch

from extreme_rgmt.models import AttentionBlock, ExtremeRGMTActorCritic, FiniteScalarQuantizer


def _observations(batch_size=3):
    return {
        "policy": torch.randn(batch_size, 93),
        "policy_history": torch.randn(batch_size, 10, 93),
        "command_window": torch.randn(batch_size, 21, 38),
        "critic": torch.randn(batch_size, 210),
    }


def _model():
    return ExtremeRGMTActorCritic(
        _observations(1),
        {
            "policy": ["policy"],
            "policy_history": ["policy_history"],
            "command_window": ["command_window"],
            "critic": ["critic"],
        },
        num_actions=29,
        actor_hidden_dims=(128, 64),
        critic_hidden_dims=(128, 64),
    )


def test_actor_critic_ppo_interface_and_gradients():
    model = _model()
    obs = _observations()

    actions = model.act(obs)
    assert actions.shape == (3, 29)
    assert model.get_actions_log_prob(actions).shape == (3,)
    assert model.action_mean.shape == (3, 29)
    assert model.action_std.shape == (3, 29)
    assert model.entropy.shape == (3,)
    assert model.act_inference(obs).shape == (3, 29)
    assert model.evaluate(obs).shape == (3, 1)

    loss = model.action_mean.square().mean() + model.evaluate(obs).square().mean()
    loss.backward()
    assert model.action_mean_head.weight.grad is not None
    assert model.value_head.weight.grad is not None
    assert torch.isfinite(model.action_mean_head.weight.grad).all()


def test_fsq_has_two_32d_tokens_and_straight_through_gradient():
    quantizer = FiniteScalarQuantizer()
    x = torch.linspace(-0.5, 0.5, 64).reshape(1, 64).requires_grad_()
    y = quantizer(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.all(y.abs() <= 1.0)


def test_history_attention_is_causal():
    block = AttentionBlock(width=8, heads=2).eval()
    mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    x = torch.randn(2, 4, 8)
    changed_future = x.clone()
    changed_future[:, 2:] += torch.randn_like(changed_future[:, 2:]) * 3.0
    y1 = block.self_attention(x, mask)
    y2 = block.self_attention(changed_future, mask)
    assert torch.allclose(y1[:, :2], y2[:, :2], atol=1e-6)
    assert not torch.allclose(y1[:, -1], y2[:, -1])


def test_input_contract_rejects_wrong_action_dimension():
    with pytest.raises(ValueError, match="29 actions"):
        ExtremeRGMTActorCritic(_observations(1), {"critic": ["critic"]}, num_actions=28)


def test_pace_reference_auxiliary_loss_is_frozen_and_mastered_only():
    model = _model()
    model.initialize_pace_reference()
    obs = _observations(batch_size=3)
    obs["extreme_rgmt_role"] = torch.tensor([[0], [1], [1]])

    model.act(obs)
    result = model.compute_sonic_aux_losses(obs)
    loss = result["aux_losses"]["pace_consolidation"]
    assert loss.ndim == 0
    assert loss.item() == pytest.approx(0.0, abs=1e-8)
    assert all(not parameter.requires_grad for parameter in model.pace_reference_policy.parameters())

    # A changed actor output must produce gradients through the trainable model,
    # while the frozen reference remains unchanged.
    with torch.no_grad():
        model.action_mean_head.bias.add_(0.25)
    model.act(obs)
    result = model.compute_sonic_aux_losses(obs)
    loss = result["aux_losses"]["pace_consolidation"]
    loss.backward()
    assert model.action_mean_head.bias.grad is not None
    assert torch.isfinite(model.action_mean_head.bias.grad).all()
    assert all(parameter.grad is None for parameter in model.pace_reference_policy.parameters())


def test_pace_auxiliary_loss_is_optional_for_stage_one_observations():
    model = _model()
    assert model.compute_sonic_aux_losses(_observations()) is None


def test_pace_auxiliary_uses_separate_mastered_batch_with_acquisition_only_ppo_batch():
    model = _model()
    model.initialize_pace_reference()
    acquisition_obs = _observations(batch_size=2)
    acquisition_obs["extreme_rgmt_role"] = torch.zeros(2, 1, dtype=torch.long)
    mastered_obs = _observations(batch_size=3)
    mastered_obs["extreme_rgmt_role"] = torch.ones(3, 1, dtype=torch.long)

    with torch.no_grad():
        model.action_mean_head.bias.add_(0.2)
    model.act(acquisition_obs)
    model._pace_consolidation_obs = mastered_obs
    result = model.compute_sonic_aux_losses(acquisition_obs)
    loss = result["aux_losses"]["pace_consolidation"]
    assert loss.item() > 0.0
    loss.backward()
    assert model.action_mean_head.bias.grad is not None
    assert torch.isfinite(model.action_mean_head.bias.grad).all()
    assert all(parameter.grad is None for parameter in model.pace_reference_policy.parameters())
    model._pace_consolidation_obs = None
