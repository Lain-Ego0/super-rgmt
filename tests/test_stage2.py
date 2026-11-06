import torch

from extreme_rgmt.stage2 import (
    PACEProgress,
    Stage2RolloutMetadata,
    annotate_stage2_observations,
    allocate_pace_environments,
    difficulty_conditioned_advantages,
    pace_objective,
    pace_role_objective,
    rollout_fragment_ids,
    star_fragment_pool,
    star_mixed_minibatch_indices,
)


def test_pace_progress_ema_matches_paper_schedule():
    state = PACEProgress(rho_ref=0.6, beta=0.0, lambda_base=0.3, kappa=5.0)
    assert state.update(valid_acquisition=50, valid_consolidation=50) == 0.3
    assert state.update(valid_acquisition=80, valid_consolidation=20) == 1.0


def test_pace_objective_regularizes_only_action_disagreement():
    acquisition = torch.tensor(2.0)
    current = torch.tensor([[1.0, 3.0]])
    base = torch.tensor([[0.0, 1.0]])
    result = pace_objective(acquisition, current, base, weight=0.5)
    torch.testing.assert_close(result, torch.tensor(3.25))


def test_star_normalizes_high_and_remaining_advantages_separately():
    raw = torch.tensor([1.0, 3.0, 10.0, 14.0])
    bins = torch.tensor([0, 0, 1, 2])
    probs = torch.tensor([0.5, 0.25, 0.25])
    normalized, weights = difficulty_conditioned_advantages(raw, bins, probs)
    torch.testing.assert_close(weights, torch.tensor([1.5, 1.5, 0.75, 0.75]))
    torch.testing.assert_close(normalized[:2], torch.tensor([-1.0, 1.0]))
    torch.testing.assert_close(normalized[2:], torch.tensor([-1.0, 1.0]))


def test_star_selects_top_fragment_per_high_difficulty_bin_and_mixes_batch():
    raw = torch.tensor([1.0, 3.0, 8.0, 2.0, 0.0, 100.0])
    bins = torch.tensor([0, 0, 0, 0, 1, 2])
    envs = torch.tensor([0, 0, 0, 0, 0, 1])
    fragments = torch.tensor([0, 0, 1, 1, 1, 0])
    probs = torch.tensor([0.5, 0.25, 0.25])
    pool, pool_probs = star_fragment_pool(raw, bins, probs, envs, fragments, top_fraction=0.5)
    assert pool.tolist() == [2, 3, 4]
    torch.testing.assert_close(pool_probs, torch.full((3,), 1.0 / 3.0))
    batch = star_mixed_minibatch_indices(pool, pool_probs, buffer_size=6, minibatch_size=4, generator=torch.Generator().manual_seed(3))
    assert batch.numel() == 4
    assert torch.all((batch >= 0) & (batch < 6))
    assert sum(int(i in pool.tolist()) for i in batch) >= 1


def test_star_falls_back_to_uniform_when_no_high_difficulty_pool():
    empty = torch.empty(0, dtype=torch.long)
    batch = star_mixed_minibatch_indices(empty, torch.empty(0), 5, 3, generator=torch.Generator().manual_seed(2))
    assert batch.numel() == 3
    assert batch.unique().numel() == 3


def test_pace_role_allocation_uses_paper_fraction_and_keeps_both_roles():
    assert allocate_pace_environments(4096, 0.8) == (3277, 819)
    assert allocate_pace_environments(2, 0.01) == (1, 1)
    assert allocate_pace_environments(1, 0.8) == (1, 0)


def test_pace_schedule_state_round_trips_for_checkpoint_resume():
    state = PACEProgress(beta=0.0)
    state.update(80, 20)
    restored = PACEProgress()
    restored.load_state_dict(state.state_dict())
    assert restored.state_dict() == state.state_dict()


def test_rollout_fragment_ids_split_at_done_per_environment():
    dones = torch.tensor([[False, False], [True, False], [False, True], [False, False]])
    env_ids, fragment_ids = rollout_fragment_ids(dones)
    assert env_ids.tolist() == [0, 1] * 4
    assert fragment_ids.tolist() == [0, 0, 0, 0, 1, 0, 1, 1]


def test_stage2_rollout_metadata_tracks_roles_bins_and_fragments():
    metadata = Stage2RolloutMetadata(num_steps=2, num_envs=2, device="cpu")
    roles = torch.tensor([0, 1])
    metadata.append(torch.tensor([2, 4]), roles, torch.tensor([False, True]))
    metadata.append(torch.tensor([3, 1]), roles, torch.tensor([True, False]))
    flat = metadata.flatten()
    assert flat["bin_ids"].tolist() == [2, 4, 3, 1]
    assert flat["role_ids"].tolist() == [0, 1, 0, 1]
    assert flat["env_ids"].tolist() == [0, 1, 0, 1]
    assert flat["fragment_ids"].tolist() == [0, 0, 0, 1]


def test_stage2_rollout_metadata_rejects_role_changes_mid_rollout():
    metadata = Stage2RolloutMetadata(num_steps=2, num_envs=2, device="cpu")
    metadata.append(torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([False, False]))
    try:
        metadata.append(torch.tensor([0, 1]), torch.tensor([1, 0]), torch.tensor([False, False]))
    except ValueError as exc:
        assert "fixed" in str(exc)
    else:
        raise AssertionError("expected fixed-role validation")


def test_pace_role_objective_regularizes_mastered_role_only():
    acq_loss = torch.tensor(2.0)
    roles = torch.tensor([0, 1, 0, 1])
    current = torch.tensor([[100.0], [2.0], [-100.0], [4.0]])
    reference = torch.tensor([[0.0], [1.0], [0.0], [2.0]])
    total, consolidation = pace_role_objective(acq_loss, roles, current, reference, 0.5)
    torch.testing.assert_close(consolidation, torch.tensor(2.5))
    torch.testing.assert_close(total, torch.tensor(3.25))


def test_pace_role_objective_has_zero_consolidation_without_mastered_samples():
    actions = torch.ones(2, 3, requires_grad=True)
    total, consolidation = pace_role_objective(
        torch.tensor(1.0), torch.zeros(2, dtype=torch.long), actions,
        torch.zeros_like(actions), 0.5,
    )
    assert consolidation.item() == 0.0
    total.backward()
    torch.testing.assert_close(actions.grad, torch.zeros_like(actions))


def test_observation_annotation_keeps_stage2_labels_aligned_in_tensordict():
    pytest = __import__("pytest")
    tensordict = pytest.importorskip("tensordict")
    TensorDict = tensordict.TensorDict

    obs = TensorDict({"policy": torch.zeros(3, 4)}, batch_size=[3])
    annotated = annotate_stage2_observations(
        obs, role_ids=torch.tensor([0, 1, 0]), bin_ids=torch.tensor([4, 2, 1])
    )
    assert annotated is obs
    assert obs["extreme_rgmt_role"].squeeze(-1).tolist() == [0, 1, 0]
    assert obs["extreme_rgmt_bin"].squeeze(-1).tolist() == [4, 2, 1]


def test_star_pace_update_installs_generator_and_restores_storage():
    from extreme_rgmt.stage2 import star_pace_update

    class Batch:
        def __init__(self, value):
            self.value = value

        def __getitem__(self, indices):
            return Batch(self.value[indices])

        def flatten(self, start, end):
            return Batch(self.value.flatten(start, end))

    class Storage:
        training_type = "rl"
        num_transitions_per_env = 2
        num_envs = 2
        device = "cpu"

        def __init__(self):
            self.observations = Batch(torch.arange(4).reshape(2, 2, 1))
            self.actions = torch.zeros(2, 2, 1)
            self.values = torch.zeros(2, 2, 1)
            self.returns = torch.tensor([[[2.0], [3.0]], [[4.0], [5.0]]])
            self.advantages = torch.ones(2, 2, 1)
            self.actions_log_prob = torch.zeros(2, 2, 1)
            self.mu = torch.zeros(2, 2, 1)
            self.sigma = torch.ones(2, 2, 1)

        def mini_batch_generator(self, batches, epochs=8):
            yield from ()

    storage = Storage()
    original_method = storage.mini_batch_generator
    metadata = {
        "bin_ids": torch.tensor([0, 1, 0, 1]),
        "role_ids": torch.tensor([0, 1, 0, 1]),
        "env_ids": torch.tensor([0, 1, 0, 1]),
        "fragment_ids": torch.tensor([0, 0, 1, 1]),
    }
    with star_pace_update(storage, metadata, torch.tensor([0.75, 0.25]),
                          num_mini_batches=2, num_epochs=1) as stats:
        assert stats["acquisition_samples"] == 2
        assert torch.equal(storage.advantages.reshape(-1)[[1, 3]], torch.zeros(2))
        minibatches = list(storage.mini_batch_generator(2, 1))
        assert len(minibatches) == 2
        assert all(batch[0].value.shape[0] == 2 for batch in minibatches)
        for batch in minibatches:
            ids = batch[0].value.squeeze(-1)
            assert int((ids % 2 == 0).sum()) == 1  # acquisition role
            assert int((ids % 2 == 1).sum()) == 1  # consolidation role
    assert torch.equal(storage.advantages, torch.ones_like(storage.advantages))
    assert storage.mini_batch_generator.__func__ is original_method.__func__
