import pytest
import torch

from extreme_rgmt.isaac_actions import compose_reference_residual


def test_reference_residual_targets_are_composed_per_environment():
    reference = torch.tensor([[0.5, 1.0], [-0.5, 0.25]])
    residual = torch.tensor([[0.1, -0.2], [-0.3, 0.4]])
    targets = compose_reference_residual(reference, residual, 0.5)
    expected = torch.tensor([[0.55, 0.9], [-0.65, 0.45]])
    torch.testing.assert_close(targets, expected)


def test_reference_residual_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shapes differ"):
        compose_reference_residual(torch.zeros(2, 3), torch.zeros(2, 2), 1.0)
