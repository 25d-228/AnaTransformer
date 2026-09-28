"""One tiny all-pairs identity, gradient, precision, and target-mask check."""

import importlib.util
import sys
from pathlib import Path

import torch

# Task-local module; no change to the package or to other experiments' trainers.
_spec = importlib.util.spec_from_file_location(
    "power_consistency_trainer", Path(__file__).with_name("trainer.py")
)
_trainer = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _trainer
_spec.loader.exec_module(_trainer)


def test_root_power_consistency_is_exact_symmetric_differentiable_and_masked():
    loss_function = _trainer.power_consistency
    floor_mass = _trainer.PROBABILITY_FLOOR_MASS
    logits1 = torch.tensor(
        [
            [[1.0, -0.5, 0.2, 2.0], [0.0, 1.0, -1.0, 0.5]],
            [[-0.2, 0.5, 1.5, -1.0], [4.0, -3.0, 0.5, 1.0]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    logits2 = torch.tensor(
        [
            [[0.2, 0.5, -0.7, 1.0], [-1.0, 0.2, 2.0, 0.5]],
            [[0.6, 0.1, -0.4, 1.0], [-2.0, 3.0, -1.0, 0.5]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    labels = torch.tensor([[3, -100], [2, 1]])
    loss = loss_function(logits1, logits2, labels)
    assert loss.dtype == torch.float64 and torch.isfinite(loss) and loss > 0
    root1 = ((1 - floor_mass) * logits1.softmax(-1) + floor_mass / 4).sqrt()
    root2 = ((1 - floor_mass) * logits2.softmax(-1) + floor_mass / 4).sqrt()
    explicit_per_token = torch.zeros_like(labels, dtype=torch.float64)
    for i in range(4):
        for j in range(i + 1, 4):
            defect = root1[..., i] + root2[..., j] - root1[..., j] - root2[..., i]
            explicit_per_token = explicit_per_token + defect.square() / 4
    expected = explicit_per_token[labels.ne(-100)].mean()
    torch.testing.assert_close(loss, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(loss_function(logits2, logits1, labels), loss, rtol=0, atol=0)
    assert loss_function(logits1, logits1, labels).item() == 0

    gradients = torch.autograd.grad(loss, (logits1, logits2))
    assert all(torch.isfinite(gradient).all() and gradient.norm() > 0 for gradient in gradients)
    assert all(torch.count_nonzero(gradient[0, 1]) == 0 for gradient in gradients)
    changed1, changed2 = logits1.detach().clone(), logits2.detach().clone()
    changed1[0, 1] = torch.tensor([500.0, -500.0, 50.0, -50.0])
    changed2[0, 1] = torch.tensor([-500.0, 500.0, -50.0, 50.0])
    torch.testing.assert_close(
        loss_function(changed1, changed2, labels), loss.detach(), rtol=0, atol=0
    )

    all_padding = torch.full_like(labels, -100)
    assert loss_function(logits1, logits2, all_padding).item() == 0
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        low1 = torch.tensor([[500.0, -500.0, 0.0, 1.0]], dtype=dtype, requires_grad=True)
        low2 = torch.tensor([[-1.0, 0.5, 1.0, -2.0]], dtype=dtype, requires_grad=True)
        low_loss = loss_function(low1, low2, torch.tensor([0]))
        assert low_loss.dtype == torch.float32 and torch.isfinite(low_loss)
        low_gradients = torch.autograd.grad(low_loss, (low1, low2))
        assert all(torch.isfinite(gradient).all() for gradient in low_gradients)
