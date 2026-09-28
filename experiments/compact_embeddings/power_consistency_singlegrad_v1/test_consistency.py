"""One focused check for the one-gradient-view estimator and forward modes."""

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from trainer import CONSISTENCY_WEIGHT, power_consistency, single_gradient_loss


def test_single_gradient_estimator_and_reference_forward():
    assert CONSISTENCY_WEIGHT == 2.0
    labels = torch.tensor([[2, -100]])
    initial = torch.tensor([[[0.4, -0.5, 0.2, 1.0], [0.2, 0.5, -0.7, 0.3]]], dtype=torch.float64)
    first_mask = torch.tensor([[[0.0, 2.0, 2.0, 2.0], [2.0, 0.0, 2.0, 2.0]]])
    second_mask = torch.tensor([[[2.0, 2.0, 0.0, 2.0], [0.0, 2.0, 2.0, 2.0]]])

    def ce(logits):
        return F.cross_entropy(logits.reshape(-1, 4), labels.reshape(-1), label_smoothing=0.1)

    def gradient(which):
        weights = initial.clone().requires_grad_()
        left, right = weights * first_mask, weights * second_mask
        if which == "both":
            loss = 0.5 * (ce(left) + ce(right)) + power_consistency(left, right, labels)
        elif which == "left":
            loss = ce(left) + 2 * power_consistency(left, right.detach(), labels)
        else:
            loss = ce(right) + 2 * power_consistency(right, left.detach(), labels)
        return torch.autograd.grad(loss, weights)[0]

    full = gradient("both")
    averaged = 0.5 * (gradient("left") + gradient("right"))
    torch.testing.assert_close(averaged, full, rtol=1e-12, atol=1e-12)
    assert torch.isfinite(full).all() and full.norm() > 0
    assert torch.count_nonzero(full[:, 1]) == 0

    class TinyDropoutModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = nn.Parameter(initial.clone())
            self.dropout = nn.Dropout(0.5)
            self.calls = []

        def forward(self, source_ids, source_mask, target_labels):
            logits = self.dropout(self.weights)
            self.calls.append((self.training, torch.is_grad_enabled(), logits.requires_grad))
            return ce(logits), logits

    torch.manual_seed(123)
    model = TinyDropoutModel().train()
    batch = SimpleNamespace(source_ids=None, source_mask=None, labels=labels)
    loss, supervised, consistency = single_gradient_loss(model, batch)
    assert model.calls == [(True, False, False), (True, True, True)]
    torch.testing.assert_close(loss, supervised + 2 * consistency)
    loss.backward()
    assert torch.isfinite(model.weights.grad).all() and model.weights.grad.norm() > 0
    assert torch.count_nonzero(model.weights.grad[:, 1]) == 0

    model.eval()
    try:
        single_gradient_loss(model, batch)
    except ValueError:
        pass
    else:
        raise AssertionError("An evaluation-mode reference must be rejected")
