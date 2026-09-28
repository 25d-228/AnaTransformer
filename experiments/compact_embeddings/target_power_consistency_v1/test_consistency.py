"""One focused check of retained all-pairs and added target/rest consistency."""

import torch
from trainer import PROBABILITY_FLOOR_MASS, consistency_loss_config, power_consistency


def test_full_and_target_four_term_consistency():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    labels = torch.tensor([[0, 2, -100], [4, 1, 3]])
    first = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    second = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    p, q = first.softmax(-1), second.softmax(-1)
    epsilon = PROBABILITY_FLOOR_MASS
    root_p = ((1 - epsilon) * p + epsilon / 5).sqrt()
    root_q = ((1 - epsilon) * q + epsilon / 5).sqrt()
    full_terms, target_terms = [], []
    for row in range(2):
        for token in range(3):
            target = labels[row, token].item()
            if target == -100:
                continue
            full_terms.append(
                sum(
                    (
                        root_p[row, token, i]
                        + root_q[row, token, j]
                        - root_p[row, token, j]
                        - root_q[row, token, i]
                    ).square()
                    for i in range(5)
                    for j in range(i + 1, 5)
                )
                / 5
            )
            a = (1 - epsilon) * p[row, token, target] + epsilon / 2
            c = (1 - epsilon) * q[row, token, target] + epsilon / 2
            b, d = 1 - a, 1 - c
            assert min(a.item(), b.item(), c.item(), d.item()) > 0
            target_terms.append(0.5 * (a.sqrt() + d.sqrt() - b.sqrt() - c.sqrt()).square())
    full = torch.stack(full_terms).mean()
    target = torch.stack(target_terms).mean()
    expected = full + target
    actual = power_consistency(first, second, labels)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert actual > full
    fast = torch.autograd.grad(actual, (first, second), retain_graph=True)
    explicit = torch.autograd.grad(expected, (first, second))
    for left, right in zip(fast, explicit, strict=True):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-10)
        assert torch.isfinite(left).all() and left.abs().sum() > 0
        assert torch.count_nonzero(left[labels == -100]) == 0
    torch.testing.assert_close(
        power_consistency(second, first, labels), actual, atol=1e-12, rtol=1e-12
    )
    assert power_consistency(first, first, labels).item() == 0
    empty = power_consistency(first, second, torch.full_like(labels, -100))
    assert empty.item() == 0
    for gradient in torch.autograd.grad(empty, (first, second)):
        assert torch.count_nonzero(gradient) == 0
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        left = (torch.randn(2, 3, 5) * 300).to(dtype).requires_grad_()
        right = (torch.randn(2, 3, 5) * 300).to(dtype).requires_grad_()
        result = power_consistency(left, right, labels)
        assert torch.isfinite(result) and 0 <= result <= 4.00001
        for gradient in torch.autograd.grad(result, (left, right)):
            assert torch.isfinite(gradient).all()
    config = consistency_loss_config()
    assert config["power"] == 0.5 and config["both_views_receive_gradients"]
    assert config["weight"] == config["target_weight"] == 1.0
