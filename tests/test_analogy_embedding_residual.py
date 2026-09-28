"""One focused linear-identity, power, numerical, count, and decoding check."""

import torch
import torch.nn.functional as F

from ana.config import ModelConfig
from ana.nn.embeddings.analogy_embedding import (
    LinearCompressedEmbedding,
    ResidualPowerAnalogyEmbedding,
    _ordered_log_completion,
)
from ana.nn.projection import SeparateQKV
from ana.registry import build_model, count_parameters, model_parameters


def test_residual_power_keeps_linear_start_and_learns_through_tied_table():
    name = "ana_embedding_residual_learned"
    large_shape = ModelConfig(
        vocab_size=10_000,
        d_model=128,
        n_heads=4,
        d_ff=232,
        n_encoder_layers=4,
        n_decoder_layers=4,
        dropout=0.3,
    )
    large = build_model(name, large_shape)
    assert count_parameters(large) == model_parameters(name, large_shape) == 2_248_544
    assert count_parameters(large) - model_parameters("embedding_linear", large_shape) == 32
    assert sum(isinstance(module, SeparateQKV) for module in large.modules()) == 12
    assert large.embedding.complement.shape == (32, 128)
    assert not large.embedding.complement.requires_grad
    del large

    # No extra random draws: the linear table/basis and all following RNG
    # state are identical. The complement is a fixed buffer, not a projector
    # paid for by extra trainable parameters.
    torch.manual_seed(42)
    linear = LinearCompressedEmbedding(64, 16, 0).double()
    linear_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(42)
    residual = ResidualPowerAnalogyEmbedding(64, 16, 0).double()
    assert torch.equal(linear_rng, torch.random.get_rng_state())
    assert torch.equal(residual.powers(), torch.ones(4, dtype=torch.float64))
    torch.testing.assert_close(residual.codes, linear.codes, rtol=0, atol=0)
    torch.testing.assert_close(residual.basis, linear.basis, rtol=0, atol=0)
    torch.testing.assert_close(
        residual.complement @ residual.complement.T,
        torch.eye(4, dtype=torch.float64),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        residual.basis @ residual.complement.T,
        torch.zeros(12, 4, dtype=torch.float64),
        rtol=0,
        atol=1e-6,
    )
    assert torch.count_nonzero(residual.residual_features(residual.codes)) == 0
    torch.testing.assert_close(residual.weight(), linear.weight(), rtol=0, atol=0)
    direction = torch.randn(64, 16, dtype=torch.float64)
    (linear_gradient,) = torch.autograd.grad((linear.weight() * direction).sum(), linear.codes)
    residual_gradient, power_gradient = torch.autograd.grad(
        (residual.weight() * direction).sum(),
        (residual.codes, residual.power_logits),
    )
    torch.testing.assert_close(residual_gradient, linear_gradient, rtol=1e-12, atol=1e-12)
    assert torch.isfinite(power_gradient).all() and power_gradient.norm() > 0

    # Completion agrees with the four-term equality and stays finite when
    # the increment is too small to recover as B-A in float32.
    for dtype in (torch.float32, torch.float64):
        a = torch.tensor([1.0, 2.0, 1e8, 1e-4], dtype=dtype, requires_grad=True)
        increment = torch.tensor([1.0, 1.0, 1e-4, 100.0], dtype=dtype, requires_grad=True)
        c = torch.tensor([3.0, 1.0, 1e-4, 100.0], dtype=dtype, requires_grad=True)
        power = torch.tensor([1.5, 1.9, 1.99, 0.76], dtype=dtype, requires_grad=True)
        log_d = _ordered_log_completion(a, increment, c, power)
        d = log_d.exp()
        assert torch.isfinite(d).all() and torch.all(d > 0)
        torch.testing.assert_close(
            a.pow(power) + d.pow(power),
            (a + increment).pow(power) + c.pow(power),
            rtol=2e-5,
            atol=2e-5,
        )
        gradients = torch.autograd.grad(log_d.sum(), (a, increment, c, power))
        assert all(torch.isfinite(gradient).all() for gradient in gradients)

    with torch.no_grad():
        residual.power_logits.copy_(torch.tensor([-3.0, -0.5, 0.5, 3.0], dtype=torch.float64))
    assert torch.all((residual.powers() > 0.75) & (residual.powers() < 2.0))
    quartet = residual.positive_features(residual.codes)
    a, b, c, d = quartet.unbind(dim=-1)
    torch.testing.assert_close(
        a.pow(residual.powers()) + d.pow(residual.powers()),
        b.pow(residual.powers()) + c.pow(residual.powers()),
    )
    torch.testing.assert_close(residual.residual_features(residual.codes), d - (b + c - a))
    ids = torch.tensor([[4, 0, 9], [7, 4, 11]])
    hidden = torch.randn(2, 3, 16, dtype=torch.float64)
    torch.testing.assert_close(residual.embed(ids), residual.weight()[ids] * residual.scale)
    torch.testing.assert_close(residual.project(hidden), F.linear(hidden, residual.weight()))

    shape = ModelConfig(
        vocab_size=64,
        d_model=16,
        n_heads=4,
        d_ff=20,
        n_encoder_layers=1,
        n_decoder_layers=1,
        dropout=0.0,
    )
    torch.manual_seed(42)
    baseline = build_model("embedding_linear", shape).double().eval()
    torch.manual_seed(42)
    model = build_model(name, shape).double().eval()
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    source = torch.tensor([[4, 5, 6, 2, 0], [7, 8, 9, 10, 2]])
    mask = source.ne(0).long()
    labels = torch.tensor([[11, 12, 2, 0], [13, 14, 15, 2]])
    target = model.shift_right(labels)
    torch.testing.assert_close(
        model.logits(source, mask, target, target.ne(0).long()),
        baseline.logits(source, mask, target, target.ne(0).long()),
        rtol=0,
        atol=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss, _ = model(source, mask, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert model.embedding.power_logits.grad.norm() > 0
    optimizer.step()
    with torch.no_grad():
        assert torch.count_nonzero(model.embedding.residual_features(model.embedding.codes)) > 0
        memory = model.encode(source, mask)
        full = model.embedding.project(model.decode(target, memory, mask, target.ne(0).long()))
        cache = model.new_cache()
        for position in range(target.shape[1]):
            incremental = model.decode_step(
                target[:, position : position + 1], memory, mask, cache, position
            )
            torch.testing.assert_close(incremental, full[:, position], rtol=1e-8, atol=1e-8)
        restored = build_model(name, shape).double().eval()
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(
            restored.embedding.weight(), model.embedding.weight(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            restored.logits(source, mask, target, target.ne(0).long()), full
        )
