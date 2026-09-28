"""One focused count, identity, gradient-activation, tying, and cache check."""

import torch
import torch.nn.functional as F

from ana.config import ModelConfig
from ana.nn.embeddings.analogy_embedding import ResidualPowerAnalogyEmbedding
from ana.registry import build_model, count_parameters, model_parameters


def test_embedding_feature_mixer_keeps_start_and_learns_after_power_moves():
    name = "ana_embedding_mix_learned"
    original = "ana_embedding_residual_learned"
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
    assert count_parameters(large) == model_parameters(name, large_shape) == 2_249_568
    assert model_parameters(name, large_shape) - model_parameters(original, large_shape) == 1_024
    assert large.embedding.learn_mixing is True
    assert large.embedding.feature_mixing.shape == (32, 32)
    torch.testing.assert_close(large.embedding.feature_mixing, torch.eye(32), rtol=0, atol=0)
    del large

    torch.manual_seed(42)
    plain = ResidualPowerAnalogyEmbedding(64, 16, 0).double()
    original_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(42)
    mixed = ResidualPowerAnalogyEmbedding(64, 16, 0, learn_mixing=True).double()
    assert torch.equal(original_rng, torch.random.get_rng_state())
    assert plain.learn_mixing is False and plain.feature_mixing is None
    assert set(plain.state_dict()) == {"codes", "basis", "power_logits", "complement"}
    assert set(mixed.state_dict()) == set(plain.state_dict()) | {"feature_mixing"}
    for key, value in plain.state_dict().items():
        torch.testing.assert_close(mixed.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(mixed.weight(), plain.weight(), rtol=0, atol=0)

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
    baseline = build_model(original, shape).double().eval()
    torch.manual_seed(42)
    model = build_model(name, shape).double().eval()
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    embedding = model.embedding
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model(source, mask, labels)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        assert embedding.power_logits.grad.norm() > 0
        if step == 0:
            assert torch.count_nonzero(embedding.feature_mixing.grad) == 0
        else:
            assert torch.count_nonzero(embedding.residual_features(embedding.codes)) > 0
            assert embedding.feature_mixing.grad.norm() > 0
        optimizer.step()

    with torch.no_grad():
        assert not torch.equal(embedding.feature_mixing, torch.eye(4, dtype=torch.float64))
        expected = embedding.codes @ embedding.basis + embedding.code_scale * (
            embedding.residual_features(embedding.codes)
            @ embedding.feature_mixing
            @ embedding.complement
        )
        torch.testing.assert_close(embedding.weight(), expected)
        ids = torch.tensor([[4, 0, 9], [7, 4, 11]])
        hidden = torch.randn(2, 3, 16, dtype=torch.float64)
        torch.testing.assert_close(embedding.embed(ids), expected[ids] * embedding.scale)
        torch.testing.assert_close(embedding.project(hidden), F.linear(hidden, expected))
        memory = model.encode(source, mask)
        full = embedding.project(model.decode(target, memory, mask, target.ne(0).long()))
        cache = model.new_cache()
        for position in range(target.shape[1]):
            incremental = model.decode_step(
                target[:, position : position + 1], memory, mask, cache, position
            )
            torch.testing.assert_close(incremental, full[:, position], rtol=1e-8, atol=1e-8)
