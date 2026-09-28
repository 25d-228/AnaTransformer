"""One focused count, quartet, initialization, learning, tying, and cache check."""

import math

import torch
import torch.nn.functional as F

from ana.config import ModelConfig
from ana.nn.embeddings.analogy_embedding import (
    WHITENING_GAIN_CAP,
    LinearCompressedEmbedding,
    PowerAnalogyEmbedding,
    _initial_feature_calibration,
)
from ana.nn.projection import SeparateQKV
from ana.registry import build_model, count_parameters, model_parameters


def test_compact_embeddings_use_learned_power_preserve_scale_and_tied_decoding():
    # Calibration is private and independent of task/model initialization RNG.
    _initial_feature_calibration.cache_clear()
    rng_before = torch.random.get_rng_state().clone()
    center, scale, whitening = _initial_feature_calibration()
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert center > 0 and scale > 0
    assert torch.linalg.svdvals(whitening).max() <= WHITENING_GAIN_CAP + 1e-10

    cases = (
        ("embedding_linear", LinearCompressedEmbedding, 232, 2_248_512, 96),
        ("ana_embedding_learned", PowerAnalogyEmbedding, 230, 2_248_528, 128),
    )
    for name, kind, d_ff, expected_count, expected_rank in cases:
        config = ModelConfig(
            vocab_size=10_000,
            d_model=128,
            n_heads=4,
            d_ff=d_ff,
            n_encoder_layers=4,
            n_decoder_layers=4,
            dropout=0.3,
        )
        torch.manual_seed(42)
        large = build_model(name, config)
        embedding = large.embedding
        assert isinstance(embedding, kind)
        assert count_parameters(large) == model_parameters(name, config) == expected_count
        assert embedding.codes.shape == (10_000, 96)
        assert embedding.code_scale == 128**-0.5
        assert sum(isinstance(module, SeparateQKV) for module in large.modules()) == 12
        with torch.no_grad():
            weight = embedding.weight()
            assert weight.shape == (10_000, 128)
            assert torch.isfinite(weight).all()
            assert weight.mean().abs() < 0.005
            assert 0.7 < weight[1:].square().mean() * 128 < 1.3
            # The float32 table multiplication adds tiny rounding residuals to
            # the rank-96 linear table, so use an explicit numerical tolerance.
            assert (
                torch.linalg.matrix_rank(weight[1:1025].double(), rtol=1e-5).item()
                == expected_rank
            )
            assert weight[0].abs().max() < 1e-6
            if isinstance(embedding, PowerAnalogyEmbedding):
                assert embedding.power_logits.shape == (32,)
                torch.testing.assert_close(embedding.powers(), torch.full((32,), 2.0))
        del large, embedding, weight

    shape = ModelConfig(
        vocab_size=64,
        d_model=16,
        n_heads=4,
        d_ff=20,
        n_encoder_layers=1,
        n_decoder_layers=1,
        dropout=0.0,
    )
    source = torch.tensor([[4, 5, 6, 2, 0], [7, 8, 9, 10, 2]])
    mask = source.ne(0).long()
    labels = torch.tensor([[11, 12, 2, 0], [13, 14, 15, 2]])
    for name, _kind, *_ in cases:
        torch.manual_seed(42)
        baseline = build_model("baseline", shape).double().eval()
        torch.manual_seed(42)
        model = build_model(name, shape).double().eval()
        embedding = model.embedding
        assert count_parameters(model) == model_parameters(name, shape)
        for key, value in baseline.state_dict().items():
            if not key.startswith("embedding."):
                torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)

        # A lookup touches only the requested latent rows and matches the same
        # freshly generated table used for vocabulary prediction.
        ids = torch.tensor([[4, 0, 5], [7, 4, 9]])
        table = embedding.weight()
        torch.testing.assert_close(embedding.embed(ids), table[ids] * math.sqrt(shape.d_model))
        (lookup_gradient,) = torch.autograd.grad(
            embedding.embed(ids).square().sum(), embedding.codes
        )
        assert torch.count_nonzero(lookup_gradient[0]) == 0
        assert torch.count_nonzero(lookup_gradient[6]) == 0
        assert lookup_gradient[4].norm() > 0
        hidden = torch.randn(2, 3, shape.d_model, dtype=torch.float64)
        torch.testing.assert_close(embedding.project(hidden), F.linear(hidden, table))

        if isinstance(embedding, PowerAnalogyEmbedding):
            # Both lookup and output projection directly depend on power before
            # any optimizer update; no initially closed residual gate is used.
            (lookup_power,) = torch.autograd.grad(
                embedding.embed(ids).square().sum(), embedding.power_logits
            )
            (output_power,) = torch.autograd.grad(
                embedding.project(hidden).square().sum(), embedding.power_logits
            )
            assert lookup_power.norm() > 0 and output_power.norm() > 0
            features = embedding.positive_features(embedding.codes)
            powers = embedding.powers()[None, :]
            assert features.min() > 0
            a, b, c, d = features.unbind(-1)
            torch.testing.assert_close(
                a.pow(powers) + d.pow(powers), b.pow(powers) + c.pow(powers)
            )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        old_table = embedding.weight().detach().clone()
        loss, _ = model(source, mask, labels)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        assert embedding.codes.grad.norm() > 0 and embedding.basis.grad.norm() > 0
        if isinstance(embedding, PowerAnalogyEmbedding):
            assert embedding.power_logits.grad.norm() > 0
        # As in the standard tied table, output prediction can train the pad
        # row even though padding lookup never supplies a code-row gradient.
        assert embedding.codes.grad[shape.pad_id].norm() > 0
        optimizer.step()

        with torch.no_grad():
            assert not torch.equal(old_table, embedding.weight())
            torch.testing.assert_close(
                embedding.project(hidden), F.linear(hidden, embedding.weight())
            )
            memory = model.encode(source, mask)
            target = model.shift_right(labels)
            full = embedding.project(model.decode(target, memory, mask, target.ne(0).long()))
            cache = model.new_cache()
            for position in range(target.shape[1]):
                incremental = model.decode_step(
                    target[:, position : position + 1], memory, mask, cache, position
                )
                torch.testing.assert_close(incremental, full[:, position], rtol=1e-8, atol=1e-8)

            restored = build_model(name, shape).double().eval()
            restored.load_state_dict(model.state_dict())
            torch.testing.assert_close(
                restored.embedding.weight(), embedding.weight(), rtol=0, atol=0
            )
            torch.testing.assert_close(
                restored.logits(source, mask, target, target.ne(0).long()), full
            )
