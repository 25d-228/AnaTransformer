"""One focused unit-row count, compatibility, tying, learning, and cache check."""

import torch
import torch.nn.functional as F

from ana.config import ModelConfig
from ana.nn.embeddings.analogy_embedding import LinearCompressedEmbedding, PowerAnalogyEmbedding
from ana.nn.projection import SeparateQKV
from ana.registry import build_model, count_parameters, model_parameters


def test_unit_embedding_rows_keep_padding_tying_power_and_old_defaults():
    cases = (
        ("embedding_unit_linear", "embedding_linear", LinearCompressedEmbedding, 232, 2_248_512),
        (
            "ana_embedding_unit_learned",
            "ana_embedding_learned",
            PowerAnalogyEmbedding,
            230,
            2_248_528,
        ),
    )
    for name, old_name, kind, d_ff, expected_count in cases:
        large_shape = ModelConfig(
            vocab_size=10_000,
            d_model=128,
            n_heads=4,
            d_ff=d_ff,
            n_encoder_layers=4,
            n_decoder_layers=4,
            dropout=0.3,
        )
        large = build_model(name, large_shape)
        assert count_parameters(large) == model_parameters(name, large_shape) == expected_count
        assert model_parameters(old_name, large_shape) == expected_count
        assert sum(isinstance(module, SeparateQKV) for module in large.modules()) == 12
        assert large.embedding.normalize_rows is True
        with torch.no_grad():
            table = large.embedding.weight()
            torch.testing.assert_close(
                table[1:].norm(dim=-1), torch.ones(9_999), rtol=1e-5, atol=1e-6
            )
            assert table[0].abs().max() < 1e-6
        del large, table

        # The default consumes identical RNG and gives bitwise identical old
        # effective-table calculations; no state keys or parameters are added.
        torch.manual_seed(42)
        default = kind(64, 16, 0).double()
        torch.manual_seed(42)
        explicit_old = kind(64, 16, 0, normalize_rows=False).double()
        assert default.normalize_rows is False
        assert default.state_dict().keys() == explicit_old.state_dict().keys()
        for key, value in default.state_dict().items():
            torch.testing.assert_close(value, explicit_old.state_dict()[key], rtol=0, atol=0)
        raw_table = (
            default._transform(default.codes)
            if isinstance(default, PowerAnalogyEmbedding)
            else default.codes @ default.basis
        )
        torch.testing.assert_close(default.weight(), raw_table, rtol=0, atol=0)

        unit = kind(64, 16, 0, normalize_rows=True).double()
        unit.load_state_dict(default.state_dict())
        # Make the padding row visibly nonzero: the exception must hold after
        # output training too, not merely for a coincidentally zero initial row.
        with torch.no_grad():
            unit.codes[0].copy_(unit.codes[4] * 0.37)
        raw_unit = (
            unit._transform(unit.codes)
            if isinstance(unit, PowerAnalogyEmbedding)
            else unit.codes @ unit.basis
        )
        table = unit.weight()
        torch.testing.assert_close(table[0], raw_unit[0], rtol=0, atol=0)
        assert not torch.isclose(raw_unit[0].norm(), torch.tensor(1.0, dtype=torch.float64))
        torch.testing.assert_close(table[1:].norm(dim=-1), torch.ones(63, dtype=torch.float64))
        ids = torch.tensor([[4, 0, 9], [7, 4, 11]])
        torch.testing.assert_close(unit.embed(ids), table[ids] * unit.scale)
        hidden = torch.randn(2, 3, 16, dtype=torch.float64)
        torch.testing.assert_close(unit.project(hidden), F.linear(hidden, table))
        (lookup_gradient,) = torch.autograd.grad(unit.embed(ids).sum(), unit.codes)
        assert torch.count_nonzero(lookup_gradient[0]) == 0
        assert torch.count_nonzero(lookup_gradient[6]) == 0

        shape = ModelConfig(
            vocab_size=64,
            d_model=16,
            n_heads=4,
            d_ff=20,
            n_encoder_layers=1,
            n_decoder_layers=1,
            dropout=0.0,
        )
        model = build_model(name, shape).double().eval()
        source = torch.tensor([[4, 5, 6, 2, 0], [7, 8, 9, 10, 2]])
        mask = source.ne(0).long()
        labels = torch.tensor([[11, 12, 2, 0], [13, 14, 15, 2]])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss, _ = model(source, mask, labels)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        assert model.embedding.codes.grad.norm() > 0
        assert model.embedding.basis.grad.norm() > 0
        if isinstance(model.embedding, PowerAnalogyEmbedding):
            assert model.embedding.power_logits.grad.norm() > 0
        optimizer.step()
        with torch.no_grad():
            table = model.embedding.weight()
            torch.testing.assert_close(table[1:].norm(dim=-1), torch.ones(63, dtype=torch.float64))
            torch.testing.assert_close(
                model.embedding.embed(ids), table[ids] * model.embedding.scale
            )
            torch.testing.assert_close(model.embedding.project(hidden), F.linear(hidden, table))
            memory = model.encode(source, mask)
            target = model.shift_right(labels)
            full = model.embedding.project(model.decode(target, memory, mask, target.ne(0).long()))
            cache = model.new_cache()
            for position in range(target.shape[1]):
                incremental = model.decode_step(
                    target[:, position : position + 1], memory, mask, cache, position
                )
                torch.testing.assert_close(incremental, full[:, position], rtol=1e-8, atol=1e-8)
