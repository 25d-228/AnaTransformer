"""One focused CPU check for the flexible rank and two-corpus model budget."""

import math

import torch
import torch.nn.functional as F
from models import MODEL_NAMES, construct, model_summary

from ana.config import ModelConfig
from ana.nn.embeddings.analogy_embedding import LinearCompressedEmbedding
from ana.registry import count_parameters


def test_compact_models():
    torch.set_num_threads(2)
    torch.manual_seed(31)
    default = LinearCompressedEmbedding(64, 16, 0)
    default_rng = torch.random.get_rng_state()
    torch.manual_seed(31)
    explicit = LinearCompressedEmbedding(64, 16, 0, code_dim=12)
    assert torch.equal(default_rng, torch.random.get_rng_state())
    for name, value in default.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[name], atol=0, rtol=0)

    enfr = ModelConfig(
        vocab_size=10000,
        d_model=128,
        d_ff=256,
        n_heads=4,
        n_encoder_layers=4,
        n_decoder_layers=4,
        dropout=0.3,
    )
    cogs = ModelConfig(
        vocab_size=835,
        d_model=512,
        d_ff=512,
        n_heads=8,
        n_encoder_layers=2,
        n_decoder_layers=2,
        dropout=0.1,
    )
    for corpus, full, expected in (
        ("enfr", enfr, (2605568, 2249936, 2213888, 2248512, 2248512)),
        ("cogs", cogs, (8844800, 5690376, 5702144, 5689236, 5689236)),
    ):
        assert (
            tuple(model_summary(corpus, name, full)["parameters"] for name in MODEL_NAMES)
            == expected
        )
    compact_cogs = construct("cogs", "embedding_linear", cogs)
    assert count_parameters(compact_cogs) == 5689236
    assert (
        compact_cogs.config.d_model,
        compact_cogs.config.d_ff,
        compact_cogs.embedding.code_dim,
    ) == (400, 509, 160)
    assert compact_cogs.embedding.codes.shape == (835, 160)
    assert compact_cogs.embedding.basis.shape == (160, 400)
    del compact_cogs

    torch.manual_seed(42)
    linear = construct("enfr", "embedding_linear", enfr)
    torch.manual_seed(42)
    powered = construct("enfr", "embedding_power_consistency", enfr)
    assert count_parameters(linear) == count_parameters(powered) == 2248512
    assert (linear.config.d_model, linear.config.d_ff, linear.embedding.code_dim) == (128, 232, 96)
    for name, value in linear.state_dict().items():
        torch.testing.assert_close(value, powered.state_dict()[name], atol=0, rtol=0)
    del linear, powered

    tiny = ModelConfig(
        vocab_size=64,
        d_model=16,
        d_ff=32,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        dropout=0.0,
    )
    model = construct("enfr", "embedding_linear", tiny)
    # Exercise a nondefault rank in the same ordinary cached decoder.
    model.embedding = LinearCompressedEmbedding(64, 16, 0, code_dim=8)
    model.double().eval()
    embedding = model.embedding
    ids = torch.tensor([[4, 0, 5], [7, 4, 9]])
    table = embedding.weight()
    torch.testing.assert_close(
        embedding.embed(ids), table[ids] * math.sqrt(16), atol=1e-12, rtol=1e-12
    )
    code_gradient, basis_gradient = torch.autograd.grad(
        embedding.embed(ids).square().sum(),
        (embedding.codes, embedding.basis),
    )
    assert code_gradient[0].count_nonzero() == 0 and code_gradient[6].count_nonzero() == 0
    assert code_gradient[4].abs().sum() > 0 and basis_gradient.abs().sum() > 0
    hidden = torch.randn(2, 3, 16, dtype=torch.float64)
    torch.testing.assert_close(
        embedding.project(hidden), F.linear(hidden, table), atol=1e-12, rtol=1e-12
    )
    output_gradient = torch.autograd.grad(embedding.project(hidden).sum(), embedding.codes)[0]
    assert output_gradient[0].abs().sum() > 0

    source = torch.tensor([[4, 5, 6, 2, 0], [7, 8, 9, 10, 2]])
    source_mask = source.ne(0)
    labels = torch.tensor([[11, 12, 2, -100], [13, 14, 15, 2]])
    target = model.shift_right(labels)
    with torch.no_grad():
        memory = model.encode(source, source_mask)
        full = embedding.project(model.decode(target, memory, source_mask, target.ne(0)))
        cache = model.new_cache()
        for position in range(target.size(1)):
            incremental = model.decode_step(
                target[:, position : position + 1], memory, source_mask, cache, position
            )
            torch.testing.assert_close(incremental, full[:, position], atol=1e-8, rtol=1e-8)
