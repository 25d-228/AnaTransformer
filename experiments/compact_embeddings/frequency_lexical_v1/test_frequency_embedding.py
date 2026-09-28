"""One focused check: budget, tied gradients, padding, saved IDs and decoding."""

import io
import math

import torch
import torch.nn.functional as F

from ana.config import ModelConfig
from ana.nn.embeddings.frequency_embedding import FrequencyAdaptiveEmbedding
from ana.registry import build_model, count_parameters


def _direct_table(embedding):
    # Independent tiny reference assembled by original vocabulary ID.
    return torch.stack(
        [
            (
                embedding.full_rows[embedding.full_index[token]]
                if embedding.full_index[token] >= 0
                else embedding.codes[embedding.rare_index[token]] @ embedding.basis
            )
            for token in range(embedding.vocab_size)
        ]
    )


def test_frequency_adaptive_embedding():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    shape = ModelConfig(
        vocab_size=10000,
        d_model=128,
        d_ff=232,
        n_heads=4,
        n_encoder_layers=4,
        n_decoder_layers=4,
        dropout=0.3,
    )
    large = build_model("embedding_linear", shape)
    large.embedding = FrequencyAdaptiveEmbedding(10000, 128, 0, list(range(3376)), 80)
    assert sum(parameter.numel() for parameter in large.embedding.parameters()) == 972288
    assert count_parameters(large) == 2248512
    assert large.embedding.full_rows.shape == (3376, 128)
    assert large.embedding.codes.shape == (6624, 80)
    assert large.embedding.basis.shape == (80, 128)
    del large

    embedding = FrequencyAdaptiveEmbedding(12, 8, 0, [5, 0, 9, 2, 1], 4).double()
    parameters = tuple(embedding.parameters())
    ids = torch.tensor([[5, 3, 0, 5], [7, 2, 9, 0]])
    pad_row = embedding.full_index[0].item()
    assert embedding.code_scale == 8**-0.5
    torch.testing.assert_close(
        embedding.basis @ embedding.basis.T,
        torch.eye(4, dtype=torch.float64) * 2,
        atol=5e-7,
        rtol=5e-7,
    )
    assert torch.count_nonzero(embedding.full_rows[pad_row]) == 0
    direct = _direct_table(embedding)
    torch.testing.assert_close(embedding.weight(), direct, atol=1e-12, rtol=1e-12)
    expected_lookup = F.embedding(ids, direct) * math.sqrt(8)
    torch.testing.assert_close(embedding.embed(ids), expected_lookup, atol=1e-12, rtol=1e-12)

    # Match ordinary embedding padding semantics, including gradient suppression.
    independent = _direct_table(embedding)
    independent = torch.where(
        torch.arange(12).eq(0).unsqueeze(-1), independent.detach(), independent
    )
    lookup_probe = torch.randn(2, 4, 8, dtype=torch.float64)
    actual_gradient = torch.autograd.grad((embedding.embed(ids) * lookup_probe).sum(), parameters)
    direct_gradient = torch.autograd.grad(
        (F.embedding(ids, independent) * math.sqrt(8) * lookup_probe).sum(),
        parameters,
    )
    for actual, expected in zip(actual_gradient, direct_gradient, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        assert torch.isfinite(actual).all() and actual.abs().sum() > 0
    assert torch.count_nonzero(actual_gradient[0][pad_row]) == 0
    assert torch.count_nonzero(actual_gradient[0][embedding.full_index[1]]) == 0
    assert torch.count_nonzero(actual_gradient[1][embedding.rare_index[4]]) == 0

    hidden = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    actual_output = embedding.project(hidden)
    direct_output = F.linear(hidden, _direct_table(embedding))
    torch.testing.assert_close(actual_output, direct_output, atol=1e-12, rtol=1e-12)
    output_probe = torch.randn_like(actual_output)
    actual_gradient = torch.autograd.grad(
        (actual_output * output_probe).sum(), (hidden, *parameters)
    )
    direct_gradient = torch.autograd.grad(
        (direct_output * output_probe).sum(), (hidden, *parameters)
    )
    for actual, expected in zip(actual_gradient, direct_gradient, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        assert torch.isfinite(actual).all() and actual.abs().sum() > 0
    assert actual_gradient[1][pad_row].abs().sum() > 0

    # A pad row changed by output training remains visible but frozen in lookup.
    with torch.no_grad():
        embedding.full_rows[pad_row].fill_(0.25)
    padding_ids = torch.tensor([[0, 0]])
    torch.testing.assert_close(
        embedding.embed(padding_ids), embedding.weight()[padding_ids] * math.sqrt(8)
    )
    pad_gradient = torch.autograd.grad(embedding.embed(padding_ids).sum(), parameters)
    assert all(torch.count_nonzero(value) == 0 for value in pad_gradient)

    # Loading must restore the token assignment, including a changed local pad index.
    serialized = io.BytesIO()
    torch.save(embedding.state_dict(), serialized)
    serialized.seek(0)
    restored = FrequencyAdaptiveEmbedding(12, 8, 0, [0, 3, 6, 8, 11], 4).double()
    restored.load_state_dict(torch.load(serialized, weights_only=True))
    for name in ("full_token_ids", "rare_token_ids", "full_index", "rare_index"):
        assert name in restored.state_dict()
        torch.testing.assert_close(
            getattr(restored, name), getattr(embedding, name), atol=0, rtol=0
        )
    torch.testing.assert_close(restored.weight(), embedding.weight(), atol=0, rtol=0)
    torch.testing.assert_close(restored.embed(ids), embedding.embed(ids), atol=0, rtol=0)
    restored_pad_gradient = torch.autograd.grad(
        restored.embed(padding_ids).sum(), restored.full_rows
    )[0]
    assert torch.count_nonzero(restored_pad_gradient) == 0

    tiny = ModelConfig(
        vocab_size=24,
        d_model=16,
        d_ff=20,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        dropout=0.0,
    )
    model = build_model("embedding_linear", tiny)
    model.embedding = FrequencyAdaptiveEmbedding(24, 16, 0, [0, 1, 2, 3, 17, 5, 10, 22], 8)
    model.double().eval()
    source = torch.tensor([[4, 5, 6, 2, 0], [7, 8, 9, 10, 2]])
    source_mask = source.ne(0)
    labels = torch.tensor([[11, 12, 2, -100], [13, 14, 15, 2]])
    target = model.shift_right(labels)
    with torch.no_grad():
        memory = model.encode(source, source_mask)
        full = model.embedding.project(model.decode(target, memory, source_mask, target.ne(0)))
        cache = model.new_cache()
        for position in range(target.size(1)):
            incremental = model.decode_step(
                target[:, position : position + 1], memory, source_mask, cache, position
            )
            torch.testing.assert_close(incremental, full[:, position], atol=1e-8, rtol=1e-8)
