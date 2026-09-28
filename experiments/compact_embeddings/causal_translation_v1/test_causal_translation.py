"""One focused check: exact four-term scoring, causality, gradients and beam state."""

import torch

from ana.causal_translation import CausalTranslationHead, CausalTranslationTransformer
from ana.config import ModelConfig
from ana.registry import build_model, count_parameters


def test_causal_translation():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    head = CausalTranslationHead(8).double()
    context = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    table = torch.randn(13, 8, dtype=torch.float64, requires_grad=True)
    source = head.source_features(context)
    vocab = head.vocabulary_features(table)
    prefix = torch.tensor([[1, 4, 5, 6], [1, 7, 0, 9]])
    mask = prefix.ne(0)
    actual = head.causal_correction(source, vocab, prefix, mask, 0)
    direct = torch.zeros_like(actual)
    for row in range(2):
        for t in range(4):
            eligible = [s for s in range(t) if mask[row, s] and prefix[row, s + 1] != 0]
            if not mask[row, t] or not eligible:
                continue
            earlier = source[row, eligible]
            current = source[row, t]
            weights = torch.softmax(-(earlier - current).square().mean(-1), dim=0)
            previous_tokens = vocab[prefix[row, torch.tensor(eligible) + 1]]
            # A + D - B - C, separately for each reference and candidate.
            defect = earlier[:, None] + vocab[None] - previous_tokens[:, None] - current
            direct[row, t] = -head.gain() * (weights[:, None] * defect.square().mean(-1)).sum(0)
    centered = actual - actual.mean(-1, keepdim=True)
    direct_centered = direct - direct.mean(-1, keepdim=True)
    torch.testing.assert_close(centered, direct_centered, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(actual[:, 0], torch.zeros_like(actual[:, 0]), atol=0, rtol=0)
    torch.testing.assert_close(actual[1, 2], torch.zeros_like(actual[1, 2]), atol=0, rtol=0)
    probe = torch.randn_like(actual)
    variables = (context, table, *tuple(head.parameters()))
    fast_grad = torch.autograd.grad((centered * probe).sum(), variables, retain_graph=True)
    slow_grad = torch.autograd.grad((direct_centered * probe).sum(), variables)
    for first, second in zip(fast_grad, slow_grad, strict=True):
        assert torch.isfinite(first).all()
        torch.testing.assert_close(first, second, atol=1e-9, rtol=1e-8)

    # Preserve the completed linear backbone's exact initialization and shape.
    shape = ModelConfig(
        vocab_size=10000,
        d_model=128,
        d_ff=232,
        n_heads=4,
        n_encoder_layers=4,
        n_decoder_layers=4,
        dropout=0.3,
    )
    torch.manual_seed(19)
    baseline = build_model("embedding_linear", shape)
    torch.manual_seed(19)
    candidate = CausalTranslationTransformer(shape)
    assert count_parameters(candidate) == 2249545
    candidate_state = candidate.state_dict()
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(candidate_state[name], value, atol=0, rtol=0)
    del baseline, candidate, candidate_state

    tiny = ModelConfig(
        vocab_size=24,
        d_model=16,
        d_ff=24,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=2,
        dropout=0.0,
    )
    model = CausalTranslationTransformer(tiny).eval()
    inputs = torch.tensor([[4, 5, 6, 2], [7, 8, 2, 0], [9, 10, 11, 2]])
    source_mask = inputs.ne(0)
    prefix = torch.tensor([[1, 12, 13, 14], [1, 15, 16, 17], [1, 18, 19, 20]])
    with torch.no_grad():
        memory = model.encode(inputs, source_mask)
        cache = model.new_cache()
        for position in range(prefix.size(1)):
            incremental = model.decode_step(
                prefix[:, position : position + 1], memory, source_mask, cache, position
            )
            full = model.logits(
                inputs,
                source_mask,
                prefix[:, : position + 1],
                torch.ones_like(prefix[:, : position + 1]),
            )[:, -1]
            torch.testing.assert_close(incremental, full, atol=3e-6, rtol=3e-6)
            if position == 1:
                parents = torch.tensor([2, 0, 2])
                # Exactly the existing beam decoder's generic cache operation.
                for self_cache, cross_cache in cache:
                    for held in (self_cache, cross_cache):
                        if held.key is not None:
                            held.key = held.key[parents]
                            held.value = held.value[parents]
                inputs, source_mask = inputs[parents], source_mask[parents]
                memory, prefix = memory[parents], prefix[parents]
        original = model.logits(inputs, source_mask, prefix, torch.ones_like(prefix))
        changed_prefix = prefix.clone()
        changed_prefix[:, -1] = 21
        changed = model.logits(inputs, source_mask, changed_prefix, torch.ones_like(prefix))
        torch.testing.assert_close(original[:, :-1], changed[:, :-1], atol=1e-6, rtol=1e-6)

    labels = torch.tensor([[4, 5, 6, 2], [7, 8, 2, -100], [9, 10, 11, 2]])
    model.train()
    loss, scores = model(inputs, source_mask, labels)
    assert torch.isfinite(loss) and torch.isfinite(scores).all()
    loss.backward()
    for parameter in model.analogy_head.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
