"""Hard global shuffles: coverage, gradients, and the existing D4 integration."""

from __future__ import annotations

from itertools import product

import pytest
import torch

from ana.config import ENCODER_ONLY, ModelConfig
from ana.nn.projection import SharedQKV
from ana.nn.shuffle import BinarySwapPermutation, ShuffledD4QKV
from ana.registry import REGISTRY, build_model, count_parameters, model_parameters

VARIANTS = [("ana_shuffle_benes_enc", "benes")]
CONFIG = ModelConfig(
    vocab_size=64,
    d_model=16,
    n_heads=4,
    d_ff=32,
    n_encoder_layers=1,
    n_decoder_layers=1,
    dropout=0.0,
)


@pytest.mark.parametrize(("name", "topology"), VARIANTS)
def test_shuffle_placement_and_declared_parameter_cost(name, topology):
    model = build_model(name, CONFIG)
    projection = model.encoder[0].self_attention.projection
    assert REGISTRY[name].sites == ENCODER_ONLY
    assert isinstance(projection, ShuffledD4QKV)
    assert type(model.decoder[0].self_attention.projection) is SharedQKV
    assert type(model.decoder[0].cross_attention.projection) is SharedQKV

    stages = 4 if topology == "butterfly" else 7
    switches_per_role = stages * CONFIG.d_model // 2
    assert projection.key_shuffle.logits.numel() == switches_per_role
    assert projection.value_shuffle.logits.numel() == switches_per_role
    assert BinarySwapPermutation.switch_count(CONFIG.d_model, topology) == switches_per_role
    assert count_parameters(model) == model_parameters(name, CONFIG)
    assert count_parameters(model) - model_parameters("ana_d4_enc", CONFIG) == (
        CONFIG.n_encoder_layers * 2 * switches_per_role
    )


@pytest.mark.parametrize(("name", "topology"), VARIANTS)
def test_identity_initialization_preserves_the_d4_model_and_backbone(name, topology):
    torch.manual_seed(11)
    reference = build_model("ana_d4_enc", CONFIG).eval()
    torch.manual_seed(11)
    shuffled = build_model(name, CONFIG).eval()

    for key, tensor in reference.state_dict().items():
        torch.testing.assert_close(shuffled.state_dict()[key], tensor, rtol=0, atol=0)
    source = torch.randint(4, CONFIG.vocab_size, (2, 5))
    labels = torch.randint(4, CONFIG.vocab_size, (2, 3))
    mask = torch.ones_like(source)
    with torch.no_grad():
        expected = reference(source, mask, labels)
        observed = shuffled(source, mask, labels)
    for actual, wanted in zip(observed, expected, strict=True):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


@pytest.mark.parametrize("topology", ["butterfly", "benes"])
@pytest.mark.parametrize("width", [4, 8, 16, 128, 512])
def test_arbitrary_switches_are_exact_permutations_in_train_and_eval(topology, width):
    torch.manual_seed(width)
    shuffle = BinarySwapPermutation(width, topology).double()
    with torch.no_grad():
        shuffle.logits.normal_()
    ascending = tuple(1 << bit for bit in range(width.bit_length() - 1))
    expected_strides = ascending if topology == "butterfly" else ascending + ascending[-2::-1]
    assert shuffle.strides == expected_strides

    indices = shuffle.permutation_indices()
    assert torch.equal(indices.sort().values, torch.arange(width))
    x = torch.randn(2, 3, width, dtype=torch.float64)
    expected = x[..., indices]
    for training in (True, False):
        shuffle.train(training)
        actual = shuffle(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual.sort().values, x.sort().values, rtol=0, atol=0)
        torch.testing.assert_close(actual.square().sum(-1), x.square().sum(-1))


def test_benes_covers_every_four_feature_permutation_but_butterfly_does_not():
    for topology, expected_count in (("butterfly", 16), ("benes", 24)):
        shuffle = BinarySwapPermutation(4, topology)
        permutations = set()
        with torch.no_grad():
            for choices in product((-1.0, 1.0), repeat=shuffle.logits.numel()):
                shuffle.logits.copy_(torch.tensor(choices).view_as(shuffle.logits))
                permutations.add(tuple(shuffle.permutation_indices().tolist()))
        assert len(permutations) == expected_count


@pytest.mark.parametrize("topology", ["butterfly", "benes"])
def test_one_switch_can_move_features_between_local_quartets(topology):
    shuffle = BinarySwapPermutation(8, topology)
    with torch.no_grad():
        shuffle.logits[shuffle.strides.index(4), 0] = 1.0
    expected = torch.tensor([4, 1, 2, 3, 0, 5, 6, 7])
    assert torch.equal(shuffle.permutation_indices(), expected)
    assert torch.equal(shuffle(torch.arange(8.0)), expected.float())


@pytest.mark.parametrize("logit", [-0.7, 0.7])
def test_switch_gradient_matches_the_surrogate_and_input_gradient_is_hard(logit):
    shuffle = BinarySwapPermutation(4, "butterfly").double()
    with torch.no_grad():
        shuffle.logits.fill_(-1.0)
        shuffle.logits[0].fill_(logit)
    x = torch.tensor([[2.0, 5.0, -1.0, 4.0]], dtype=torch.float64, requires_grad=True)
    upstream = torch.tensor([[3.0, -2.0, 7.0, 1.0]], dtype=torch.float64)
    (shuffle(x) * upstream).sum().backward()

    probability = torch.sigmoid(torch.tensor(logit, dtype=torch.float64))
    pairs = x.reshape(1, 2, 2)
    upstream_pairs = upstream.reshape(1, 2, 2)
    expected_logit_grad = (
        ((pairs[..., 1] - pairs[..., 0]) * (upstream_pairs[..., 0] - upstream_pairs[..., 1]))
        .sum(0)
        * probability
        * (1 - probability)
    )
    torch.testing.assert_close(shuffle.logits.grad[0], expected_logit_grad)
    assert torch.isfinite(shuffle.logits.grad).all()
    assert torch.count_nonzero(shuffle.logits.grad) > 0
    expected_input_grad = upstream if logit < 0 else upstream_pairs.flip(-1).reshape_as(upstream)
    torch.testing.assert_close(x.grad, expected_input_grad, rtol=0, atol=0)


@pytest.mark.parametrize("topology", ["butterfly", "benes"])
def test_shuffled_projection_split_paths_and_all_parameter_gradients(topology):
    torch.manual_seed(17)
    projection = ShuffledD4QKV(CONFIG.d_model, topology)
    with torch.no_grad():
        projection.key_shuffle.logits.normal_()
        projection.value_shuffle.logits.normal_()
        for role in (projection.query_role, projection.key_role, projection.value_role):
            role.router.weight.normal_(std=0.1)
    query_input = torch.randn(2, 3, CONFIG.d_model)
    kv_input = torch.randn(2, 5, CONFIG.d_model)
    query_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    kv_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])

    for training in (True, False):
        projection.train(training)
        for query, kv, q_mask, k_mask in (
            (query_input, query_input, query_mask, query_mask),
            (query_input, kv_input, query_mask, kv_mask),
        ):
            actual_q, actual_k, actual_v = projection(query, kv, q_mask, k_mask)
            expected_k, expected_v = projection.key_value(kv, k_mask)
            torch.testing.assert_close(actual_q, projection.query_only(query, q_mask))
            torch.testing.assert_close(actual_k, expected_k)
            torch.testing.assert_close(actual_v, expected_v)

    projection.train()
    outputs = projection(query_input, kv_input, query_mask, kv_mask)
    sum((output * torch.randn_like(output)).sum() for output in outputs).backward()
    for name, parameter in projection.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name


def test_shuffle_rejects_unsupported_shapes_and_topologies():
    for width in (0, 3, 12):
        with pytest.raises(ValueError):
            BinarySwapPermutation(width)
    with pytest.raises(ValueError):
        BinarySwapPermutation(8, "unknown")
