"""Compact tied lexical tables with ordinary linear or four-term power features.

The numerical analogy holds for the positive quartet, before common centering
and the learned basis. The final signed embedding is not itself asserted to be
a numerical analogy. Parameters are latent token codes, the shared basis,
learned powers where used, and an optional small feature mixer. A dense table
is generated when needed and is never retained as a parameter or inference cache.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import Tensor, nn

RADIUS_EPSILON = 1e-4
CONTRAST_LIMIT = 0.95
INITIAL_POWER = 2.0
MIN_POWER = 0.5
MAX_POWER = 4.0
CALIBRATION_SAMPLES = 65_536
CALIBRATION_SEED = 24_071_870
WHITENING_GAIN_CAP = 8.0
RADIUS_RAW_MEAN = math.log(math.expm1(1.0 - RADIUS_EPSILON))
RADIUS_RAW_STD = 0.5


@lru_cache(maxsize=1)
def _initial_feature_calibration() -> tuple[Tensor, Tensor, Tensor]:
    """Private, fixed population estimate; never consumes the model's RNG.

    All four coordinates use ONE scalar center and ONE scalar RMS scale.
    The covariance is formed around that common center, not each token's own
    mean. The symmetric inverse square root balances the four initial feature
    directions; its amplification is capped at eight after the common scaling.
    This is an initialization of a subsequently free learned basis, not a
    whitening operation repeatedly applied during training.
    """
    generator = torch.Generator(device="cpu").manual_seed(CALIBRATION_SEED)
    raw = torch.randn(CALIBRATION_SAMPLES, 3, generator=generator, dtype=torch.float64)
    radius = F.softplus(raw[:, 0] * RADIUS_RAW_STD + RADIUS_RAW_MEAN) + RADIUS_EPSILON
    u = CONTRAST_LIMIT * raw[:, 1].tanh()
    v = CONTRAST_LIMIT * raw[:, 2].tanh()
    bases = torch.stack((1 + u, 1 + v, 1 - v, 1 - u), dim=-1)
    features = radius[:, None] * bases.pow(1.0 / INITIAL_POWER)
    center = features.mean()
    centered = features - center
    scale = centered.square().mean().sqrt()
    standardized = centered / scale
    covariance = standardized.T @ standardized / CALIBRATION_SAMPLES
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    gains = eigenvalues.clamp_min(1e-12).rsqrt().clamp_max(WHITENING_GAIN_CAP)
    whitening = (eigenvectors * gains) @ eigenvectors.T
    return center, scale, whitening


def _check_shape(vocab_size: int, d_model: int, pad_id: int) -> None:
    if d_model < 4 or d_model % 4:
        raise ValueError("compressed embeddings require width divisible by four")
    if vocab_size < 1 or not 0 <= pad_id < vocab_size:
        raise ValueError("compressed embeddings require a valid vocabulary and padding id")


def _unit_rows_except_padding(weight: Tensor, is_padding: Tensor) -> Tensor:
    """Normalize the effective table, leaving its trainable padding row alone.

    Choose the denominator before dividing, so a near-zero padding row neither
    becomes a unit rounding error nor acquires a 1/epsilon gradient. The same
    operation serves requested-row lookup and complete-table output scoring.
    """
    norm = torch.linalg.vector_norm(weight, dim=-1, keepdim=True).clamp_min(1e-6)
    denominator = torch.where(is_padding[..., None], torch.ones_like(norm), norm)
    return weight / denominator


class LinearCompressedEmbedding(nn.Module):
    """A configurable latent code and linear basis; default width is 3*d/4.

    Codes have standard deviation sigma=1/sqrt(d_model), as the ordinary
    embedding does. The basis has orthogonal rows with norm sqrt(d/code_dim),
    giving average embedding variance 1/d_model without shrinking code updates.
    Padding lookup suppresses gradients to its code row, as nn.Embedding does;
    tied vocabulary prediction may still train that row, as in TiedEmbedding.
    Optional row normalization gives non-padding effective rows unit L2 norm
    (with a 1e-6 denominator floor) in both lookup and output prediction.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        pad_id: int,
        normalize_rows: bool = False,
        *,
        code_dim: int | None = None,
    ) -> None:
        super().__init__()
        _check_shape(vocab_size, d_model, pad_id)
        if not isinstance(normalize_rows, bool):
            raise ValueError("normalize_rows must be a boolean")
        if code_dim is not None and (
            isinstance(code_dim, bool)
            or not isinstance(code_dim, int)
            or not 1 <= code_dim <= d_model
        ):
            raise ValueError("code width must be an integer between one and model width")
        self.normalize_rows = normalize_rows
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_groups = d_model // 4
        self.code_dim = 3 * self.n_groups if code_dim is None else code_dim
        self.pad_id = pad_id
        self.scale = math.sqrt(d_model)
        self.code_scale = d_model**-0.5
        self.codes = nn.Parameter(torch.randn(vocab_size, self.code_dim) * self.code_scale)
        self.basis = nn.Parameter(torch.empty(self.code_dim, d_model))
        nn.init.orthogonal_(self.basis, gain=math.sqrt(d_model / self.code_dim))
        with torch.no_grad():
            self.codes[pad_id].zero_()

    def weight(self) -> Tensor:
        weight = self.codes @ self.basis
        if self.normalize_rows:
            ids = torch.arange(self.vocab_size, device=weight.device)
            return _unit_rows_except_padding(weight, ids.eq(self.pad_id))
        return weight

    def embed(self, token_ids: Tensor) -> Tensor:
        codes = F.embedding(token_ids, self.codes, padding_idx=self.pad_id)
        weight = codes @ self.basis
        if self.normalize_rows:
            weight = _unit_rows_except_padding(weight, token_ids.eq(self.pad_id))
        return weight * self.scale

    def project(self, hidden: Tensor) -> Tensor:
        return F.linear(hidden, self.weight())


def _ordered_log_completion(a: Tensor, increment: Tensor, c: Tensor, power: Tensor) -> Tensor:
    """log(D) for A:B :: C:D, with B=A+increment>A.

    More explicitly, A^p+D^p=B^p+C^p. Keep the positive increment separately:
    B-A and B^p-A^p can lose a small increment by floating-point cancellation.
    The log(expm1(t)) expression below stays finite for large positive t.
    """
    t = power * torch.log1p(increment / a)
    log_gap = power * a.log() + t + torch.log(-torch.expm1(-t))
    return torch.logaddexp(power * c.log(), log_gap) / power


class ResidualPowerAnalogyEmbedding(LinearCompressedEmbedding):
    """Keep the exact linear table and add only non-arithmetic completion.

    A=softplus(x)+epsilon, B=A+softplus(y)+epsilon, C=softplus(z)+epsilon
    describe three positive terms. Complete D_p by A^p+D_p^p=B^p+C^p and
    append h_p=D_p-D_1 along a fixed complement of the initial linear span.
    At p=1 the ordinary linear function and its code gradients are retained;
    p itself still has a generally nonzero derivative on the first update.
    An optional identity-initialized mixer learns combinations within the same
    fixed complementary span without adding a new feature or output direction.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        pad_id: int,
        learn_mixing: bool = False,
    ) -> None:
        if not isinstance(learn_mixing, bool):
            raise ValueError("learn_mixing must be a boolean")
        # Same code/basis draws and scales as the linear control. Everything
        # added below is deterministic and consumes no additional random draws.
        super().__init__(vocab_size, d_model, pad_id, normalize_rows=False)
        self.learn_mixing = learn_mixing
        self.power_logits = nn.Parameter(torch.zeros(self.n_groups))
        with torch.no_grad():
            orthogonal, _ = torch.linalg.qr(self.basis.T, mode="complete")
        self.register_buffer("complement", orthogonal[:, self.code_dim :].T.contiguous())
        self.feature_mixing = nn.Parameter(torch.eye(self.n_groups)) if learn_mixing else None

    def powers(self) -> Tensor:
        # Zero offsets produce bitwise p=1 even after dtype changes. There is
        # no p==1 branch or closed output gate to suppress its derivative.
        origin = torch.full_like(self.power_logits, -math.log(4.0))
        return 1.0 + 1.25 * ((self.power_logits + origin).sigmoid() - origin.sigmoid())

    def _positive_inputs(self, codes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        raw = codes.reshape(*codes.shape[:-1], self.n_groups, 3) / self.code_scale
        positive = F.softplus(raw) + RADIUS_EPSILON
        return positive.unbind(dim=-1)

    def positive_features(self, codes: Tensor) -> Tensor:
        """The actual positive A, B, C, D quartet, before signed conversion."""
        a, increment, c = self._positive_inputs(codes)
        d = _ordered_log_completion(a, increment, c, self.powers()).exp()
        return torch.stack((a, a + increment, c, d), dim=-1)

    def residual_features(self, codes: Tensor) -> Tensor:
        a, increment, c = self._positive_inputs(codes)
        power = self.powers()
        log_d = _ordered_log_completion(a, increment, c, power)
        # Use the identical differentiable kernel at constant p=1. Detaching
        # this reference would change code gradients at the linear start.
        log_reference = _ordered_log_completion(a, increment, c, torch.ones_like(power))
        return log_reference.exp() * torch.expm1(log_d - log_reference)

    def _transform(self, codes: Tensor) -> Tensor:
        if self.feature_mixing is None:
            return codes @ self.basis + self.code_scale * (
                self.residual_features(codes) @ self.complement
            )
        features = self.residual_features(codes) @ self.feature_mixing
        return codes @ self.basis + self.code_scale * (features @ self.complement)

    def weight(self) -> Tensor:
        return self._transform(self.codes)

    def embed(self, token_ids: Tensor) -> Tensor:
        codes = F.embedding(token_ids, self.codes, padding_idx=self.pad_id)
        return self._transform(codes) * self.scale


class PowerAnalogyEmbedding(nn.Module):
    """Generate four positive power-analogy terms from three codes per token.

    A=R(1+u)^(1/p), B=R(1+v)^(1/p), C=R(1-v)^(1/p),
    D=R(1-u)^(1/p). Thus A^p+D^p=B^p+C^p=2R^p. Nonlinear
    features themselves, not features raised back to p, enter the Transformer.

    In normalized coordinates, initial raw radii are
    N(inverse_softplus(1-epsilon), 0.5^2), and contrasts are N(0,1).
    Stored codes are these coordinates times sigma=1/sqrt(d_model), keeping
    their update scale comparable to the ordinary embedding. The feature map
    divides by sigma before the radius chart. Fixed calibration uses this same
    normalized distribution at p=2, without reading vocabulary or task data.
    Optional row normalization acts only after the learned basis and preserves
    padding unchanged. It does not change the positive quartet construction.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        pad_id: int,
        normalize_rows: bool = False,
    ) -> None:
        super().__init__()
        _check_shape(vocab_size, d_model, pad_id)
        if not isinstance(normalize_rows, bool):
            raise ValueError("normalize_rows must be a boolean")
        self.normalize_rows = normalize_rows
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_groups = d_model // 4
        self.code_dim = 3 * self.n_groups
        self.pad_id = pad_id
        self.scale = math.sqrt(d_model)
        self.code_scale = d_model**-0.5
        codes = torch.randn(vocab_size, self.n_groups, 3)
        codes[..., 0].mul_(RADIUS_RAW_STD).add_(RADIUS_RAW_MEAN)
        codes.mul_(self.code_scale)
        self.codes = nn.Parameter(codes.reshape(vocab_size, self.code_dim))
        initial_logit = math.log((INITIAL_POWER - MIN_POWER) / (MAX_POWER - INITIAL_POWER))
        self.power_logits = nn.Parameter(torch.full((self.n_groups,), initial_logit))

        center, scale, whitening = _initial_feature_calibration()
        self.register_buffer("feature_center", center.to(dtype=self.codes.dtype).clone())
        self.register_buffer("feature_scale", scale.to(dtype=self.codes.dtype).clone())
        initial_basis = torch.block_diag(*([whitening] * self.n_groups)) / math.sqrt(d_model)
        self.basis = nn.Parameter(initial_basis.to(dtype=self.codes.dtype))
        with torch.no_grad():
            # R=center and u=v=0 generate four copies of the global center,
            # hence an initially zero padding embedding without deleting any
            # trainable code row or imposing a per-token zero-sum constraint.
            padding = self.codes[pad_id].view(self.n_groups, 3)
            padding.zero_()
            padding[:, 0] = self.code_scale * torch.log(
                torch.expm1(self.feature_center - RADIUS_EPSILON)
            )

    def powers(self) -> Tensor:
        return MIN_POWER + (MAX_POWER - MIN_POWER) * self.power_logits.sigmoid()

    def positive_features(self, codes: Tensor) -> Tensor:
        raw = codes.reshape(*codes.shape[:-1], self.n_groups, 3) / self.code_scale
        radius = F.softplus(raw[..., 0]) + RADIUS_EPSILON
        u = CONTRAST_LIMIT * raw[..., 1].tanh()
        v = CONTRAST_LIMIT * raw[..., 2].tanh()
        bases = torch.stack((1 + u, 1 + v, 1 - v, 1 - u), dim=-1)
        return radius[..., None] * bases.pow(self.powers()[:, None].reciprocal())

    def _transform(self, codes: Tensor) -> Tensor:
        features = self.positive_features(codes).flatten(start_dim=-2)
        return ((features - self.feature_center) / self.feature_scale) @ self.basis

    def weight(self) -> Tensor:
        weight = self._transform(self.codes)
        if self.normalize_rows:
            ids = torch.arange(self.vocab_size, device=weight.device)
            return _unit_rows_except_padding(weight, ids.eq(self.pad_id))
        return weight

    def embed(self, token_ids: Tensor) -> Tensor:
        codes = F.embedding(token_ids, self.codes, padding_idx=self.pad_id)
        weight = self._transform(codes)
        if self.normalize_rows:
            weight = _unit_rows_except_padding(weight, token_ids.eq(self.pad_id))
        return weight * self.scale

    def project(self, hidden: Tensor) -> Tensor:
        return F.linear(hidden, self.weight())
