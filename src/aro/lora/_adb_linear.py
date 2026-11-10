"""ADB-transpose-parameterised low-rank linear layer.

The effective weight is ``W_0 + A @ diag(D) @ B.T`` where ``W_0`` is a frozen
base weight (equal to the pretrained weight via :meth:`from_linear`), ``A``
is (out_features, rank) with orthonormal columns, ``B`` is (in_features,
rank) with orthonormal columns (both on the Stiefel manifold), and ``D``
is a free (rank,) signed diagonal. The name ``ADB`` mirrors the
multiplication order ``A @ diag(D) @ B.T``.

``D`` is optimized **freely** (no softplus / non-negativity constraint):
with orthonormal ``A`` and ``B``, a negative ``D_i`` is equivalent to
flipping the sign of column ``i`` of ``A`` or ``B``, so a
signed ``D`` is fully general and ``|D|`` are the singular values of the
adapter. ``D`` is initialized to ``d_init`` (default ``1e-3``) so the
adapter starts close to zero (finetuning: the layer begins near the
pretrained function) while staying off the ``sigma -> 0`` boundary that is
ill-conditioned for a fixed-rank retraction.

The forward pass computes the effective weight ``W_0 + (A * D) @ B.T``
directly. The trainable diagonal parameter is named ``d_raw`` (kept for
optimizer compatibility) and is used **directly** as ``D`` (no softplus);
``D`` is an identity alias. ``A``, ``B``, and ``d_raw`` are all frozen
(``requires_grad=False``): this layer targets an embedded fixed-rank
optimizer that treats the product ``A @ diag(D) @ B.T`` as a point on the
fixed-rank manifold -- capturing the ambient gradient from the activations
and retracting ``A``, ``D``, ``B`` manually -- so no factor gradients are
ever needed from autograd.
"""

import math
from typing import Any, Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LoRAADBLinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` with ADB low-rank factorization.

    The frozen weight ``W_0`` is stored as a parameter so FSDP can shard it.
    ``A`` (out_features, rank), ``d_raw`` (rank,), and ``B`` (in_features, rank) are the
    SVD-coordinate factors of the adapter ``W_adapter = A diag(D) B.T``:
    ``A`` has orthonormal columns (``U``), ``B`` orthonormal columns (``V``), and
    ``d_raw`` is the free signed diagonal ``D`` (``|D|`` are the singular
    values).

    Effective weight: ``W_0 + A diag(D) B.T``, shape (out_features,
    in_features). The adapter starts at ``|D| = d_init`` on all ``r``
    singular directions; with a small ``d_init`` (default ``1e-3``) the
    effective weight begins close to the base ``W_0`` (the pretrained weight
    via :meth:`from_linear`) while staying off the ``sigma -> 0`` boundary
    that is ill-conditioned for a fixed-rank retraction.

    The factors are **frozen** (``requires_grad = False``). This layer
    targets embedded fixed-rank optimization
    (:class:`~aro.lora.FixedRankRGD`), which updates the *product*
    ``W_adapter`` as a point on the fixed-rank manifold -- capturing the
    ambient gradient from the activations and retracting ``A``, ``D``, ``B``
    manually -- so no factor gradients are ever needed.
    """

    weight: nn.Parameter  # frozen base W_0 (== pretrained weight via from_linear)
    A: Tensor  # (out_features, rank), orthonormal columns
    d_raw: Tensor  # (rank,) free signed diagonal, used directly as D
    B: Tensor  # (in_features, rank), orthonormal columns
    _merged_weight: Tensor | None  # cached W_eff for eval (see merge_weight)
    bias: nn.Parameter | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = False,
        d_init: float = 1e-3,
        manual_factors: bool = False,
    ) -> None:
        super().__init__()
        # Require 2 * rank <= min(in, out). This is a layer-intrinsic storage
        # bound: the factors A, B cost r * (in + out), which exceeds a dense
        # in * out weight once r passes in*out/(in+out) -- exactly min/2 (i.e.
        # 2r = min) for a square layer -- so beyond it the adapter saves nothing
        # over full fine-tuning. It also matches the rank-2r tangent frame an
        # embedded fixed-rank retraction needs, and subsumes the basic
        # rank <= min(in, out) requirement of the SVD parameterization.
        if rank <= 0 or 2 * rank > min(in_features, out_features):
            raise ValueError(
                "LoRAADBLinear requires 0 < 2 * rank <= min(in_features, out_features), "
                + f"but got rank={rank} and min(in={in_features}, out={out_features})="
                + f"{min(in_features, out_features)}; use a smaller rank or full fine-tuning."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.weight = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)

        # Stiefel-manifold init: A and B have orthonormal columns
        # (A in St(out, rank), B in St(in, rank)).
        a = torch.empty(out_features, rank)
        b = torch.empty(in_features, rank)
        # Free signed diagonal (no softplus), used directly as D. Init to
        # ``d_init`` (default 1e-3 -> small singular values, so W_eff = W_0 + adapter
        # starts close to the base weight while staying off the sigma -> 0 boundary
        # that is ill-conditioned for a fixed-rank retraction).
        d_raw = torch.full((rank,), d_init)
        if manual_factors:
            self.register_buffer("A", a)
            self.register_buffer("d_raw", d_raw)
            self.register_buffer("B", b)
        else:
            self.A = nn.Parameter(a, requires_grad=False)
            self.d_raw = nn.Parameter(d_raw, requires_grad=False)
            self.B = nn.Parameter(b, requires_grad=False)
        nn.init.orthogonal_(self.A)
        nn.init.orthogonal_(self.B)

        if bias:
            fan_in = in_features
            bound = 1 / math.sqrt(fan_in)
            self.bias = nn.Parameter(
                torch.empty(out_features).uniform_(-bound, bound),
                requires_grad=False,
            )
        else:
            self.bias = None

        # Non-persistent so state_dict keeps the unfused representation. Registering
        # it as a buffer makes dtype/device moves update an active eval cache too.
        self.register_buffer("_merged_weight", None, persistent=False)
        self._adapter_enabled = True

    def set_adapter_enabled(self, enabled: bool) -> None:
        """Enable or disable the low-rank update."""
        self._adapter_enabled = enabled

    @property
    def D(self) -> Tensor:
        """Free signed diagonal ``D = d_raw`` (identity alias), shape (rank,)."""
        return self.d_raw

    def _adapter(self) -> Tensor:
        """Current adapter ``A @ diag(D) @ B.T`` (out_features, in_features)."""
        return (self.A * self.d_raw.unsqueeze(0)) @ self.B.T

    def _weight_eff(self) -> Tensor:
        """Effective weight ``W_0 + adapter`` (out_features, in_features)."""
        return self.weight + self._adapter()

    def merge_weight(self) -> None:
        """Cache ``W_eff`` so :meth:`forward` skips the per-call adapter rebuild.

        During generation ``forward`` runs once per decode token, so rebuilding the
        ``(out, in)`` ``W_eff`` (two low-rank products) every call dominates the cheap
        single-token linear. Precompute it once here and serve it as a plain linear.
        Call :meth:`unmerge_weight` before any further training so the (changing)
        adapter is used again -- a stale cache would otherwise freeze the weight.
        :class:`~aro.lora.FixedRankRGD` enforces that from its side: it refuses to
        be constructed on a merged module, captures no gradient from a forward
        served by the cache, and drops the cache whenever a step moves the factors.

        Computed under autocast keyed to the base ``weight`` dtype, then explicitly
        cast to that dtype. The layer does not impose a separate factor dtype policy:
        the factors follow ordinary module conversion rules. Matching the cache to the
        base dtype keeps the merged forward compatible with the rest of the layer.

        The merged and low-rank paths are algebraically equivalent but not bitwise
        identical in finite precision: folding the adapter into the weight changes the
        GEMM and accumulation order relative to applying the factors to activations.
        Expect differences within dtype rounding; use one path consistently when
        comparing evaluation results.
        """
        dt = self.weight.dtype
        use_ac = dt in (torch.float16, torch.bfloat16)
        with torch.no_grad(), torch.autocast(self.weight.device.type, dtype=dt, enabled=use_ac):
            self._merged_weight = self._weight_eff().to(dt)

    def unmerge_weight(self) -> None:
        """Drop the cached ``W_eff`` (revert to rebuilding the adapter per forward)."""
        self._merged_weight = None

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        self.unmerge_weight()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: Tensor) -> Tensor:
        if not self._adapter_enabled:
            return F.linear(x, self.weight, self.bias)
        # Merged (eval): serve the cached dense W_eff as a plain linear.
        if self._merged_weight is not None:
            if self.training:
                raise RuntimeError(
                    "LoRAADBLinear has a merged adapter while training; "
                    + "call unmerge_weight() before the training forward"
                )
            return F.linear(x, self._merged_weight, self.bias)
        # Training: apply the adapter in low-rank form rather than materializing the
        # dense (out, in) ``W_eff``. ``F.linear(x, W_eff)`` would build -- and, for the
        # grad_x backprop that carries the loss to earlier layers, *retain* -- a full
        # (out, in) copy of the weight per layer (~14 GB across an 8B model, since the
        # factors are frozen but grad still flows through W_eff to the input). The
        # low-rank path reuses the frozen base ``weight`` by reference and keeps only
        # the tiny (N, rank) intermediates. Inner ``F.linear`` calls take no bias
        # (bias=None default); the bias is applied once in ``base``.
        base = F.linear(x, self.weight, self.bias)
        adapter = F.linear(F.linear(x, self.B.T) * self.d_raw, self.A)
        return base + adapter

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        d_init: float = 1e-3,
        manual_factors: bool = False,
    ) -> Self:
        """Create a finetuning adapter from a pretrained ``nn.Linear``.

        Copies ``linear.weight`` into the frozen base ``W_0`` (exactly). The
        effective weight ``W_0 + A diag(D) B.T`` starts close to
        ``linear.weight`` for a small ``d_init`` (the frozen adapter has
        singular values ``d_init``), while staying off the ``sigma -> 0``
        boundary that is ill-conditioned for a fixed-rank retraction. The
        factors are frozen (an embedded fixed-rank optimizer updates them by
        a manual retraction).
        """
        lora = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            bias=linear.bias is not None,
            d_init=d_init,
            manual_factors=manual_factors,
        )
        lora.to(device=linear.weight.device, dtype=linear.weight.dtype)
        with torch.no_grad():
            # Base = pretrained weight (exact). The adapter (W_eff = base + adapter)
            # is applied on top, not baked into base.
            lora.weight.copy_(linear.weight.detach())
            if lora.bias is not None and linear.bias is not None:
                lora.bias.copy_(linear.bias.detach())
        # The bias is part of the frozen pretrained function (like W_0).
        if lora.bias is not None:
            lora.bias.requires_grad_(False)
        lora.train(linear.training)
        return lora
