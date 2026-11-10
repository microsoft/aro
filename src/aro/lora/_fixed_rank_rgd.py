"""Plain Riemannian gradient descent on the embedded fixed-rank manifold.

The LoRA adapter ``W = A diag(D) B^T`` is a point on the fixed-rank manifold
``M_r = {W : rank(W) = r}``, stored in SVD coordinates by
:class:`~aro.lora.LoRAADBLinear`: ``U = A`` (``out x r``, orthonormal columns),
``V = B`` (``in x r``, orthonormal columns) and the signed diagonal ``D``
(``|D|`` are the singular values).

Rather than optimizing the factors ``A, B`` (which carry a ``GL(r)`` gauge), this
updates the **product** ``W`` with the *embedded* Riemannian gradient, projected
onto the tangent space of ``M_r`` and retracted back onto the manifold by a
truncated SVD -- see ``_fixed_rank_geometry`` for the pure tensor math behind
the tangent projection, retraction, and vector transport reused below, and
``_fixed_rank_conditioning`` for the optional spectral conditioning that reshapes
the tangent step before it is retracted. This module supplies the hook-based
ambient-gradient capture and the :class:`FixedRankRGD` optimizer shell that
reads/writes the factors.

**Gradient capture (hooks, frozen factors).** ``A, D, B`` are frozen
(``requires_grad = False``); the ambient gradient is read straight from the
activations. For a linear layer ``G = gy^T x`` (output grad ``gy = dL/dy`` outer
input ``x``), so a forward hook captures ``x`` and hooks the output to capture
``gy`` on the backward pass, reducing them to the frame projections

    G V = gy^T (x B)   (out, r),   G^T U = x^T (gy A)   (in, r),

never forming ``G``. These accumulate across micro-batches into a per-module
stash and are consumed (and cleared) by ``step``. The core ``M = U^T G V`` is
then ``A^T (G V)`` -- exact, with no ``1/D`` recovery and no factor gradients.

Because the factors are frozen, the layer output only requires grad (so the
capture hook fires) if the *input* requires grad; the trainer must enable that on
the frozen embedding (e.g. ``model.enable_input_require_grads()``). Forwards that
do not require grad -- a validation loop under ``torch.no_grad()`` -- capture
nothing, so validation never pollutes the training gradient. Neither does a
forward served from a merged weight cache (see :meth:`LoRAADBLinear.merge_weight`),
which is a frozen snapshot of the factors rather than the live adapter.

**Activation memory.** Capturing ``x`` costs extra memory that a frozen linear
layer would not otherwise pay: because the base weight does not require grad,
autograd has no reason to save the layer input, so the hook's reference is what
keeps it alive, from the forward until that micro-batch's backward runs. Budget
one extra input activation per adapted layer -- ``N * in_features`` elements, e.g.
32 MiB for 4096 tokens x 4096 features in bf16 -- plus, transiently inside the
grad hook, fp32 copies of ``x`` and ``gy`` (``4 * N * (in + out)`` bytes) for the
frame projections, which run in fp32 so the retraction is stable regardless of
the autocast dtype. Under activation checkpointing the no-grad forward captures
nothing and only the recompute pass captures, so the retention window shrinks to
that segment's recompute-plus-backward. The projections themselves are tiny
(``(out, r)`` and ``(in, r)``) and are the only per-step state the stash holds.
"""

from collections.abc import Callable
from typing import Any, Literal
from weakref import finalize, ref

import torch
import torch.distributed as dist
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.hooks import RemovableHandle

from ._adb_linear import LoRAADBLinear
from ._fixed_rank_conditioning import FrameUpdate, _condition, _rms_normalized
from ._fixed_rank_geometry import _retract, _tangent_factored, _transport

try:  # ``torch.distributed`` (hence DTensor) is absent from some torch builds.
    from torch.distributed.tensor import DTensor

    _SHARDED_TYPES: tuple[type, ...] = (DTensor,)
except ImportError:  # pragma: no cover - depends on the torch build
    _SHARDED_TYPES = ()

type _Blocks = tuple[Tensor, Tensor, Tensor]
"""A triple of fp32 tensors: the master ``(U, D, V)`` or a velocity ``(M, U_p, V_p)``."""

_FP32_KEYS = ("master", "momentum", "frame")
"""Optimizer-state entries held in fp32 whatever the factors are.

``master`` and ``momentum`` are :obj:`_Blocks` triples; ``frame`` is a single
(min(out, in), 2r) conditioning frame.
"""

_CONDITIONING: tuple[str, ...] = ("none", "spectral")
"""Accepted values of the ``conditioning`` argument."""

_FRAME_UPDATES: tuple[FrameUpdate | None, ...] = (None, "qr", "polar")
"""Accepted values of the ``frame_update`` argument."""


def _remove_handles(handles: list[RemovableHandle]) -> None:
    """Remove every hook handle and empty the list (idempotent)."""
    for handle in handles:
        handle.remove()
    handles.clear()


def _fp32_tensor(value: Tensor, device: torch.device, copy: bool = False) -> Tensor:
    """Return one stored tensor as fp32 on ``device`` (see :func:`_fp32_blocks`)."""
    return value.to(device=device, dtype=torch.float32, copy=copy)


def _fp32_blocks(values: Any, device: torch.device, copy: bool = False) -> _Blocks:
    """Return the three stored blocks as fp32 tensors on ``device``.

    ``Tensor.to`` returns ``self`` when nothing changes, so this is free on the hot
    path and repairs state that is no longer fp32 on the module's device -- which
    happens whenever a module is moved after the optimizer was built. Pass
    ``copy=True`` to force fresh storage (loading state must not alias the
    caller's checkpoint tensors).
    """
    x, y, z = values
    return (
        _fp32_tensor(x, device, copy),
        _fp32_tensor(y, device, copy),
        _fp32_tensor(z, device, copy),
    )


def _check_supported(module: LoRAADBLinear) -> None:
    """Reject adapters whose factors this optimizer cannot own and retract.

    Raises:
        ValueError: If the module holds a merged weight cache.
        TypeError: If a factor is sharded (DTensor / FSDP ``FlatParameter``).
    """
    if module._merged_weight is not None:
        raise ValueError(
            "LoRAADBLinear has a merged weight cache, which a step would immediately "
            + "make stale (the cache is a snapshot of the factors, and forwards served "
            + "from it contribute no gradient). Call unmerge_weight(), or leave the "
            + "merged_adapters() block, before constructing FixedRankRGD."
        )
    for name in ("A", "d_raw", "B"):
        factor = getattr(module, name)
        # FSDP1's FlatParameter lives in a private module, so match it by name rather
        # than importing an unstable path; FSDP2 and tensor parallel use DTensor.
        if isinstance(factor, _SHARDED_TYPES) or type(factor).__name__ == "FlatParameter":
            raise TypeError(
                f"FixedRankRGD needs plain replicated factors, but {name} is a "
                + f"{type(factor).__name__}. The retraction is a global SVD of the whole "
                + "adapter and the hooks capture unsharded activations, so keep the "
                + "adapters replicated (DDP) and exclude them from FSDP/tensor-parallel "
                + "sharding."
            )


def _check_conditioning(
    conditioning: str,
    sinkhorn_steps: int,
    tau: float,
    frame_update: FrameUpdate | None,
    track_frame: bool,
) -> None:
    """Reject conditioning arguments that are out of range or have no effect.

    A knob that cannot reach the update is an error rather than a silent no-op, so a
    run never quietly optimizes something other than what was configured. ``tau`` is
    the exception: its useful default is non-zero, so an explicit ``tau`` cannot be
    told apart from the default one and it is only range-checked here -- under
    ``conditioning="none"`` it is simply ignored, like every other spectral-only knob
    would be if it were not rejected.

    Raises:
        ValueError: If a value is out of range, if a conditioning knob is set while
            ``conditioning="none"``, if ``track_frame`` has no ``frame_update`` to
            iterate, or for the ``"qr"`` + ``track_frame`` + ``sinkhorn_steps=0``
            combination, whose fixed point is the spectral basis that is already
            recomputed exactly every step (use ``"polar"``, or drop ``track_frame``).
    """
    if conditioning not in _CONDITIONING:
        raise ValueError(f"conditioning must be one of {_CONDITIONING}, got {conditioning!r}")
    if sinkhorn_steps < 0:
        raise ValueError(f"sinkhorn_steps must be >= 0, got {sinkhorn_steps}")
    if not 0.0 <= tau < 1.0:
        raise ValueError(f"tau must be in [0, 1), got {tau}")
    if frame_update not in _FRAME_UPDATES:
        raise ValueError(f"frame_update must be one of {_FRAME_UPDATES}, got {frame_update!r}")
    if conditioning == "none":
        unusable = [
            name
            for name, is_set in (
                ("sinkhorn_steps", sinkhorn_steps > 0),
                ("frame_update", frame_update is not None),
                ("track_frame", track_frame),
            )
            if is_set
        ]
        if unusable:
            raise ValueError(
                f"{', '.join(unusable)} only applies to conditioning='spectral', but "
                + "conditioning='none' was requested; pass conditioning='spectral' to "
                + "condition the step, or drop the argument."
            )
        return
    if track_frame and frame_update is None:
        raise ValueError("track_frame requires a frame_update ('qr' or 'polar')")
    if track_frame and frame_update == "qr" and sinkhorn_steps == 0:
        raise ValueError(
            "track_frame with frame_update='qr' needs sinkhorn_steps > 0: without the "
            + "column balance the update is a power iteration whose fixed point is the "
            + "spectral basis, which is already recomputed exactly on every step, so a "
            + "tracked estimate of it is strictly worse. Use frame_update='polar', or "
            + "drop track_frame."
        )


class FixedRankRGD(Optimizer):
    """Riemannian gradient descent on the embedded fixed-rank manifold.

    Takes :class:`~aro.lora.LoRAADBLinear` adapters as whole modules and updates
    the product ``W = A diag(D) B^T`` as a single point on ``M_r`` (embedded
    metric), writing the retracted ``U, D, V`` back into ``A``, ``d_raw``, ``B``.
    The factors are frozen and the ambient gradient is captured from the
    forward/backward activations by hooks installed at construction. Call
    :meth:`remove_hooks` when the optimizer is discarded; it is idempotent, and a
    finalizer runs it anyway once the optimizer is collected, so a forgotten call
    cannot leave live hooks capturing into a stash nobody consumes.

    The retraction is a truncated SVD, so the written-back ``D`` holds singular
    values (non-negative, descending) -- the signed freedom ``LoRAADBLinear``
    allows in ``D`` is absorbed into the frames ``A`` / ``B``.

    **Merged weight caches.** A cache from :meth:`LoRAADBLinear.merge_weight` is a
    snapshot of the factors, so it is stale the moment a step retracts them:
    constructing on an already-merged module is an error, forwards served from a
    cache contribute no gradient, and a step drops any cache that appeared since.

    **Master state (precision).** The point is carried in fp32 optimizer state
    (``U, D, V``) and the module's factors are a write-back mirror of it. With
    bf16/fp16 factors a step is typically smaller than one ulp of ``D``, so
    re-reading the factors each step would round every update away; the fp32
    master accumulates them instead and the mirror simply renders the current
    point at the factor dtype. The master is seeded from the factors on first use,
    so external writes to them (loading a checkpoint into the *module* after the
    optimizer was built) need :meth:`resync_from_modules` to take effect.

    **Spectral conditioning (opt-in).** With ``conditioning="spectral"`` the tangent
    step is reshaped before the retraction: it is reduced to its coefficients in an
    orthonormal ``2r``-frame, its spectrum is whitened Muon-style with a rank floor,
    optionally Sinkhorn-balanced over the reduced columns, and projected back onto
    the tangent space (see ``_fixed_rank_conditioning``). The frame
    uses the smaller dimension, choosing the input side when ``in <= out``.
    Input-side conditioning operates on the transposed tangent and remaps the
    result back before RMS normalization and retraction. This shapes the *step only* --
    the raw Riemannian velocity is still what the momentum buffer accumulates and
    transports -- so the conditioning is never baked into the running average. The
    default ``conditioning="none"`` skips the spectral transform; with
    ``rms_normalize=False`` (also the default), the step is exactly plain Riemannian
    gradient descent. Plain mode rejects every spectral knob that would
    have no effect (except ``tau``, whose non-zero default makes an explicit value
    indistinguishable; it is simply ignored). Because a whitened step's magnitude is
    decoupled from the gradient norm, ``lr`` generally needs retuning when conditioning
    is switched on. Whitening-only and Sinkhorn outputs need not have the same
    norm, so changing ``sinkhorn_steps`` can also require retuning ``lr``.

    **Distributed (replicated factors).** Every rank must hold the same
    replicated factors. Different modules may live on different devices, but each
    module's factors must be colocated and corresponding modules on every rank must
    use devices supported by the process-group backend. ``step`` all-reduces the two
    frame projections per module, so all ranks retract identical inputs from
    identical factors and stay in sync. FSDP2 may shard the frozen base weights;
    exclude ``A``, ``d_raw``, and ``B`` with ``fully_shard(..., ignored_params=...)``
    at every sharding level, and synchronize their initial values across ranks.
    Move ignored factors to the compute device yourself: FSDP does not move them.
    Construct the optimizer after sharding. Sharded factors (FSDP, tensor parallel)
    are rejected at construction. Ranks reach the same collectives in the same order: a
    rank that captured nothing contributes zeros rather than skipping the module. Conditioning
    adds no collectives -- it is a deterministic function of the reduced gradient and
    the replicated state, including any tracked frame, so the ranks stay identical.

    **Loss scaling (AMP).** The factors never carry a ``.grad``, so
    :class:`torch.amp.GradScaler` cannot unscale or inf-check this optimizer --
    never pass it to ``scaler.step``. Report the scale to :meth:`step` instead and
    manage the scaler's next scale explicitly using :attr:`found_non_finite`.
    If any adapter has a
    non-finite projection, the whole optimizer step is skipped, matching
    ``GradScaler``'s optimizer-level overflow semantics. When this is the only
    optimizer, ``scaler.update()`` cannot infer overflow or growth from factor
    gradients: pass an explicit ``new_scale`` on both successful and skipped
    steps, tracking successful steps for the scaler's growth interval yourself.

    Args:
        adb_modules: The ADB adapters to optimize. Each module's factors must be
            colocated and must not hold a merged weight cache; different modules may
            live on different devices.
        lr: Learning rate (Riemannian step size).
        momentum: Riemannian heavy-ball coefficient in ``[0, 1)`` (0 disables).
            The velocity is a tangent vector, transported across steps by
            projection onto the current tangent space (see :meth:`step`).
        conditioning: ``"none"`` (default) retracts the raw Riemannian step;
            ``"spectral"`` conditions it first (whitening, and the knobs below). No
            spectral conditioning is ever applied implicitly. ``sinkhorn_steps``,
            ``frame_update``, and ``track_frame`` require ``"spectral"``; ``tau`` is
            ignored in plain mode. ``rms_normalize`` independently rescales either
            a plain or a spectral step.
        sinkhorn_steps: Reduced Sinkhorn sweeps under ``conditioning="spectral"``.
            ``0`` (default) whitens the spectrum only (Muon); a positive value
            additionally balances the reduced columns, alternating that many times.
        tau: Rank floor of the spectral whitening, relative to the largest singular
            value of the step: directions at or above ``tau * sigma_max`` whiten
            approximately to unit, with an additive epsilon in the denominator.
            Weaker directions are unchanged within each row pass but still enter
            column normalization. Spectral-only, like the knobs above, but silently
            ignored under ``conditioning="none"`` rather than rejected: its default
            is non-zero, so "explicitly set" cannot be told apart from "left alone".
        frame_update: Procrustes refinement of the conditioning frame, applied before
            the step is conditioned in it. ``None`` (default) conditions in the fresh
            spectral basis. ``"qr"`` is the whitening step, whose fixed point without
            Sinkhorn is that same basis; ``"polar"`` is the Stiefel polar step, which
            ascends the alignment between the step and the gradient.
        track_frame: Carry the conditioning frame across steps instead of re-deriving
            the spectral basis each time, so it accumulates toward the fixed point of
            ``frame_update`` (which it therefore requires). The whole
            (min(out, in), 2r) frame is carried on the selected conditioning side,
            which keeps the tracking gauge-coherent, and it is
            bootstrapped from the spectral basis on the first step. The carried frame
            must remain finite with orthonormal columns. It is passed directly to
            QR/polar refinement without an extra repair or a host-side finiteness
            check. Experimental output-only frames from earlier versions of this
            integration must be discarded before resuming layers with ``in <= out``,
            including square layers; no automatic frame conversion is performed.
        rms_normalize: Rescale the (conditioned) step to unit entrywise RMS, i.e.
            ``||step||_F = sqrt(out * in)``, before the retraction. A pure per-layer
            shape rescale that decouples the step size from the gradient norm; like
            the conditioner it shapes the step only, never the transported velocity.
            Its ``lr`` is *not* interchangeable with a full-rank Muon / AdamW ``lr``:
            a tangent step has rank at most ``2r``, so that Frobenius budget lands on
            those few directions and inflates each singular value by roughly
            ``sqrt(min(out, in) / rank)`` relative to a full-rank step of the same
            norm. Tune ``lr`` per shape and rank.
        eps: Floor keeping the singular values strictly positive after each step.
        process_group: Process group for the gradient all-reduce (default: the
            default group). Only used when ``torch.distributed`` is initialized.
        gradient_divisor: Divisor applied to the captured (and, when distributed,
            all-reduced) ambient gradient. Defaults to 1 in a single process and to
            the group's world size when distributed, i.e. the reduction *averages*
            per-rank ambient gradients, matching a loss each rank averages over its
            local batch. Set it explicitly for any other convention -- e.g. a loss
            summed over a global token count, or gradient accumulation that sums
            micro-batch losses -- and it then applies in single-process runs too.
    """

    def __init__(
        self,
        adb_modules: list[LoRAADBLinear],
        *,
        lr: float = 1e-3,
        momentum: float = 0.0,
        conditioning: Literal["none", "spectral"] = "none",
        sinkhorn_steps: int = 0,
        tau: float = 1e-4,
        frame_update: FrameUpdate | None = None,
        track_frame: bool = False,
        rms_normalize: bool = False,
        eps: float = 1e-8,
        process_group: dist.ProcessGroup | None = None,
        gradient_divisor: float | None = None,
    ) -> None:
        if not adb_modules:
            raise ValueError("adb_modules must not be empty")
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if gradient_divisor is not None and gradient_divisor <= 0.0:
            raise ValueError(f"gradient_divisor must be > 0, got {gradient_divisor}")
        _check_conditioning(conditioning, sinkhorn_steps, tau, frame_update, track_frame)
        for module in adb_modules:
            _check_supported(module)
            factor_devices = {module.A.device, module.d_raw.device, module.B.device}
            if len(factor_devices) != 1:
                found = ", ".join(sorted(str(device) for device in factor_devices))
                raise ValueError(f"each ADB module must use one factor device, but found {found}")
        self._adb_modules = list(adb_modules)
        self._conditioning: Literal["none", "spectral"] = conditioning
        self._sinkhorn_steps = sinkhorn_steps
        self._tau = tau
        self._frame_update: FrameUpdate | None = frame_update
        self._track_frame = track_frame
        self._rms_normalize = rms_normalize
        self._process_group = process_group
        self._gradient_divisor = gradient_divisor
        self._found_non_finite = False
        self._capture_active = True

        # Register the (frozen) factors so the param group carries ``lr`` for the
        # scheduler; the update itself writes them back manually in ``step``. The
        # ``LoRAADBLinear`` layer already froze them and set ``D = d_init > 0``.
        # These tensors are also the optimizer's *anchors*: they keep identifying
        # their module's state and param group even after a ``module.to(...)`` that
        # swaps the module's tensors (buffer moves rebind, they do not mutate in
        # place, so ``manual_factors=True`` factors are new objects afterwards).
        self._anchors = [(m.A, m.d_raw, m.B) for m in self._adb_modules]
        super().__init__(
            [factor for anchor in self._anchors for factor in anchor],
            {"lr": lr, "eps": eps, "momentum": momentum},
        )

        # Per-module accumulators for the frame-projected ambient gradient,
        # [G V (out, r), G^T U (in, r)], filled by the hooks and cleared by step.
        self._stash: list[list[Tensor | None]] = [[None, None] for _ in self._adb_modules]
        self._handles: list[RemovableHandle] = []
        for index, module in enumerate(self._adb_modules):
            self._handles.append(module.register_forward_hook(self._make_forward_hook(index)))
        # The optimizer owns the modules and each module owns a capture hook, so the
        # hook closures only hold a *weak* reference back (see _make_forward_hook) and
        # this finalizer detaches them if the optimizer is dropped without
        # ``remove_hooks``; otherwise the pair would keep each other alive and the
        # stash would keep growing under an owner that can no longer consume it.
        self._finalizer = finalize(self, _remove_handles, self._handles)

    @property
    def found_non_finite(self) -> bool:
        """Whether the last :meth:`step` was skipped for a non-finite gradient.

        Rank-consistent: the check reads the all-reduced gradient (a non-finite
        contribution on any rank poisons the sum on every rank).
        """
        return self._found_non_finite

    def _make_forward_hook(self, index: int) -> Callable[..., None]:
        """Build a forward hook that captures the input and hooks the output grad.

        The closures keep only a weak reference to the optimizer: the hook lives on
        the module, the optimizer owns the module, and a strong reference here would
        make the two mutually immortal. A capture whose optimizer has been collected
        is dropped.
        """
        optimizer = ref(self)

        def forward_hook(
            module: LoRAADBLinear, inputs: tuple[Tensor, ...], output: Tensor
        ) -> None:
            opt = optimizer()
            if opt is None or not opt._capture_active:
                return
            # Nothing to capture unless this forward is part of the training function
            # and actually used the live factors: no grad (e.g. a validation forward
            # under ``torch.no_grad``, or an input that does not require grad), a
            # disabled adapter, or a forward served from a merged weight cache.
            if not output.requires_grad or not module._adapter_enabled:
                return
            if module._merged_weight is not None:
                return
            # Flatten any leading (batch, seq, ...) dims to a single sample axis.
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])  # (N, in)

            def grad_hook(grad_output: Tensor) -> None:
                opt = optimizer()
                if opt is None or not opt._capture_active:
                    return
                gy = grad_output.reshape(-1, grad_output.shape[-1])  # (N, out)
                with torch.no_grad():
                    # Linear algebra in fp32 for a stable retraction, regardless of the
                    # autocast/compute dtype of the activations. The frames come from
                    # the (possibly low-precision) factors rather than the fp32 master:
                    # they are the frames this forward actually used, and their rounding
                    # is dominated by the activation dtype anyway.
                    xf = x.float()
                    gyf = gy.float()
                    a32 = module.A.float()
                    b32 = module.B.float()
                    gv = gyf.transpose(-2, -1) @ (xf @ b32)  # (out, r) = G V
                    gtu = xf.transpose(-2, -1) @ (gyf @ a32)  # (in, r) = G^T U
                    slot = opt._stash[index]
                    slot[0] = gv if slot[0] is None else slot[0] + gv
                    slot[1] = gtu if slot[1] is None else slot[1] + gtu

            output.register_hook(grad_hook)

        return forward_hook

    def _reduced_gradients(self, grad_scale: float | None) -> list[tuple[Tensor, Tensor] | None]:
        """Pop, reduce and rescale every module's accumulated ``(G V, G^T U)``.

        An entry is ``None`` where :meth:`step` must leave the module completely
        alone -- factors, master and velocity untouched, in particular *no* momentum
        decay. That happens when no gradient was captured for it anywhere or its
        reduced projections are exactly zero. A non-finite projection sets
        :attr:`found_non_finite`, causing :meth:`step` to discard every entry and
        leave the whole optimizer untouched.

        Under an initialized process group every rank must reach the same collectives
        in the same order, so a locally missing gradient contributes zeros instead of
        skipping the module: with a gradient on some other rank the reduction is a
        genuine partial-batch gradient, and with a gradient on no rank it reduces to
        exactly zero, which is how the "nothing captured" skip stays identical to the
        single-process one. Both skip predicates read the reduced tensors, which agree
        across ranks, so the ranks never diverge.
        """
        distributed = dist.is_initialized()
        divisor = self._gradient_divisor
        if divisor is None:
            divisor = float(dist.get_world_size(self._process_group)) if distributed else 1.0
        if grad_scale is not None:
            divisor *= grad_scale

        reduced: list[tuple[Tensor, Tensor] | None] = []
        for index, module in enumerate(self._adb_modules):
            gv, gtu = self._stash[index]
            self._stash[index] = [None, None]
            if gv is None or gtu is None:
                if not distributed:
                    reduced.append(None)
                    continue
                device, r = module.A.device, module.rank
                if gv is None:
                    gv = torch.zeros(module.out_features, r, device=device, dtype=torch.float32)
                if gtu is None:
                    gtu = torch.zeros(module.in_features, r, device=device, dtype=torch.float32)
            if distributed:
                dist.all_reduce(gv, group=self._process_group)
                dist.all_reduce(gtu, group=self._process_group)
            if divisor != 1.0:
                gv = gv.div(divisor)
                gtu = gtu.div(divisor)
            reduced.append((gv, gtu))
        return self._drop_unusable(reduced)

    def _drop_unusable(
        self, reduced: list[tuple[Tensor, Tensor] | None]
    ) -> list[tuple[Tensor, Tensor] | None]:
        """Blank out the gradients ``step`` must skip, in one host synchronization.

        A non-finite gradient marks the whole optimizer step as unusable -- an fp16
        backward overflowed, so the caller is expected to lower the loss scale (see
        the class docstring). An exactly-zero gradient is dropped only for its module:
        this deliberate distributed convention keeps missing-gradient ranks aligned
        with single-process semantics, while also treating exact cross-rank
        cancellation as no update. Both flags are computed for every module first and
        read back together, so the whole step costs a single device-to-host
        synchronization.
        """
        pending = [(index, pair) for index, pair in enumerate(reduced) if pair is not None]
        if not pending:
            return reduced
        # Modules may live on different devices. Move each two-flag summary to one
        # device before the final stack so only scalar status crosses devices.
        device = pending[0][1][0].device
        flags = torch.stack(
            [
                torch.stack(
                    (
                        gv.isfinite().all() & gtu.isfinite().all(),
                        gv.any() | gtu.any(),
                    )
                ).to(device)
                for _, (gv, gtu) in pending
            ]
        ).tolist()
        for (index, _), (finite, nonzero) in zip(pending, flags, strict=True):
            if not finite:
                self._found_non_finite = True
            if not finite or not nonzero:
                reduced[index] = None
        return reduced

    def _master(self, index: int, state: dict[str, Any]) -> _Blocks:
        """The module's fp32 master ``(U, D, V)``, seeded from its factors on first use."""
        module = self._adb_modules[index]
        stored = state.get("master")
        if stored is None:
            # ``copy=True`` throughout: the master must never alias the factors it
            # mirrors, which a no-op ``.float()`` / ``.contiguous()`` would leave it
            # doing (e.g. for already-fp32 factors, or a rank-1 transpose).
            master = (
                module.A.detach().to(dtype=torch.float32, copy=True),
                module.d_raw.detach().to(dtype=torch.float32, copy=True),
                module.B.detach().to(dtype=torch.float32, copy=True),
            )
        else:
            master = _fp32_blocks(stored, module.A.device)
        state["master"] = master
        return master

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], float] | None = None,
        *,
        grad_scale: float | None = None,
    ) -> None:
        """One embedded Riemannian-gradient-descent step on every adapter.

        Consumes the ambient gradients the hooks captured since the last step (see
        :meth:`_reduced_gradients` for what is skipped) and writes the retracted
        point to the fp32 master and to the module's factors, dropping any weight
        cache that was merged in the meantime, since the factors just moved.

        Args:
            closure: Not supported; must be ``None``. The step is driven by gradients
                captured during ``backward``, which re-evaluating the model here would
                not refresh.
            grad_scale: The factor the loss was multiplied by before ``backward``
                (``scaler.get_scale()`` under AMP). Divided out of the captured
                gradients on top of ``gradient_divisor``; ``None`` means an unscaled
                loss. Because the captured projections are fp32, a large scale is safe
                through the all-reduce -- only a backward that already overflowed in
                fp16 produces the non-finite values :attr:`found_non_finite` reports.

        Raises:
            ValueError: If ``closure`` is not ``None`` or ``grad_scale`` is not > 0.
        """
        if closure is not None:
            raise ValueError(
                "FixedRankRGD does not support closures: a step consumes the ambient "
                + "gradients its hooks captured during backward, so re-evaluating the "
                + "model inside step would not refresh them. Call backward(), then step()."
            )
        if grad_scale is not None and grad_scale <= 0.0:
            raise ValueError(f"grad_scale must be > 0, got {grad_scale}")
        self._found_non_finite = False
        reduced_gradients = self._reduced_gradients(grad_scale)
        if self._found_non_finite:
            return
        groups: dict[Tensor, dict[str, Any]] = {
            p: group for group in self.param_groups for p in group["params"]
        }
        for index, reduced in enumerate(reduced_gradients):
            if reduced is not None:
                self._update(index, reduced, groups)

    def _update(
        self,
        index: int,
        reduced: tuple[Tensor, Tensor],
        groups: dict[Tensor, dict[str, Any]],
    ) -> None:
        """Retract one module along its reduced ambient gradient."""
        module = self._adb_modules[index]
        anchor = self._anchors[index][0]
        group = groups[anchor]
        state = self.state[anchor]
        lr, eps, mu = group["lr"], group["eps"], group["momentum"]
        gv, gtu = reduced
        # SVD coordinates of the current point, in fp32: U, D, V (the module's
        # factors are the low-precision mirror of these).
        u, df, v = self._master(index, state)

        m, u_p, v_p = _tangent_factored(gv, gtu, u, v)

        # Riemannian heavy-ball momentum (EMA form). Writing the velocity as
        # "vel" (a tangent vector; U/V/U1/V1 always denote frames):
        #   vel_t = mu * P_T(vel_{t-1}) + (1-mu) * P_T(G_t).
        # The stored velocity is already transported to the CURRENT frames (done
        # at the end of the previous step), so P_T(vel_{t-1}) is exactly the
        # stored decomposition and the combine is componentwise (the
        # decomposition is linear). The (1-mu) weight makes vel a true EMA of
        # the transported Riemannian gradients, decoupling the update magnitude
        # -- and lr scale -- from mu. Zero-init (vel_0 = 0): the first step is
        # vel_1 = (1-mu) G_1, so -- unlike clone-the-first-gradient init -- the
        # velocity never anchors to G_1, keeping a given mu comparable to Adam
        # and other zero-initialized EMAs. No Adam-style 1/(1-mu^t) bias
        # correction: it only rescales the first ~1/(1-mu) warmup steps.
        if mu > 0.0:
            buf = state.get("momentum")
            if buf is None:  # zero-init: vel_0 = 0  ->  vel_1 = (1-mu) G_1
                m_vel = torch.zeros_like(m)
                up_vel = torch.zeros_like(u_p)
                vp_vel = torch.zeros_like(v_p)
            else:  # velocity decomposition, already perp to the current U / V
                m_vel, up_vel, vp_vel = _fp32_blocks(buf, m.device)
            m = mu * m_vel + (1.0 - mu) * m
            u_p = mu * up_vel + (1.0 - mu) * u_p
            v_p = mu * vp_vel + (1.0 - mu) * v_p

        step = self._step_direction(state, m, u_p, v_p, u, v)
        u_new, d_new, v_new = _retract(u, df, v, *step, lr, eps)

        # Transport this step's velocity onto the retracted point's frames and
        # store its compact decomposition there, so the next step needs no
        # transport (the stored blocks are already perp to the current frames).
        # This is the *raw* velocity: any conditioning above shaped the retracted
        # direction only and must not leak into the running average.
        if mu > 0.0:
            state["momentum"] = _transport(u, v, u_new, v_new, m, u_p, v_p)

        state["master"] = (u_new, d_new, v_new)
        module.A.copy_(u_new)  # ``copy_`` casts to the factor dtype
        module.B.copy_(v_new)
        module.d_raw.copy_(d_new)
        if module._merged_weight is not None:
            module.unmerge_weight()

    def _step_direction(
        self,
        state: dict[str, Any],
        m: Tensor,
        u_p: Tensor,
        v_p: Tensor,
        u: Tensor,
        v: Tensor,
    ) -> _Blocks:
        """Shape the raw Riemannian velocity into the direction actually retracted.

        Applies the requested spectral conditioning and RMS rescale out of place, so
        the caller's blocks -- which are what the momentum buffer transports -- are
        left untouched. Spectral conditioning uses the smaller side (input on ties),
        with input-side ties; input-side blocks are transposed and swapped, then remapped
        before RMS normalization. When the frame is tracked it is read from (and written back
        to) ``state`` in fp32 on the point's device, so it survives a module move and
        a checkpoint round trip like the rest of the state. The frame must be finite
        with orthonormal columns; the conditioner uses it directly for refinement.
        """
        if self._conditioning == "spectral":
            precond_input = v.shape[-2] <= u.shape[-2]
            if precond_input:
                m, u_p, v_p, u, v = m.transpose(-2, -1), v_p, u_p, v, u
            frame = state.get("frame") if self._track_frame else None
            m, u_p, v_p, frame = _condition(
                m,
                u_p,
                v_p,
                u,
                v,
                frame=None if frame is None else _fp32_tensor(frame, u.device),
                sinkhorn_steps=self._sinkhorn_steps,
                tau=self._tau,
                frame_update=self._frame_update,
            )
            if self._track_frame:
                state["frame"] = frame
            if precond_input:
                m, u_p, v_p = m.transpose(-2, -1), v_p, u_p
        if self._rms_normalize:
            m, u_p, v_p = _rms_normalized(m, u_p, v_p)
        return m, u_p, v_p

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load optimizer state, keeping the fp32 entries in fp32.

        ``Optimizer.load_state_dict`` casts floating-point state to each parameter's
        dtype, which would round the fp32 master, velocity and conditioning frame down
        to bf16/fp16 factors on every resume and undo the point of keeping a master
        copy. Load everything else through the base implementation, then re-attach
        those entries from the checkpoint unrounded.
        """
        saved: dict[Any, dict[str, Any]] = state_dict["state"]
        stripped = {
            key: {name: value for name, value in entry.items() if name not in _FP32_KEYS}
            for key, entry in saved.items()
        }
        super().load_state_dict({**state_dict, "state": stripped})
        # Re-attach them against the same saved-id -> parameter mapping the base class
        # uses (ids run in param-group order); an id it cannot map keeps its own key,
        # just as the base class leaves such state untouched.
        params = [p for group in self.param_groups for p in group["params"]]
        saved_ids = [i for group in state_dict["param_groups"] for i in group["params"]]
        id_map = dict(zip(saved_ids, params, strict=True))
        for key, entry in saved.items():
            target = id_map.get(key, key)
            for name in _FP32_KEYS:
                stored = entry.get(name)
                if stored is None:
                    continue
                fallback = stored.device if isinstance(stored, Tensor) else stored[0].device
                device = target.device if isinstance(target, Tensor) else fallback
                self.state[target][name] = (
                    _fp32_tensor(stored, device, copy=True)
                    if isinstance(stored, Tensor)
                    else _fp32_blocks(stored, device, copy=True)
                )

    def resync_from_modules(self) -> None:
        """Re-seed the fp32 master from the modules' current factors.

        The master is authoritative between steps, so call this after writing the
        factors from outside the optimizer (e.g. ``module.load_state_dict`` on a
        resume that does not also restore this optimizer's state). The momentum
        velocity is dropped as well: it is a tangent vector at the point being
        discarded, and transporting it would need the frames that just went away. So
        is any tracked conditioning frame, which the next step re-bootstraps from the
        spectral basis of the new point.
        """
        for anchor, _, _ in self._anchors:
            self.state.pop(anchor, None)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear param grads (a no-op on the frozen factors) and the gradient stash."""
        super().zero_grad(set_to_none=set_to_none)
        self._stash = [[None, None] for _ in self._adb_modules]

    def remove_hooks(self) -> None:
        """Detach the capture hooks and drop any captured gradients (idempotent)."""
        self._capture_active = False
        self._finalizer.detach()
        _remove_handles(self._handles)
        self._stash = [[None, None] for _ in self._adb_modules]
