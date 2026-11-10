"""Tests for :class:`aro.lora.FixedRankRGD` (embedded fixed-rank Riemannian GD).

The optimizer never forms the ``out x in`` ambient gradient, so every check here
compares the factored update against an explicit dense reference built from
``G = gy^T x``. See ``test_fixed_rank_geometry.py`` for the tangent projection,
retraction, and transport helpers in isolation; this file covers the gradient
capture hooks and their lifecycle, the optimizer step, gradient scaling (the
divisor and AMP loss scale), the fp32 master state, momentum, state dict
round-tripping, argument validation, and the distributed all-reduce.
"""

import copy
import gc
import os
import socket
from collections.abc import Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.testing
from torch import Tensor

from aro.lora import FixedRankRGD, LoRAADBLinear, disabled_adapters
from aro.lora._fixed_rank_geometry import _retract, _tangent_factored

IN_FEATURES = 8
OUT_FEATURES = 6
RANK = 2


def _layer(
    in_features: int = IN_FEATURES,
    out_features: int = OUT_FEATURES,
    rank: int = RANK,
    d: list[float] | None = None,
    seed: int = 0,
    manual_factors: bool = False,
) -> LoRAADBLinear:
    """ADB layer with a well-separated spectrum (avoids SVD subspace ambiguity)."""
    torch.manual_seed(seed)
    layer = LoRAADBLinear(
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        manual_factors=manual_factors,
    )
    with torch.no_grad():
        layer.d_raw.copy_(torch.tensor(d if d is not None else [1.0, 0.5][:rank]))
    return layer


def _frames(layer: LoRAADBLinear) -> tuple[Tensor, Tensor, Tensor]:
    """Current SVD coordinates ``(U, D, V)`` with ``U = A``, ``V = B``, in fp32."""
    return (
        layer.A.detach().float().clone(),
        layer.d_raw.detach().float().clone(),
        layer.B.detach().float().clone(),
    )


def _adapter(layer: LoRAADBLinear) -> Tensor:
    """Dense adapter ``A diag(D) B.T`` (out, in), built at the factor dtype then upcast."""
    return ((layer.A * layer.d_raw) @ layer.B.T).detach().float()


def _dense_tangent(g: Tensor, u: Tensor, v: Tensor) -> Tensor:
    """Dense tangent projection ``P_U G + G P_V - P_U G P_V``."""
    pu = u @ u.transpose(-2, -1)
    pv = v @ v.transpose(-2, -1)
    return pu @ g + g @ pv - pu @ g @ pv


def _truncate(w: Tensor, rank: int) -> Tensor:
    """Rank-``rank`` truncated SVD of ``w`` (the metric projection onto ``M_r``)."""
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    return (u[:, :rank] * s[:rank]) @ vh[:rank]


def _backward(layer: LoRAADBLinear, x: Tensor, gy: Tensor) -> Tensor:
    """Run one forward/backward with output grad ``gy``; return the ambient ``G``."""
    y = layer(x.detach().clone().requires_grad_(True))
    y.backward(gy)
    return gy.float().transpose(-2, -1) @ x.float()  # G = dL/dW = gy^T x


def _velocity(opt: FixedRankRGD, layer: LoRAADBLinear) -> tuple[Tensor, ...]:
    """Snapshot of the stored momentum velocity ``(M, U_p, V_p)``."""
    return tuple(t.clone() for t in opt.state[layer.A]["momentum"])


class FlatParameter(torch.Tensor):
    """Stand-in for FSDP1's flattened parameter, which the optimizer matches by name."""


def _no_grad_forward(layer: LoRAADBLinear) -> None:
    """A validation-style forward under ``torch.no_grad``."""
    layer.eval()
    with torch.no_grad():
        layer(torch.randn(16, IN_FEATURES))
    layer.train()


def _plain_forward(layer: LoRAADBLinear) -> None:
    """A forward whose input does not require grad, so the output does not either."""
    layer(torch.randn(4, IN_FEATURES))


def _disabled_forward(layer: LoRAADBLinear) -> None:
    """A backward through the base weight only."""
    with disabled_adapters(layer):
        layer(torch.randn(4, IN_FEATURES, requires_grad=True)).backward(
            torch.randn(4, OUT_FEATURES)
        )


def _merged_forward(layer: LoRAADBLinear) -> None:
    """A backward through a merged (cached, frozen) effective weight."""
    layer.eval()
    layer.merge_weight()
    try:
        layer(torch.randn(4, IN_FEATURES, requires_grad=True)).backward(
            torch.randn(4, OUT_FEATURES)
        )
    finally:
        layer.unmerge_weight()
        layer.train()


class TestGradientCapture:
    @pytest.mark.parametrize(
        "batch_shape", [(5,), (3, 4)], ids=["flat-batch", "leading-dims-flattened"]
    )
    def test_captures_frame_projections_of_ambient_gradient(
        self, batch_shape: tuple[int, ...]
    ) -> None:
        layer = _layer()
        opt = FixedRankRGD([layer])
        u, _, v = _frames(layer)
        torch.manual_seed(1)
        x = torch.randn(*batch_shape, IN_FEATURES, requires_grad=True)
        gy = torch.randn(*batch_shape, OUT_FEATURES)

        layer(x).backward(gy)

        flat_x = x.detach().reshape(-1, IN_FEATURES)
        g = gy.reshape(-1, OUT_FEATURES).transpose(-2, -1) @ flat_x
        gv, gtu = opt._stash[0]
        assert gv is not None and gtu is not None
        torch.testing.assert_close(gv, g @ v, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(gtu, g.transpose(-2, -1) @ u, atol=1e-5, rtol=1e-5)

    def test_accumulates_across_microbatches(self) -> None:
        layer = _layer()
        opt = FixedRankRGD([layer])
        u, _, v = _frames(layer)
        torch.manual_seed(3)
        total = torch.zeros(OUT_FEATURES, IN_FEATURES)
        for _ in range(3):
            x = torch.randn(4, IN_FEATURES)
            gy = torch.randn(4, OUT_FEATURES)
            total = total + _backward(layer, x, gy)

        gv, gtu = opt._stash[0]
        assert gv is not None and gtu is not None
        torch.testing.assert_close(gv, total @ v, atol=1e-4, rtol=1e-5)
        torch.testing.assert_close(gtu, total.transpose(-2, -1) @ u, atol=1e-4, rtol=1e-5)

    @pytest.mark.parametrize(
        "forward",
        [_no_grad_forward, _plain_forward, _disabled_forward, _merged_forward],
        ids=["no-grad", "input-without-grad", "disabled-adapter", "merged-weight"],
    )
    def test_only_a_live_training_forward_is_captured(
        self, forward: Callable[[LoRAADBLinear], None]
    ) -> None:
        """None of these forwards is part of the differentiated function.

        Each must leave an in-progress accumulation exactly as it was.
        """
        layer = _layer()
        opt = FixedRankRGD([layer])
        _, _, v = _frames(layer)
        torch.manual_seed(4)
        g = _backward(layer, torch.randn(4, IN_FEATURES), torch.randn(4, OUT_FEATURES))

        forward(layer)

        gv, _ = opt._stash[0]
        assert gv is not None
        torch.testing.assert_close(gv, g @ v, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize(
        "consume",
        [lambda opt: opt.zero_grad(), lambda opt: opt.step()],
        ids=["zero_grad", "step"],
    )
    def test_consuming_the_stash_clears_it(self, consume: Callable[[FixedRankRGD], None]) -> None:
        layer = _layer()
        opt = FixedRankRGD([layer], lr=1e-2)
        torch.manual_seed(8)
        _backward(layer, torch.randn(4, IN_FEATURES), torch.randn(4, OUT_FEATURES))
        assert opt._stash[0][0] is not None

        consume(opt)

        assert opt._stash[0] == [None, None]


class TestHookLifecycle:
    def test_remove_hooks_is_idempotent_and_drops_any_captured_gradient(self) -> None:
        layer = _layer()
        before_hooks = len(layer._forward_hooks)
        opt = FixedRankRGD([layer], lr=0.1)
        assert len(layer._forward_hooks) == before_hooks + 1
        torch.manual_seed(11)
        _backward(layer, torch.randn(4, IN_FEATURES), torch.randn(4, OUT_FEATURES))
        before_adapter = _adapter(layer)

        opt.remove_hooks()
        opt.remove_hooks()

        assert len(layer._forward_hooks) == before_hooks
        assert opt._handles == []
        assert opt._stash[0] == [None, None]  # dropped, so a step is a no-op
        opt.step()
        torch.testing.assert_close(_adapter(layer), before_adapter, atol=0.0, rtol=0.0)
        torch.manual_seed(12)
        _backward(layer, torch.randn(4, IN_FEATURES), torch.randn(4, OUT_FEATURES))
        assert opt._stash[0] == [None, None]  # no longer captured either

    def test_remove_hooks_deactivates_an_outstanding_output_hook(self) -> None:
        layer = _layer()
        opt = FixedRankRGD([layer], lr=0.1)
        before_adapter = _adapter(layer)
        torch.manual_seed(13)
        output = layer(torch.randn(4, IN_FEATURES, requires_grad=True))

        opt.remove_hooks()
        output.backward(torch.randn(4, OUT_FEATURES))
        opt.step()

        assert opt._stash[0] == [None, None]
        torch.testing.assert_close(_adapter(layer), before_adapter, atol=0.0, rtol=0.0)

    def test_dropping_the_optimizer_removes_its_hooks(self) -> None:
        """The hook holds only a weak reference back, so the pair is collectable."""
        layer = _layer()
        opt = FixedRankRGD([layer])
        assert len(layer._forward_hooks) == 1

        del opt
        gc.collect()

        assert len(layer._forward_hooks) == 0
        torch.manual_seed(12)
        _backward(layer, torch.randn(4, IN_FEATURES), torch.randn(4, OUT_FEATURES))


class TestStep:
    @pytest.mark.parametrize(
        ("layer_factory", "lr_from_scheduler"),
        [
            (_layer, False),
            (lambda: _layer(manual_factors=True), False),
            (_layer, True),
        ],
        ids=["parameters", "manual-factor-buffers", "scheduler-lr"],
    )
    def test_matches_dense_reference_update(
        self, layer_factory: Callable[[], LoRAADBLinear], lr_from_scheduler: bool
    ) -> None:
        """One captured step == truncated SVD of ``W - lr P_T(G)`` from a dense ``G``.

        Covers the default (parameter) factors, ``manual_factors=True`` buffers, and a
        scheduler overriding ``param_groups[0]['lr']`` (the step must read the current
        group lr, not the constructor value).
        """
        layer = layer_factory()
        opt = FixedRankRGD([layer], lr=0.1)
        u, d, v = _frames(layer)
        w = (u * d) @ v.transpose(-2, -1)
        torch.manual_seed(50)
        g = _backward(layer, torch.randn(5, IN_FEATURES), torch.randn(5, OUT_FEATURES))
        if lr_from_scheduler:
            opt.param_groups[0]["lr"] = 0.02
        lr = opt.param_groups[0]["lr"]

        opt.step()

        expected = _truncate(w - lr * _dense_tangent(g, u, v), RANK)
        torch.testing.assert_close(_adapter(layer), expected, atol=1e-5, rtol=1e-5)
        # Coordinates are written back in the P1 orientation: A = U and B = V.
        assert layer.A.shape == (OUT_FEATURES, RANK)
        assert layer.B.shape == (IN_FEATURES, RANK)

    @pytest.mark.parametrize("momentum", [0.0, 0.9], ids=["plain", "momentum"])
    def test_repeated_steps_reduce_loss_and_keep_factors_orthonormal(
        self, momentum: float
    ) -> None:
        torch.manual_seed(53)
        target = torch.randn(OUT_FEATURES, RANK) @ torch.randn(RANK, IN_FEATURES)
        layer = _layer(d=[0.5, 0.25])
        opt = FixedRankRGD([layer], lr=0.5, momentum=momentum)
        x = torch.randn(64, IN_FEATURES, requires_grad=True)
        goal = x.detach() @ target.T

        def loss_value() -> Tensor:
            return ((layer(x) - goal) ** 2).mean()

        first = float(loss_value().detach())
        losses: list[float] = []
        for _ in range(100):
            loss = loss_value()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))

        # The rank-r target is reachable exactly, so the loss collapses.
        assert losses[-1] < 1e-3 * first
        assert losses[10] < losses[0]
        eye = torch.eye(RANK)
        torch.testing.assert_close(layer.A.T @ layer.A, eye, atol=1e-4, rtol=0)
        torch.testing.assert_close(layer.B.T @ layer.B, eye, atol=1e-4, rtol=0)
        assert bool(torch.isfinite(layer.d_raw).all())
        assert bool((layer.d_raw > 0).all())
        assert torch.linalg.matrix_rank(_adapter(layer)) == RANK

    @pytest.mark.parametrize(
        "both_have_gradients",
        [True, False],
        ids=["every-module-updates", "only-modules-with-gradients-update"],
    )
    def test_only_modules_with_gradients_are_updated(self, both_have_gradients: bool) -> None:
        used, other = _layer(seed=1), _layer(seed=2)
        opt = FixedRankRGD([used, other], lr=0.1)
        before_used, before_other = _adapter(used), _adapter(other)
        torch.manual_seed(56)
        _backward(used, torch.randn(4, IN_FEATURES), torch.randn(4, OUT_FEATURES))
        if both_have_gradients:
            _backward(other, torch.randn(4, IN_FEATURES), torch.randn(4, OUT_FEATURES))

        opt.step()

        assert not torch.allclose(_adapter(used), before_used)
        if both_have_gradients:
            assert not torch.allclose(_adapter(other), before_other)
        else:
            torch.testing.assert_close(_adapter(other), before_other, atol=0.0, rtol=0.0)

    def test_invalidates_a_merged_weight_cache(self) -> None:
        """A cache merged after construction is stale once the factors move."""
        layer = _layer()
        opt = FixedRankRGD([layer], lr=0.1)
        torch.manual_seed(58)
        _backward(layer, torch.randn(4, IN_FEATURES), torch.randn(4, OUT_FEATURES))
        layer.merge_weight()

        opt.step()

        assert layer._merged_weight is None
        layer.eval()
        with torch.no_grad():
            expected = layer.weight + _adapter(layer)
            torch.testing.assert_close(
                layer(torch.eye(IN_FEATURES)).T, expected, atol=1e-5, rtol=1e-5
            )


class TestGradientScaling:
    @pytest.mark.parametrize(
        ("divisor", "grad_scale"),
        [(2.0, None), (0.5, None), (None, 1024.0), (4.0, 8.0)],
        ids=["divisor-halves", "divisor-doubles", "loss-scale", "both"],
    )
    def test_gradient_is_divided_by_the_divisor_and_the_loss_scale(
        self, divisor: float | None, grad_scale: float | None
    ) -> None:
        """Neither is a distributed-only knob, and the two compose.

        ``grad_scale`` is the AMP contract: a loss scaled by ``s`` before ``backward``
        scales the captured gradient by ``s``, and reporting ``s`` divides it back out.
        """
        layer = _layer()
        opt = FixedRankRGD([layer], lr=0.1, gradient_divisor=divisor)
        u, d, v = _frames(layer)
        w = (u * d) @ v.transpose(-2, -1)
        torch.manual_seed(80)
        x = torch.randn(5, IN_FEATURES)
        gy = torch.randn(5, OUT_FEATURES) * (1.0 if grad_scale is None else grad_scale)
        g = _backward(layer, x, gy)

        opt.step(grad_scale=grad_scale)

        applied = (divisor or 1.0) * (grad_scale or 1.0)
        expected = _truncate(w - 0.1 * _dense_tangent(g / applied, u, v), RANK)
        torch.testing.assert_close(_adapter(layer), expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("bad", [float("inf"), float("nan")], ids=["inf", "nan"])
    def test_non_finite_gradient_is_skipped_without_touching_state(self, bad: float) -> None:
        layer = _layer()
        opt = FixedRankRGD([layer], lr=0.1, momentum=0.9)
        torch.manual_seed(83)
        _backward(layer, torch.randn(4, IN_FEATURES), torch.randn(4, OUT_FEATURES))
        opt.step()
        point = _adapter(layer)
        velocity = _velocity(opt, layer)
        gy = torch.randn(4, OUT_FEATURES)
        gy[0, 0] = bad

        _backward(layer, torch.randn(4, IN_FEATURES), gy)
        opt.step()

        assert opt.found_non_finite
        torch.testing.assert_close(_adapter(layer), point, atol=0.0, rtol=0.0)
        for kept, current in zip(velocity, _velocity(opt, layer), strict=True):
            torch.testing.assert_close(current, kept, atol=0.0, rtol=0.0)
        assert opt._stash[0] == [None, None]

    def test_non_finite_gradient_skips_the_whole_optimizer_step(self) -> None:
        finite, overflowing = _layer(seed=1), _layer(seed=2)
        layers = [finite, overflowing]
        opt = FixedRankRGD(layers, lr=0.1, momentum=0.9)
        torch.manual_seed(84)
        for layer in layers:
            _backward(layer, torch.randn(4, IN_FEATURES), torch.randn(4, OUT_FEATURES))
        opt.step()
        points = [_adapter(layer) for layer in layers]
        states = [
            {
                name: tuple(t.clone() for t in opt.state[layer.A][name])
                for name in ("master", "momentum")
            }
            for layer in layers
        ]
        bad_gradient = torch.randn(4, OUT_FEATURES)
        bad_gradient[0, 0] = float("inf")

        _backward(finite, torch.randn(4, IN_FEATURES), torch.randn(4, OUT_FEATURES))
        _backward(overflowing, torch.randn(4, IN_FEATURES), bad_gradient)
        opt.step()

        assert opt.found_non_finite
        for layer, point, state in zip(layers, points, states, strict=True):
            torch.testing.assert_close(_adapter(layer), point, atol=0.0, rtol=0.0)
            for name in ("master", "momentum"):
                for kept, current in zip(state[name], opt.state[layer.A][name], strict=True):
                    torch.testing.assert_close(current, kept, atol=0.0, rtol=0.0)
        assert opt._stash == [[None, None], [None, None]]


class TestPrecision:
    def test_master_accumulates_updates_smaller_than_the_factor_resolution(self) -> None:
        """Sub-ulp steps would vanish if each step re-read the bf16 factors."""
        layer = _layer().to(torch.bfloat16)
        opt = FixedRankRGD([layer], lr=3e-5)
        torch.manual_seed(85)
        x = torch.randn(4, IN_FEATURES, dtype=torch.bfloat16)
        gy = torch.randn(4, OUT_FEATURES, dtype=torch.bfloat16)
        start = _adapter(layer)

        _backward(layer, x, gy)
        opt.step()
        one_step = _adapter(layer)
        for _ in range(49):
            _backward(layer, x, gy)
            opt.step()

        assert torch.equal(one_step, start)  # a single step rounds away entirely
        assert float((_adapter(layer) - start).norm()) > 1e-3  # 50 of them do not
        assert all(t.dtype == torch.float32 for t in opt.state[layer.A]["master"])


class TestModuleMoves:
    def test_manual_factor_buffers_survive_a_dtype_move(self) -> None:
        """``module.to`` rebinds buffers to new tensors; the state must follow the module."""
        layer = _layer(manual_factors=True)
        opt = FixedRankRGD([layer], lr=0.1, momentum=0.9)
        torch.manual_seed(87)
        _backward(layer, torch.randn(4, IN_FEATURES), torch.randn(4, OUT_FEATURES))
        opt.step()
        velocity = _velocity(opt, layer)
        anchor = layer.A

        layer.to(torch.bfloat16)
        assert layer.A is not anchor
        _backward(
            layer,
            torch.randn(4, IN_FEATURES, dtype=torch.bfloat16),
            torch.randn(4, OUT_FEATURES, dtype=torch.bfloat16),
        )
        opt.step()

        # Same state entry, carried over from before the move (so the velocity was
        # used rather than silently restarted), and the factors moved.
        assert "momentum" in opt.state[anchor]
        assert not all(
            torch.equal(kept, current)
            for kept, current in zip(velocity, opt.state[anchor]["momentum"], strict=True)
        )
        assert layer.A.dtype == torch.bfloat16
        eye = torch.eye(RANK)
        torch.testing.assert_close(layer.A.float().T @ layer.A.float(), eye, atol=1e-2, rtol=0)


class TestMomentum:
    def test_momentum_zero_keeps_no_velocity(self) -> None:
        layer = _layer()
        opt = FixedRankRGD([layer], lr=0.1, momentum=0.0)
        torch.manual_seed(60)
        _backward(layer, torch.randn(4, IN_FEATURES), torch.randn(4, OUT_FEATURES))

        opt.step()

        assert all("momentum" not in state for state in opt.state.values())

    def test_velocity_is_stored_as_a_tangent_at_the_new_point(self) -> None:
        layer = _layer()
        opt = FixedRankRGD([layer], lr=0.1, momentum=0.9)
        torch.manual_seed(62)
        _backward(layer, torch.randn(4, IN_FEATURES), torch.randn(4, OUT_FEATURES))

        opt.step()

        m_vel, up_vel, vp_vel = opt.state[layer.A]["momentum"]
        assert m_vel.shape == (RANK, RANK)
        zeros = torch.zeros(RANK, RANK)
        torch.testing.assert_close(layer.A.T @ up_vel, zeros, atol=1e-5, rtol=0)
        torch.testing.assert_close(layer.B.T @ vp_vel, zeros, atol=1e-5, rtol=0)

    @pytest.mark.parametrize(
        "gradient",
        [None, torch.zeros(4, IN_FEATURES)],
        ids=["nothing-captured", "exactly-zero"],
    )
    def test_a_step_without_a_gradient_does_not_decay_the_velocity(
        self, gradient: Tensor | None
    ) -> None:
        """No gradient means "leave this adapter alone", not "apply a decayed velocity".

        This is the semantics the distributed path is forced into: a rank without a
        gradient contributes zeros, so a missing gradient and a captured-but-zero one
        must match.
        """
        layer = _layer()
        opt = FixedRankRGD([layer], lr=0.05, momentum=0.9)
        torch.manual_seed(64)
        _backward(layer, torch.randn(4, IN_FEATURES), torch.randn(4, OUT_FEATURES))
        opt.step()
        point = _adapter(layer)
        velocity = _velocity(opt, layer)

        if gradient is not None:
            _backward(layer, gradient, torch.randn(4, OUT_FEATURES))
        opt.step()

        torch.testing.assert_close(_adapter(layer), point, atol=0.0, rtol=0.0)
        for kept, current in zip(velocity, _velocity(opt, layer), strict=True):
            torch.testing.assert_close(current, kept, atol=0.0, rtol=0.0)


class TestStateDict:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
    def test_state_round_trips(self, dtype: torch.dtype) -> None:
        """A resume reproduces the run exactly, including at low factor precision.

        The loaded master and velocity must also stay fp32 -- plain
        ``Optimizer.load_state_dict`` would cast them down to the factor dtype.
        """
        torch.manual_seed(70)
        batches = [
            (
                torch.randn(4, IN_FEATURES).to(dtype),
                torch.randn(4, OUT_FEATURES).to(dtype),
            )
            for _ in range(2)
        ]
        layer = _layer().to(dtype)
        resumed = copy.deepcopy(layer)
        opt = FixedRankRGD([layer], lr=0.05, momentum=0.9)
        resumed_opt = FixedRankRGD([resumed], lr=0.05, momentum=0.9)

        _backward(layer, *batches[0])
        opt.step()
        resumed.load_state_dict(layer.state_dict())
        resumed_opt.load_state_dict(copy.deepcopy(opt.state_dict()))

        for key in ("master", "momentum"):
            loaded = resumed_opt.state[resumed.A][key]
            assert all(t.dtype == torch.float32 for t in loaded)
            for restored, original in zip(loaded, opt.state[layer.A][key], strict=True):
                assert restored.data_ptr() != original.data_ptr()

        _backward(layer, *batches[1])
        opt.step()
        _backward(resumed, *batches[1])
        resumed_opt.step()

        torch.testing.assert_close(_adapter(resumed), _adapter(layer), atol=0.0, rtol=0.0)

    def test_state_dict_exposes_the_hyperparameters(self) -> None:
        layer = _layer()
        opt = FixedRankRGD([layer], lr=0.03, momentum=0.7, eps=1e-6)

        group = opt.state_dict()["param_groups"][0]

        assert group["lr"] == pytest.approx(0.03)
        assert group["momentum"] == pytest.approx(0.7)
        assert group["eps"] == pytest.approx(1e-6)


class TestInvalidArguments:
    @pytest.mark.parametrize(
        ("build", "match"),
        [
            (lambda: FixedRankRGD([]), "must not be empty"),
            (lambda: FixedRankRGD([_layer()], lr=0.0), "lr must be > 0"),
            (lambda: FixedRankRGD([_layer()], eps=0.0), "eps must be > 0"),
            (lambda: FixedRankRGD([_layer()], momentum=-0.1), r"momentum must be in \[0, 1\)"),
            (lambda: FixedRankRGD([_layer()], momentum=1.0), r"momentum must be in \[0, 1\)"),
            (
                lambda: FixedRankRGD([_layer()], gradient_divisor=0.0),
                "gradient_divisor must be > 0",
            ),
        ],
    )
    def test_rejects_invalid_hyperparameters(
        self, build: Callable[[], FixedRankRGD], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            build()

    def test_does_not_install_hooks_when_construction_fails(self) -> None:
        layer = _layer()
        with pytest.raises(ValueError):
            FixedRankRGD([layer], lr=-1.0)
        assert len(layer._forward_hooks) == 0

    def test_rejects_a_module_with_a_merged_weight(self) -> None:
        layer = _layer()
        layer.merge_weight()
        with pytest.raises(ValueError, match="merged weight cache"):
            FixedRankRGD([layer])

    def test_rejects_factors_split_across_devices_before_installing_hooks(self) -> None:
        layer = _layer(manual_factors=True)
        layer.B = layer.B.to("meta")

        with pytest.raises(ValueError, match="each ADB module must use one factor device"):
            FixedRankRGD([layer])

        assert len(layer._forward_hooks) == 0

    @pytest.mark.parametrize("sharded", ["dtensor", "flat-parameter"])
    def test_rejects_sharded_factors(self, sharded: str) -> None:
        """FSDP / tensor parallel: the retraction needs the whole, replicated adapter."""
        from torch.distributed.tensor import DTensor

        cls = DTensor if sharded == "dtensor" else FlatParameter
        layer = _layer(manual_factors=True)
        layer.A = torch.zeros(OUT_FEATURES, RANK).as_subclass(cls)

        with pytest.raises(TypeError, match="plain replicated factors"):
            FixedRankRGD([layer])

    def test_rejects_a_closure(self) -> None:
        layer = _layer()
        opt = FixedRankRGD([layer], lr=0.1)
        with pytest.raises(ValueError, match="does not support closures"):
            opt.step(lambda: 0.0)

    @pytest.mark.parametrize("grad_scale", [0.0, -1.0])
    def test_rejects_invalid_grad_scale(self, grad_scale: float) -> None:
        layer = _layer()
        opt = FixedRankRGD([layer], lr=0.1)
        with pytest.raises(ValueError, match="grad_scale must be > 0"):
            opt.step(grad_scale=grad_scale)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _init_process_group(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def _rank_batch(rank: int) -> tuple[Tensor, Tensor]:
    """Deterministic per-rank batch that every rank can rebuild for the reference."""
    return (
        torch.randn(4, IN_FEATURES, generator=torch.Generator().manual_seed(100 + rank)),
        torch.randn(4, OUT_FEATURES, generator=torch.Generator().manual_seed(200 + rank)),
    )


def _expected_point(
    layer_frames: tuple[Tensor, Tensor, Tensor], g: Tensor, scale: float
) -> Tensor:
    """Adapter after one ``lr=0.1`` retraction along ``scale * P_T(g)``."""
    u, d, v = layer_frames
    m, u_p, v_p = _tangent_factored(g @ v, g.transpose(-2, -1) @ u, u, v)
    u_new, d_new, v_new = _retract(u, d, v, scale * m, scale * u_p, scale * v_p, lr=0.1, eps=1e-8)
    return (u_new * d_new) @ v_new.transpose(-2, -1)


def _asymmetric_worker(rank: int, world_size: int, port: int) -> None:
    """Assert ranks stay in lockstep when only some of them captured a gradient."""
    _init_process_group(rank, world_size, port)
    try:
        layer = _layer()  # identical on every rank (seeded)
        opt = FixedRankRGD([layer], lr=0.1, momentum=0.9)
        frames = _frames(layer)
        x, gy = _rank_batch(0)
        if rank == 0:
            _backward(layer, x, gy)

        opt.step()

        # The other ranks contributed zeros to keep the collectives aligned, so every
        # rank retracts the same partial-batch gradient (and the first EMA velocity is
        # ``(1 - momentum) * G``).
        g = (gy.transpose(-2, -1) @ x) / float(world_size)
        expected = _expected_point(frames, g, scale=0.1)
        torch.testing.assert_close(_adapter(layer), expected, atol=1e-5, rtol=1e-5)
        gathered = [torch.zeros_like(layer.A) for _ in range(world_size)]
        dist.all_gather(gathered, layer.A.detach())
        for other in gathered[1:]:
            torch.testing.assert_close(other, gathered[0], atol=0.0, rtol=0.0)

        # No rank captured anything: the reduction is exactly zero, which must leave
        # the module completely alone -- velocity included, exactly as in a single
        # process.
        point = _adapter(layer)
        velocity = _velocity(opt, layer)
        opt.step()
        torch.testing.assert_close(_adapter(layer), point, atol=0.0, rtol=0.0)
        for kept, current in zip(velocity, _velocity(opt, layer), strict=True):
            torch.testing.assert_close(current, kept, atol=0.0, rtol=0.0)
        opt.remove_hooks()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="requires torch.distributed with the gloo backend",
)
def test_two_process_gradient_reduction_is_rank_asymmetric() -> None:
    """Ranks without a captured gradient contribute zeros, keeping every rank in lockstep."""
    mp.spawn(_asymmetric_worker, args=(2, _free_port()), nprocs=2, join=True)
