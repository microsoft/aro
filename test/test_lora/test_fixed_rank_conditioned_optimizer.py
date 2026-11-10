"""Tests for the :class:`aro.lora.FixedRankRGD` spectral-conditioning integration.

``aro.lora._fixed_rank_conditioning`` is validated on its own, dense-vs-factored, in
``test_fixed_rank_conditioning.py``; this file covers the optimizer-level wiring
around it instead of re-deriving its math -- the reference step below composes the
already-validated primitives (``_tangent_factored``, ``_condition``,
``_rms_normalized``) directly. That covers: option validation, the fp32 tracked
frame in the optimizer state (round-tripping, resync, refinement), and that
conditioning shapes the retracted step only, never the momentum buffer.
"""

import copy
from typing import Any

import pytest
import torch
import torch.testing
from torch import Tensor

from aro.lora import FixedRankRGD, LoRAADBLinear
from aro.lora._fixed_rank_conditioning import FrameUpdate, _condition, _rms_normalized
from aro.lora._fixed_rank_geometry import _tangent_factored, _transport

IN_FEATURES = 8
OUT_FEATURES = 6
RANK = 2

SHAPES = [
    pytest.param(6, 8, False, id="wide"),
    pytest.param(8, 6, True, id="tall"),
    pytest.param(6, 6, True, id="square"),
]


def _layer(
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    *,
    out_features: int = OUT_FEATURES,
    in_features: int = IN_FEATURES,
) -> LoRAADBLinear:
    """ADB layer with a well-separated spectrum (avoids SVD subspace ambiguity)."""
    torch.manual_seed(seed)
    layer = LoRAADBLinear(in_features=in_features, out_features=out_features, rank=RANK)
    with torch.no_grad():
        layer.d_raw.copy_(torch.tensor([1.0, 0.5]))
    return layer.to(dtype)


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


def _backward(layer: LoRAADBLinear, seed: int) -> Tensor:
    """Run one forward/backward on random data; return the ambient ``G = gy^T x``."""
    torch.manual_seed(seed)
    dtype = layer.A.dtype
    x = torch.randn(5, layer.in_features, dtype=dtype, device=layer.A.device, requires_grad=True)
    gy = torch.randn(5, layer.out_features, dtype=dtype, device=layer.A.device)
    layer(x).backward(gy)
    return gy.float().transpose(-2, -1) @ x.detach().float()


def _truncate(w: Tensor, rank: int = RANK) -> Tensor:
    """Rank-``rank`` truncated SVD of ``w`` (the metric projection onto ``M_r``)."""
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    return (u[:, :rank] * s[:rank]) @ vh[:rank]


def _dense(m: Tensor, u_p: Tensor, v_p: Tensor, u: Tensor, v: Tensor) -> Tensor:
    """Rebuild the ambient ``(out, in)`` matrix of a tangent decomposition."""
    return u @ m @ v.transpose(-2, -1) + u_p @ v.transpose(-2, -1) + u @ v_p.transpose(-2, -1)


class TestOptimizerConditioning:
    def test_the_default_leaves_the_step_unconditioned(self) -> None:
        plain, explicit = _layer(), _layer()
        plain_opt = FixedRankRGD([plain], lr=0.1, momentum=0.9)
        explicit_opt = FixedRankRGD([explicit], lr=0.1, momentum=0.9, conditioning="none")

        for seed in (40, 41):
            _backward(plain, seed)
            _backward(explicit, seed)
            plain_opt.step()
            explicit_opt.step()

        torch.testing.assert_close(_adapter(explicit), _adapter(plain), atol=0.0, rtol=0.0)
        assert all("frame" not in state for state in plain_opt.state.values())

    @pytest.mark.parametrize(("out_features", "in_features", "input_side"), SHAPES)
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
            ),
        ],
    )
    @pytest.mark.parametrize(
        ("sinkhorn_steps", "frame_update", "rms_normalize"),
        [
            (0, None, False),
            (2, None, False),
            (0, None, True),
            (2, None, True),
            (2, "qr", False),
            (2, "polar", False),
            (2, "polar", True),
        ],
        ids=[
            "whiten",
            "sinkhorn",
            "whiten-rms",
            "sinkhorn-rms",
            "sinkhorn-qr",
            "sinkhorn-polar",
            "sinkhorn-polar-rms",
        ],
    )
    def test_matches_the_reference_conditioning_side(
        self,
        out_features: int,
        in_features: int,
        input_side: bool,
        device: str,
        sinkhorn_steps: int,
        frame_update: FrameUpdate | None,
        rms_normalize: bool,
    ) -> None:
        """Conditioning uses the smaller side, with input-side ties."""
        layer = _layer(out_features=out_features, in_features=in_features).to(device)
        opt = FixedRankRGD(
            [layer],
            lr=0.1,
            conditioning="spectral",
            sinkhorn_steps=sinkhorn_steps,
            frame_update=frame_update,
            rms_normalize=rms_normalize,
        )
        u, d, v = _frames(layer)
        w = (u * d) @ v.transpose(-2, -1)
        _backward(layer, seed=42)
        gv, gtu = opt._stash[0]
        assert gv is not None and gtu is not None
        m, u_p, v_p = _tangent_factored(gv, gtu, u, v)

        opt.step()

        if input_side:
            core, left, right, _ = _condition(
                m.T, v_p, u_p, v, u, sinkhorn_steps=sinkhorn_steps, frame_update=frame_update
            )
            blocks = (core.T, right, left)
        else:
            blocks = _condition(
                m, u_p, v_p, u, v, sinkhorn_steps=sinkhorn_steps, frame_update=frame_update
            )[:3]
        if rms_normalize:
            blocks = _rms_normalized(*blocks)
        torch.testing.assert_close(
            _adapter(layer), _truncate(w - 0.1 * _dense(*blocks, u, v)), atol=1e-5, rtol=1e-5
        )

    def test_conditioning_shapes_the_step_but_not_the_momentum_buffer(self) -> None:
        """The buffer accumulates the raw Riemannian velocity, conditioned or not."""
        layer = _layer()
        momentum = 0.9
        opt = FixedRankRGD(
            [layer], lr=0.07, momentum=momentum, conditioning="spectral", rms_normalize=True
        )
        u, _, v = _frames(layer)
        g = _backward(layer, seed=44)

        opt.step()

        m, u_p, v_p = _tangent_factored(g @ v, g.transpose(-2, -1) @ u, u, v)
        raw = tuple((1.0 - momentum) * block for block in (m, u_p, v_p))
        u_new, _, v_new = _frames(layer)
        expected = _transport(u, v, u_new, v_new, *raw)
        for stored, want in zip(opt.state[layer.A]["momentum"], expected, strict=True):
            torch.testing.assert_close(stored, want, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize(("out_features", "in_features", "input_side"), SHAPES)
    def test_tracked_frame_is_carried_across_steps(
        self, out_features: int, in_features: int, input_side: bool
    ) -> None:
        tracked = _layer(out_features=out_features, in_features=in_features)
        untracked = copy.deepcopy(tracked)
        shared: dict[str, Any] = {
            "lr": 0.05,
            "conditioning": "spectral",
            "sinkhorn_steps": 2,
            "frame_update": "polar",
        }
        tracked_opt = FixedRankRGD([tracked], track_frame=True, **shared)
        untracked_opt = FixedRankRGD([untracked], **shared)

        for seed in (46, 47, 48):
            _backward(tracked, seed)
            _backward(untracked, seed)
            tracked_opt.step()
            untracked_opt.step()

        frame = tracked_opt.state[tracked.A]["frame"]
        assert frame.shape == (in_features if input_side else out_features, 2 * RANK)
        assert frame.dtype == torch.float32
        torch.testing.assert_close(
            frame.transpose(-2, -1) @ frame, torch.eye(2 * RANK), atol=1e-5, rtol=0
        )
        assert "frame" not in untracked_opt.state[untracked.A]
        assert not torch.allclose(_adapter(tracked), _adapter(untracked), atol=1e-6)

    @pytest.mark.parametrize(("out_features", "in_features", "input_side"), SHAPES)
    @pytest.mark.parametrize("frame_update", ["qr", "polar"])
    def test_tracked_frame_round_trips_through_the_state_dict_in_fp32(
        self, out_features: int, in_features: int, input_side: bool, frame_update: FrameUpdate
    ) -> None:
        """A resume reproduces the run, and the frame is not rounded to the factors."""
        layer = _layer(dtype=torch.bfloat16, out_features=out_features, in_features=in_features)
        resumed = copy.deepcopy(layer)
        kwargs: dict[str, Any] = {
            "lr": 0.05,
            "momentum": 0.9,
            "conditioning": "spectral",
            "sinkhorn_steps": 2,
            "frame_update": frame_update,
            "track_frame": True,
            "rms_normalize": True,
        }
        opt = FixedRankRGD([layer], **kwargs)
        resumed_opt = FixedRankRGD([resumed], **kwargs)

        _backward(layer, seed=50)
        opt.step()
        resumed.load_state_dict(layer.state_dict())
        resumed_opt.load_state_dict(copy.deepcopy(opt.state_dict()))

        loaded = resumed_opt.state[resumed.A]["frame"]
        original = opt.state[layer.A]["frame"]
        assert loaded.shape == (in_features if input_side else out_features, 2 * RANK)
        assert loaded.dtype == torch.float32
        assert loaded.data_ptr() != original.data_ptr()
        torch.testing.assert_close(loaded, original, atol=0.0, rtol=0.0)
        _backward(layer, seed=51)
        opt.step()
        _backward(resumed, seed=51)
        resumed_opt.step()
        torch.testing.assert_close(_adapter(resumed), _adapter(layer), atol=0.0, rtol=0.0)

    def test_resync_drops_and_rebuilds_the_tracked_frame(self) -> None:
        layer = _layer()
        opt = FixedRankRGD(
            [layer],
            lr=0.05,
            conditioning="spectral",
            sinkhorn_steps=2,
            frame_update="polar",
            track_frame=True,
        )
        _backward(layer, seed=52)
        opt.step()
        assert "frame" in opt.state[layer.A]

        opt.resync_from_modules()

        assert "frame" not in opt.state[layer.A]

        _backward(layer, seed=53)
        opt.step()
        frame = opt.state[layer.A]["frame"]
        torch.testing.assert_close(
            frame.transpose(-2, -1) @ frame, torch.eye(2 * RANK), atol=1e-5, rtol=0
        )

    @pytest.mark.parametrize(("out_features", "in_features", "input_side"), SHAPES)
    @pytest.mark.parametrize("frame_update", ["qr", "polar"])
    def test_tracked_frame_refinement_matches_the_conditioner(
        self, out_features: int, in_features: int, input_side: bool, frame_update: FrameUpdate
    ) -> None:
        layer = _layer(out_features=out_features, in_features=in_features)
        opt = FixedRankRGD(
            [layer],
            lr=0.05,
            conditioning="spectral",
            sinkhorn_steps=2,
            frame_update=frame_update,
            track_frame=True,
        )
        _backward(layer, seed=56)
        opt.step()
        state = opt.state[layer.A]
        u, _, v = _frames(layer)

        _backward(layer, seed=57)
        gv, gtu = opt._stash[0]
        assert gv is not None and gtu is not None
        m, u_p, v_p = _tangent_factored(gv, gtu, u, v)
        args = (m.T, v_p, u_p, v, u) if input_side else (m, u_p, v_p, u, v)
        expected_frame = _condition(
            *args,
            frame=state["frame"],
            sinkhorn_steps=2,
            frame_update=frame_update,
        )[3]
        opt.step()

        frame = opt.state[layer.A]["frame"]
        torch.testing.assert_close(frame, expected_frame, atol=0, rtol=0)
        torch.testing.assert_close(
            frame.transpose(-2, -1) @ frame, torch.eye(2 * RANK), atol=1e-5, rtol=0
        )
        eye = torch.eye(RANK)
        torch.testing.assert_close(layer.A.T @ layer.A, eye, atol=1e-5, rtol=0)
        torch.testing.assert_close(layer.B.T @ layer.B, eye, atol=1e-5, rtol=0)
        assert bool(_adapter(layer).isfinite().all())

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"conditioning": "muon"}, "conditioning must be one of"),
            ({"sinkhorn_steps": 1}, "sinkhorn_steps only applies"),
            ({"frame_update": "qr"}, "frame_update only applies"),
            ({"track_frame": True}, "track_frame only applies"),
            ({"conditioning": "spectral", "sinkhorn_steps": -1}, "sinkhorn_steps must be >= 0"),
            ({"conditioning": "spectral", "tau": 1.0}, r"tau must be in \[0, 1\)"),
            ({"conditioning": "spectral", "frame_update": "svd"}, "frame_update must be one of"),
            (
                {"conditioning": "spectral", "track_frame": True},
                "track_frame requires a frame_update",
            ),
            (
                {"conditioning": "spectral", "track_frame": True, "frame_update": "qr"},
                "needs sinkhorn_steps > 0",
            ),
        ],
        ids=[
            "unknown-mode",
            "sinkhorn-without-conditioning",
            "frame-update-without-conditioning",
            "track-without-conditioning",
            "negative-sinkhorn-steps",
            "tau-out-of-range",
            "unknown-frame-update",
            "track-without-frame-update",
            "qr-tracking-without-sinkhorn",
        ],
    )
    def test_rejects_invalid_combinations(self, kwargs: dict[str, Any], match: str) -> None:
        """A knob that cannot reach the update is an error, never a silent no-op."""
        with pytest.raises(ValueError, match=match):
            FixedRankRGD([_layer()], **kwargs)

    def test_rms_normalize_alone_is_a_valid_step_rescale(self) -> None:
        """It is a shape-only rescale, so it does not require the conditioner."""
        layer = _layer()
        opt = FixedRankRGD([layer], lr=1e-3, rms_normalize=True)
        u, d, v = _frames(layer)
        w = (u * d) @ v.transpose(-2, -1)
        g = _backward(layer, seed=54)

        opt.step()

        m, u_p, v_p = _tangent_factored(g @ v, g.transpose(-2, -1) @ u, u, v)
        step = _dense(*_rms_normalized(m, u_p, v_p), u, v)
        torch.testing.assert_close(
            _adapter(layer), _truncate(w - 1e-3 * step), atol=1e-5, rtol=1e-5
        )

    def test_repeated_conditioned_steps_reduce_the_loss(self) -> None:
        torch.manual_seed(55)
        target = torch.randn(OUT_FEATURES, RANK) @ torch.randn(RANK, IN_FEATURES)
        layer = _layer()
        opt = FixedRankRGD(
            [layer],
            lr=0.02,
            momentum=0.9,
            conditioning="spectral",
            sinkhorn_steps=1,
            rms_normalize=True,
        )
        x = torch.randn(64, IN_FEATURES, requires_grad=True)
        goal = x.detach() @ target.transpose(-2, -1)

        losses: list[float] = []
        for _ in range(60):
            loss = ((layer(x) - goal) ** 2).mean()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))

        assert losses[-1] < 0.2 * losses[0]
        eye = torch.eye(RANK)
        torch.testing.assert_close(layer.A.T @ layer.A, eye, atol=1e-4, rtol=0)
        torch.testing.assert_close(layer.B.T @ layer.B, eye, atol=1e-4, rtol=0)
        assert bool(torch.isfinite(layer.d_raw).all())
