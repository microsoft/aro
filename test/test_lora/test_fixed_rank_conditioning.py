"""Tests for the spectral conditioning helpers in ``aro.lora._fixed_rank_conditioning``.

The conditioner never forms the ``out x in`` ambient matrix, so every check here
rebuilds the conditioned tangent vector densely and compares it against an explicit
reference: reduce the tangent to its coefficients in an orthonormal ``2r``-frame,
whiten (and optionally Sinkhorn-balance) them, and project back. These tests cover
the pure tensor helpers only. Optimizer controls, tracked-frame persistence, and
optimizer-level integration are covered in ``test_fixed_rank_conditioned_optimizer.py``.
"""

import pytest
import torch
import torch.testing
from torch import Tensor

from aro.lora import _fixed_rank_conditioning as conditioning
from aro.lora._fixed_rank_conditioning import (
    FrameUpdate,
    _condition,
    _condition_reduced,
    _rms_normalized,
    _spectral_frame,
    _whiten_rows,
)
from aro.lora._fixed_rank_geometry import _tangent_factored

IN_FEATURES = 8
OUT_FEATURES = 6
RANK = 2
TAU = 1e-4

type _Tangent = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
"""A tangent decomposition and its point: ``(M, U_p, V_p, U, V)``."""


def _tangent(seed: int, *, dtype: torch.dtype | None = None) -> _Tangent:
    """Tangent projection of a random ambient gradient at a random point."""
    torch.manual_seed(seed)
    u = torch.linalg.qr(torch.randn(OUT_FEATURES, RANK, dtype=dtype))[0]
    v = torch.linalg.qr(torch.randn(IN_FEATURES, RANK, dtype=dtype))[0]
    g = torch.randn(OUT_FEATURES, IN_FEATURES, dtype=dtype)
    m, u_p, v_p = _tangent_factored(g @ v, g.transpose(-2, -1) @ u, u, v)
    return m, u_p, v_p, u, v


def _collapsed_tangent(seed: int, gap: float) -> _Tangent:
    """Tangent whose perpendicular blocks lose one direction, at full overall scale.

    The second column of each perpendicular block is replaced by the first plus ``gap``
    times its own direction: ``gap = 0`` makes the block exactly rank ``r - 1`` and a
    small positive ``gap`` leaves it merely ill-conditioned. This is the case that
    breaks a QR of the block alone -- the missing direction is completed with something
    unrelated to the frame it must be perpendicular to -- without shrinking the tangent.
    """
    m, u_p, v_p, u, v = _tangent(seed)
    u_p[:, 1] = u_p[:, 0] + gap * u_p[:, 1]
    v_p[:, 1] = v_p[:, 0] + gap * v_p[:, 1]
    return m, u_p, v_p, u, v


def _dense(m: Tensor, u_p: Tensor, v_p: Tensor, u: Tensor, v: Tensor) -> Tensor:
    """Rebuild the ambient ``(out, in)`` matrix of a tangent decomposition."""
    return u @ m @ v.transpose(-2, -1) + u_p @ v.transpose(-2, -1) + u @ v_p.transpose(-2, -1)


def _dense_basis(xi: Tensor) -> Tensor:
    """Top-``2r`` left singular basis of ``xi``, straight from a dense SVD."""
    return torch.linalg.svd(xi, full_matrices=False)[0][:, : 2 * RANK]


def _dense_reduced(red: Tensor, sinkhorn_steps: int, tau: float) -> Tensor:
    """Explicit normalization recurrence, independent of production helpers."""

    def whiten(x: Tensor) -> Tensor:
        norms = x.norm(dim=-1, keepdim=True)
        return torch.where(norms < tau * norms.max(), x, x / (norms + 1e-12))

    if sinkhorn_steps > 0:
        for _ in range(sinkhorn_steps):
            red = whiten(red)
            red = red / (red.norm(dim=-2, keepdim=True) + 1e-12)
    else:
        red = whiten(red)
    return red


def _dense_condition(
    xi: Tensor,
    u: Tensor,
    v: Tensor,
    *,
    frame: Tensor | None = None,
    sinkhorn_steps: int = 0,
    tau: float = TAU,
    frame_update: FrameUpdate | None = None,
) -> tuple[Tensor, Tensor]:
    """Reference ``(step, frame)``: ``P_T(F f(F^T xi))`` on the ambient matrix."""
    used = _dense_basis(xi) if frame is None else frame
    if frame_update is not None:
        update = xi @ _dense_reduced(used.transpose(-2, -1) @ xi, sinkhorn_steps, tau).transpose(
            -2, -1
        )
        if frame_update == "polar":
            a, _, bt = torch.linalg.svd(update, full_matrices=False)
            used = a @ bt
        else:
            used = torch.linalg.qr(update)[0]
    red = _dense_reduced(used.transpose(-2, -1) @ xi, sinkhorn_steps, tau)
    step = used @ red
    pu, pv = u @ u.transpose(-2, -1), v @ v.transpose(-2, -1)  # P_U step + step P_V - P_U step P_V
    return pu @ step + step @ pv - pu @ step @ pv, used


class TestSpectralFrame:
    def test_is_the_top_2r_left_singular_basis(self) -> None:
        m, u_p, v_p, u, v = _tangent(seed=1)

        frame = _spectral_frame(m, u_p, v_p, u, v)

        xi = _dense(m, u_p, v_p, u, v)
        expected = _dense_basis(xi)
        # Singular vectors are defined up to sign, so compare the projectors.
        torch.testing.assert_close(
            frame @ frame.transpose(-2, -1),
            expected @ expected.transpose(-2, -1),
            atol=1e-5,
            rtol=0,
        )

    def test_has_orthonormal_columns(self) -> None:
        m, u_p, v_p, u, v = _tangent(seed=2)

        frame = _spectral_frame(m, u_p, v_p, u, v)

        assert frame.shape == (OUT_FEATURES, 2 * RANK)
        torch.testing.assert_close(
            frame.transpose(-2, -1) @ frame, torch.eye(2 * RANK), atol=1e-5, rtol=0
        )

    def test_spans_the_column_space_of_the_tangent(self) -> None:
        """Reducing in the frame loses nothing: ``F F^T xi == xi``."""
        m, u_p, v_p, u, v = _tangent(seed=3)
        xi = _dense(m, u_p, v_p, u, v)

        frame = _spectral_frame(m, u_p, v_p, u, v)

        torch.testing.assert_close(
            frame @ (frame.transpose(-2, -1) @ xi), xi, atol=1e-5, rtol=1e-5
        )

    @pytest.mark.parametrize("gap", [0.0, 1e-9], ids=["exact", "ill-conditioned"])
    def test_a_collapsed_direction_keeps_the_frame_orthonormal(self, gap: float) -> None:
        """The completion of a rank-deficient block must stay off ``U``.

        With the direction collapsed, ``U_p`` no longer determines its own second
        complement column: a QR of ``U_p`` alone fills it in with whatever is
        orthogonal to the columns it already emitted, which need not be perpendicular
        to ``U``, and ``[U, Q_up]`` -- hence the frame -- stops being orthonormal.
        """
        m, u_p, v_p, u, v = _collapsed_tangent(seed=60, gap=gap)

        frame = _spectral_frame(m, u_p, v_p, u, v)

        assert frame.shape == (OUT_FEATURES, 2 * RANK)
        torch.testing.assert_close(
            frame.transpose(-2, -1) @ frame, torch.eye(2 * RANK), atol=1e-5, rtol=0
        )

    @pytest.mark.parametrize("gap", [0.0, 1e-9], ids=["exact", "ill-conditioned"])
    def test_a_collapsed_direction_keeps_the_tangent_span(self, gap: float) -> None:
        """A rank-deficient tangent still reduces losslessly: ``F F^T xi == xi``.

        The top-``2r`` left singular basis is ambiguous here (the collapsed direction
        carries no singular value to order it by), so only the span is well defined --
        and it is exactly what the conditioner needs the frame for.
        """
        m, u_p, v_p, u, v = _collapsed_tangent(seed=61, gap=gap)
        xi = _dense(m, u_p, v_p, u, v)
        assert float(torch.linalg.svdvals(xi)[2 * RANK - 1]) < 1e-6 * float(xi.norm())

        frame = _spectral_frame(m, u_p, v_p, u, v)

        torch.testing.assert_close(
            frame @ (frame.transpose(-2, -1) @ xi), xi, atol=1e-5, rtol=1e-5
        )


class TestWhitenRows:
    @pytest.mark.parametrize("tau", [-0.1, 1.0, 2.0, float("nan"), float("inf"), -float("inf")])
    def test_rejects_an_invalid_floor(self, tau: float) -> None:
        with pytest.raises(ValueError, match="tau must be finite and in"):
            _whiten_rows(torch.ones(2 * RANK, IN_FEATURES), tau)

    def test_normalizes_rows_above_the_floor(self) -> None:
        torch.manual_seed(4)
        red = torch.randn(2 * RANK, IN_FEATURES)

        whitened = _whiten_rows(red, TAU)

        torch.testing.assert_close(whitened.norm(dim=-1), torch.ones(2 * RANK), atol=1e-6, rtol=0)

    def test_leaves_sub_floor_rows_untouched(self) -> None:
        """Near-null directions must not be amplified to unit scale."""
        torch.manual_seed(5)
        red = torch.randn(2 * RANK, IN_FEATURES)
        red[2] *= 1e-8

        whitened = _whiten_rows(red, TAU)

        torch.testing.assert_close(whitened[2], red[2], atol=0.0, rtol=0.0)
        torch.testing.assert_close(whitened[0], red[0] / red[0].norm(), atol=1e-6, rtol=1e-6)

    def test_a_zero_input_stays_zero(self) -> None:
        red = torch.zeros(2 * RANK, IN_FEATURES)

        whitened = _whiten_rows(red, TAU)

        torch.testing.assert_close(whitened, red, atol=0.0, rtol=0.0)


class TestConditionReduced:
    @pytest.mark.parametrize("sinkhorn_steps", [-1, -3, 1.5])
    def test_rejects_invalid_sweep_counts(self, sinkhorn_steps: int) -> None:
        with pytest.raises(ValueError, match="sinkhorn_steps must be a nonnegative integer"):
            _condition_reduced(torch.ones(2 * RANK, IN_FEATURES), sinkhorn_steps, TAU)

    @pytest.mark.parametrize("sinkhorn_steps", [0, 2])
    @pytest.mark.parametrize("tau", [-0.1, 1.0, float("nan"), float("inf")])
    def test_rejects_an_invalid_floor(self, sinkhorn_steps: int, tau: float) -> None:
        with pytest.raises(ValueError, match="tau must be finite and in"):
            _condition_reduced(torch.ones(2 * RANK, IN_FEATURES), sinkhorn_steps, tau)

    @pytest.mark.parametrize("sinkhorn_steps", [0, 1, 3])
    def test_matches_the_explicit_reference(self, sinkhorn_steps: int) -> None:
        torch.manual_seed(6)
        red = torch.randn(2 * RANK, IN_FEATURES)

        result = _condition_reduced(red, sinkhorn_steps, TAU)

        expected = _dense_reduced(red, sinkhorn_steps, TAU)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_zero_steps_only_whitens(self) -> None:
        torch.manual_seed(7)
        red = torch.randn(2 * RANK, IN_FEATURES)

        torch.testing.assert_close(
            _condition_reduced(red, 0, TAU), _whiten_rows(red, TAU), atol=0.0, rtol=0.0
        )

    def test_sinkhorn_balances_the_columns(self) -> None:
        """The sweep ends on the column normalization, so the columns come out even."""
        torch.manual_seed(8)
        red = torch.randn(2 * RANK, IN_FEATURES) * torch.linspace(1e-2, 1e2, IN_FEATURES)

        result = _condition_reduced(red, 3, TAU)

        torch.testing.assert_close(result.norm(dim=-2), torch.ones(IN_FEATURES), atol=1e-6, rtol=0)

    @pytest.mark.parametrize("sinkhorn_steps", [1, 3])
    def test_sinkhorn_keeps_the_original_column_scale(self, sinkhorn_steps: int) -> None:
        """Sinkhorn ends on unit columns, without a whitening-only norm rescale."""
        torch.manual_seed(9)
        red = torch.randn(2 * RANK, IN_FEATURES) * torch.linspace(1e-2, 1e2, IN_FEATURES)

        whitened = _condition_reduced(red, 0, TAU)
        balanced = _condition_reduced(red, sinkhorn_steps, TAU)

        assert float(balanced.norm()) == pytest.approx(IN_FEATURES**0.5, rel=1e-5)
        assert float(balanced.norm() / whitened.norm()) == pytest.approx(
            (IN_FEATURES / (2 * RANK)) ** 0.5, rel=1e-5
        )
        assert not torch.allclose(
            balanced / balanced.norm(), whitened / whitened.norm(), atol=1e-2
        )
        spread = whitened.norm(dim=-2)
        assert float(spread.max() / spread.min()) > 10.0  # whitening leaves them uneven
        balance = balanced.norm(dim=-2)
        assert float(balance.max() / balance.min()) == pytest.approx(1.0, rel=1e-5)


class TestCondition:
    def test_rejects_invalid_conditioning_options(self) -> None:
        tangent = _tangent(seed=25)
        with pytest.raises(ValueError, match="sinkhorn_steps must be a nonnegative integer"):
            _condition(*tangent, sinkhorn_steps=-1)
        with pytest.raises(ValueError, match="tau must be finite and in"):
            _condition(*tangent, tau=float("nan"))

    @pytest.mark.parametrize(
        ("sinkhorn_steps", "frame_update"),
        [(0, None), (1, "qr"), (3, "polar")],
        ids=["whiten", "sinkhorn-qr", "sinkhorn-polar"],
    )
    def test_matches_the_dense_reference(
        self, sinkhorn_steps: int, frame_update: FrameUpdate | None
    ) -> None:
        m, u_p, v_p, u, v = _tangent(seed=10)
        xi = _dense(m, u_p, v_p, u, v)

        m_c, u_p_c, v_p_c, frame = _condition(
            m,
            u_p,
            v_p,
            u,
            v,
            sinkhorn_steps=sinkhorn_steps,
            tau=TAU,
            frame_update=frame_update,
        )

        expected, expected_frame = _dense_condition(
            xi, u, v, sinkhorn_steps=sinkhorn_steps, frame_update=frame_update
        )
        torch.testing.assert_close(_dense(m_c, u_p_c, v_p_c, u, v), expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            frame @ frame.transpose(-2, -1),
            expected_frame @ expected_frame.transpose(-2, -1),
            atol=1e-5,
            rtol=0,
        )

    def test_result_is_a_tangent_vector_at_the_same_point(self) -> None:
        m, u_p, v_p, u, v = _tangent(seed=11)

        _, u_p_c, v_p_c, _ = _condition(m, u_p, v_p, u, v, sinkhorn_steps=2)

        zeros = torch.zeros(RANK, RANK)
        torch.testing.assert_close(u.transpose(-2, -1) @ u_p_c, zeros, atol=1e-5, rtol=0)
        torch.testing.assert_close(v.transpose(-2, -1) @ v_p_c, zeros, atol=1e-5, rtol=0)

    def test_whitening_the_reduced_rows_whitens_the_spectrum(self) -> None:
        """Muon's ``sigma_i -> 1``: in the spectral frame the row norms *are* the spectrum.

        The tangent projection that follows pulls the whitened matrix back onto the
        tangent space, so the retracted step is only *closer* to flat, not exactly flat.
        """
        m, u_p, v_p, u, v = _tangent(seed=12)
        xi = _dense(m, u_p, v_p, u, v)
        frame = _spectral_frame(m, u_p, v_p, u, v)

        whitened = frame @ _dense_reduced(frame.transpose(-2, -1) @ xi, 0, TAU)

        raw = torch.linalg.svdvals(xi)[: 2 * RANK]
        assert float(raw.max() / raw.min()) > 3.0  # the raw spectrum is spread out
        torch.testing.assert_close(
            torch.linalg.svdvals(whitened)[: 2 * RANK], torch.ones(2 * RANK), atol=1e-4, rtol=0
        )
        step = torch.linalg.svdvals(_dense(*_condition(m, u_p, v_p, u, v)[:3], u, v))[: 2 * RANK]
        assert float(step.max() / step.min()) < float(raw.max() / raw.min())

    def test_whitening_is_transpose_symmetric(self) -> None:
        """``xi^T`` is the same tangent with ``(U, V)``, ``(M, M^T)`` and the blocks swapped."""
        m, u_p, v_p, u, v = _tangent(seed=13)

        m_c, u_p_c, v_p_c, _ = _condition(m, u_p, v_p, u, v)
        m_t, u_p_t, v_p_t, _ = _condition(m.transpose(-2, -1), v_p, u_p, v, u)

        torch.testing.assert_close(m_t.transpose(-2, -1), m_c, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(v_p_t, u_p_c, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(u_p_t, v_p_c, atol=1e-5, rtol=1e-5)

    def test_sinkhorn_is_not_transpose_symmetric(self) -> None:
        """The column balance acts on the input side, so it distinguishes the sides."""
        m, u_p, v_p, u, v = _tangent(seed=14)

        m_c, _, _, _ = _condition(m, u_p, v_p, u, v, sinkhorn_steps=3)
        m_t, _, _, _ = _condition(m.transpose(-2, -1), v_p, u_p, v, u, sinkhorn_steps=3)

        assert not torch.allclose(m_t.transpose(-2, -1), m_c, atol=1e-3)

    @pytest.mark.parametrize(
        ("sinkhorn_steps", "gap"), [(0, 0.0), (2, 1e-9)], ids=["exact", "ill-conditioned"]
    )
    def test_a_collapsed_direction_keeps_a_usable_frame_and_step(
        self, sinkhorn_steps: int, gap: float
    ) -> None:
        """A rank-deficient tangent still conditions in an orthonormal, spanning frame.

        The collapsed direction is where an unrepaired QR complement would leak out of
        ``U``'s orthogonal complement, making the reduction oblique.
        """
        m, u_p, v_p, u, v = _collapsed_tangent(seed=62, gap=gap)
        xi = _dense(m, u_p, v_p, u, v)

        m_c, u_p_c, v_p_c, frame = _condition(
            m, u_p, v_p, u, v, sinkhorn_steps=sinkhorn_steps, tau=TAU
        )

        assert all(bool(block.isfinite().all()) for block in (m_c, u_p_c, v_p_c, frame))
        torch.testing.assert_close(
            frame.transpose(-2, -1) @ frame, torch.eye(2 * RANK), atol=1e-5, rtol=0
        )
        torch.testing.assert_close(
            frame @ (frame.transpose(-2, -1) @ xi), xi, atol=1e-5, rtol=1e-5
        )
        zeros = torch.zeros(RANK, RANK)
        torch.testing.assert_close(u.transpose(-2, -1) @ u_p_c, zeros, atol=1e-5, rtol=0)
        torch.testing.assert_close(v.transpose(-2, -1) @ v_p_c, zeros, atol=1e-5, rtol=0)

    @pytest.mark.parametrize("sinkhorn_steps", [0, 2])
    def test_an_ill_conditioned_direction_is_floored_not_whitened(
        self, sinkhorn_steps: int
    ) -> None:
        """``tau`` is what keeps the near-null direction from being blown up to unit.

        Once a direction collapses, its row of the reduced coefficients is rounding
        noise (~1e-8 of the largest row here), and whitening it unfloored would hand
        that noise the same weight as a real direction -- an amplification of eight
        orders of magnitude, which the tangent projection then only partly absorbs.
        """
        m, u_p, v_p, u, v = _collapsed_tangent(seed=63, gap=1e-9)
        xi = _dense(m, u_p, v_p, u, v)
        frame = _spectral_frame(m, u_p, v_p, u, v)
        red = frame.transpose(-2, -1) @ xi
        rows = red.norm(dim=-1)
        assert float(rows[-1] / rows.max()) < 1e-6  # the collapsed direction is noise

        floored = _condition_reduced(red, sinkhorn_steps, TAU)
        unfloored = _condition_reduced(red, sinkhorn_steps, 0.0)

        assert float(floored.norm(dim=-1)[-1]) < 1e-6
        assert float(unfloored.norm(dim=-1)[-1]) > 0.1
        step = _condition(m, u_p, v_p, u, v, sinkhorn_steps=sinkhorn_steps, tau=TAU)
        assert all(bool(block.isfinite().all()) for block in step[:3])
        spectrum = torch.linalg.svdvals(_dense(*step[:3], u, v))[: 2 * RANK]
        assert float(spectrum[-1] / spectrum[0]) < 1e-6  # it stays out of the step

    def test_sinkhorn_step_matches_the_original_unrescaled_reference(self) -> None:
        """The full tangent projection retains the unrescaled recurrence."""
        torch.manual_seed(21)
        in_features, out_features = 64, 16
        u = torch.linalg.qr(torch.randn(out_features, RANK))[0]
        v = torch.linalg.qr(torch.randn(in_features, RANK))[0]
        g = torch.randn(out_features, in_features)
        m, u_p, v_p = _tangent_factored(g @ v, g.transpose(-2, -1) @ u, u, v)

        whitened = _dense(*_condition(m, u_p, v_p, u, v, sinkhorn_steps=0)[:3], u, v)
        balanced = _dense(*_condition(m, u_p, v_p, u, v, sinkhorn_steps=3)[:3], u, v)

        expected, _ = _dense_condition(_dense(m, u_p, v_p, u, v), u, v, sinkhorn_steps=3)
        torch.testing.assert_close(balanced, expected, atol=1e-5, rtol=1e-5)
        different = balanced / balanced.norm(), whitened / whitened.norm()
        assert not torch.allclose(*different, atol=1e-2)

    @pytest.mark.parametrize("sinkhorn_steps", [0, 2])
    def test_an_exactly_zero_tangent_conditions_to_zero(self, sinkhorn_steps: int) -> None:
        """Additive epsilon keeps an exactly zero tangent finite and zero."""
        _, _, _, u, v = _tangent(seed=16)
        zeros = (
            torch.zeros(RANK, RANK),
            torch.zeros(OUT_FEATURES, RANK),
            torch.zeros(IN_FEATURES, RANK),
        )

        m_c, u_p_c, v_p_c, frame = _condition(*zeros, u, v, sinkhorn_steps=sinkhorn_steps)

        for block, expected in zip((m_c, u_p_c, v_p_c), zeros, strict=True):
            torch.testing.assert_close(block, expected, atol=1e-12, rtol=0)
        assert bool(frame.isfinite().all())

    def test_a_non_finite_result_falls_back_to_the_raw_step(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Conditioning never turns a usable step into ``nan``."""
        m, u_p, v_p, u, v = _tangent(seed=17)
        base = _spectral_frame(m, u_p, v_p, u, v)
        monkeypatch.setattr(
            conditioning,
            "_project_factored",
            lambda *_: (m * torch.nan, u_p * torch.inf, v_p * torch.nan),
        )

        m_c, u_p_c, v_p_c, frame = _condition(m, u_p, v_p, u, v, frame_update="polar")

        torch.testing.assert_close(m_c, m, atol=0.0, rtol=0.0)
        torch.testing.assert_close(u_p_c, u_p, atol=0.0, rtol=0.0)
        torch.testing.assert_close(v_p_c, v_p, atol=0.0, rtol=0.0)
        torch.testing.assert_close(frame, base, atol=1e-6, rtol=0)

    def test_a_supplied_frame_is_used_without_an_update(self) -> None:
        """The supplied frame is used without an additional QR."""
        m, u_p, v_p, u, v = _tangent(seed=18)
        supplied = torch.linalg.qr(torch.randn(OUT_FEATURES, 2 * RANK))[0]

        m_c, u_p_c, v_p_c, frame = _condition(m, u_p, v_p, u, v, frame=supplied)

        torch.testing.assert_close(frame, supplied, atol=0, rtol=0)
        expected, _ = _dense_condition(_dense(m, u_p, v_p, u, v), u, v, frame=supplied)
        torch.testing.assert_close(_dense(m_c, u_p_c, v_p_c, u, v), expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("frame_update", ["qr", "polar"])
    def test_refinement_uses_the_supplied_frame_directly(self, frame_update: FrameUpdate) -> None:
        m, u_p, v_p, u, v = _tangent(seed=21, dtype=torch.float64)
        torch.manual_seed(22)
        supplied = torch.linalg.qr(torch.randn(OUT_FEATURES, 2 * RANK, dtype=u.dtype))[0]

        m_c, u_p_c, v_p_c, frame = _condition(
            m, u_p, v_p, u, v, frame=supplied, sinkhorn_steps=2, frame_update=frame_update
        )

        expected, expected_frame = _dense_condition(
            _dense(m, u_p, v_p, u, v),
            u,
            v,
            frame=supplied,
            sinkhorn_steps=2,
            frame_update=frame_update,
        )
        torch.testing.assert_close(
            _dense(m_c, u_p_c, v_p_c, u, v), expected, atol=1e-10, rtol=1e-10
        )
        torch.testing.assert_close(
            frame.transpose(-2, -1) @ frame,
            torch.eye(2 * RANK, dtype=u.dtype),
            atol=1e-10,
            rtol=0,
        )
        torch.testing.assert_close(
            frame @ frame.transpose(-2, -1),
            expected_frame @ expected_frame.transpose(-2, -1),
            atol=1e-10,
            rtol=0,
        )

    def test_a_tracked_frame_stays_orthonormal_and_keeps_moving(self) -> None:
        """Feeding the returned frame back in iterates the Procrustes update."""
        m, u_p, v_p, u, v = _tangent(seed=19)
        fresh = _spectral_frame(m, u_p, v_p, u, v)

        first = _condition(m, u_p, v_p, u, v, sinkhorn_steps=2, frame_update="polar")[3]
        second = _condition(
            m, u_p, v_p, u, v, frame=first, sinkhorn_steps=2, frame_update="polar"
        )[3]

        for frame in (first, second):
            torch.testing.assert_close(
                frame.transpose(-2, -1) @ frame, torch.eye(2 * RANK), atol=1e-5, rtol=0
            )
        assert not torch.allclose(first, fresh, atol=1e-3)
        assert not torch.allclose(second, first, atol=1e-6)

    def test_a_supplied_frame_needs_no_decomposition_or_host_scalar_read(self) -> None:
        tangent = _tangent(seed=27)
        supplied = torch.linalg.qr(torch.randn(OUT_FEATURES, 2 * RANK))[0]
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            _condition(*tangent, frame=supplied)

        unwanted = {
            "aten::linalg_qr",
            "aten::linalg_svd",
            "aten::_linalg_svd",
            "aten::_local_scalar_dense",
        }
        assert not unwanted.intersection(event.key for event in profile.key_averages())

    def test_rejects_a_frame_of_the_wrong_shape(self) -> None:
        m, u_p, v_p, u, v = _tangent(seed=20)

        with pytest.raises(ValueError, match=r"frame must have shape \(6, 4\)"):
            _condition(m, u_p, v_p, u, v, frame=torch.eye(OUT_FEATURES))


class TestRmsNormalized:
    def test_rescales_to_unit_rms(self) -> None:
        m, u_p, v_p, u, v = _tangent(seed=30)

        blocks = _rms_normalized(m, u_p, v_p)

        norm = float(_dense(*blocks, u, v).norm())
        assert norm == pytest.approx((OUT_FEATURES * IN_FEATURES) ** 0.5, rel=1e-5)

    def test_only_changes_the_scale(self) -> None:
        m, u_p, v_p, u, v = _tangent(seed=31)
        xi = _dense(m, u_p, v_p, u, v)

        blocks = _rms_normalized(m, u_p, v_p)

        scaled = _dense(*blocks, u, v)
        torch.testing.assert_close(scaled / scaled.norm(), xi / xi.norm(), atol=1e-6, rtol=1e-6)

    def test_amplifies_the_spectrum_by_shape_over_rank(self) -> None:
        """Why its ``lr`` is not a full-rank Muon / AdamW ``lr`` (see the docstring).

        A whitened full-rank step has unit singular values; here the same Frobenius
        budget ``sqrt(out * in)`` lands on at most ``2r`` directions, so each one comes
        out around ``sqrt(out * in / 2r)`` -- a shape- and rank-dependent amplification
        that has to be tuned out of ``lr`` rather than carried over.
        """
        m, u_p, v_p, u, v = _tangent(seed=32)

        blocks = _rms_normalized(*_condition(m, u_p, v_p, u, v)[:3])

        spectrum = torch.linalg.svdvals(_dense(*blocks, u, v))[: 2 * RANK]
        flat = (OUT_FEATURES * IN_FEATURES / (2 * RANK)) ** 0.5
        assert float(spectrum.min()) > 0.5 * flat  # every direction, not just the top one
        assert float(spectrum.max()) == pytest.approx(flat, rel=0.5)
        assert flat > 3.0  # and that is already far from the unit step of full-rank Muon

    def test_a_zero_step_stays_zero(self) -> None:
        zeros = (
            torch.zeros(RANK, RANK),
            torch.zeros(OUT_FEATURES, RANK),
            torch.zeros(IN_FEATURES, RANK),
        )

        blocks = _rms_normalized(*zeros)

        for block, expected in zip(blocks, zeros, strict=True):
            torch.testing.assert_close(block, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
class TestNormalizationParity:
    @pytest.mark.parametrize("sinkhorn_steps", [0, 1, 3])
    @pytest.mark.parametrize("tau", [0.0, TAU])
    @pytest.mark.parametrize(
        "kind", ["dense", "rank-one", "sparse", "near-null", "tiny", "zero", "large"]
    )
    def test_matches_reference_recurrence(
        self, device: str, dtype: torch.dtype, sinkhorn_steps: int, tau: float, kind: str
    ) -> None:
        red = torch.arange(1, 33, device=device, dtype=dtype).reshape(8, 4).mT
        if kind in ("rank-one", "sparse", "near-null", "zero"):
            red.zero_()
            if kind == "rank-one":
                red[0] = 1.0 / 8**0.5
            elif kind == "sparse":
                red[:, :4] = torch.eye(4, device=device, dtype=dtype)
            elif kind == "near-null":
                red[0, 0], red[1, 1] = 1.0, 1e-8
        elif kind == "tiny":
            red.mul_(1e-12)
        elif kind == "large":
            red.mul_(1e20)
        original = red.clone()

        result = _condition_reduced(red, sinkhorn_steps, tau)

        torch.testing.assert_close(
            result, _dense_reduced(red, sinkhorn_steps, tau), atol=0, rtol=0
        )
        torch.testing.assert_close(red, original, atol=0, rtol=0)

    def test_whitening_adds_epsilon_instead_of_clamping(
        self, device: str, dtype: torch.dtype
    ) -> None:
        red = torch.eye(2, device=device, dtype=dtype) * 1e-12
        torch.testing.assert_close(
            _whiten_rows(red, TAU), torch.eye(2, device=device, dtype=dtype) * 0.5, atol=0, rtol=0
        )

    def test_column_normalization_still_includes_sub_floor_rows(
        self, device: str, dtype: torch.dtype
    ) -> None:
        red = torch.diag(torch.tensor([1.0, 1e-8, 0.0, 0.0], device=device, dtype=dtype))
        assert float(_whiten_rows(red, TAU)[1, 1]) < 1e-7
        assert float(_condition_reduced(red, 1, TAU)[1, 1]) > 0.99

    @pytest.mark.parametrize("magnitude", [0.0, 1e-35, 1e-12, 1.0, 1e20])
    def test_rms_matches_reference_formula(
        self, device: str, dtype: torch.dtype, magnitude: float
    ) -> None:
        m = torch.diag(torch.tensor([magnitude, -magnitude], device=device, dtype=dtype)).mT
        u_p = torch.zeros(4, 2, device=device, dtype=dtype)
        v_p = torch.zeros_like(u_p)
        u_p[2, 0] = magnitude
        v_p[3, 1] = -magnitude
        norm = (m.pow(2).sum() + u_p.pow(2).sum() + v_p.pow(2).sum()).sqrt()
        scale = 4.0 / norm.clamp_min(1e-12)
        expected = (scale * m, scale * u_p, scale * v_p)

        result = _rms_normalized(m, u_p, v_p)

        for actual, reference in zip(result, expected, strict=True):
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
