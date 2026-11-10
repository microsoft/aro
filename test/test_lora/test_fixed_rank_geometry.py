"""Tests for the fixed-rank manifold geometry helpers in ``aro.lora._fixed_rank_geometry``.

These helpers never form the ``out x in`` ambient gradient, so every check here
compares the factored tangent projection, retraction, and transport against an
explicit dense reference.
"""

import torch
import torch.testing
from torch import Tensor

from aro.lora._fixed_rank_geometry import _retract, _tangent_factored, _transport

IN_FEATURES = 8
OUT_FEATURES = 6
RANK = 2


def _dense_tangent(g: Tensor, u: Tensor, v: Tensor) -> Tensor:
    """Dense tangent projection ``P_U G + G P_V - P_U G P_V``."""
    pu = u @ u.transpose(-2, -1)
    pv = v @ v.transpose(-2, -1)
    return pu @ g + g @ pv - pu @ g @ pv


def _truncate(w: Tensor, rank: int) -> Tensor:
    """Rank-``rank`` truncated SVD of ``w`` (the metric projection onto ``M_r``)."""
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    return (u[:, :rank] * s[:rank]) @ vh[:rank]


class TestTangentFactored:
    """The factored tangent decomposition against a dense projection reference."""

    @staticmethod
    def _random_point(seed: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        torch.manual_seed(seed)
        u = torch.linalg.qr(torch.randn(OUT_FEATURES, RANK))[0]
        v = torch.linalg.qr(torch.randn(IN_FEATURES, RANK))[0]
        d = torch.tensor([1.0, 0.5])
        g = torch.randn(OUT_FEATURES, IN_FEATURES)
        return u, d, v, g

    def test_matches_dense_projection(self) -> None:
        u, _, v, g = self._random_point(seed=20)

        m, u_p, v_p = _tangent_factored(g @ v, g.transpose(-2, -1) @ u, u, v)

        xi = u @ m @ v.transpose(-2, -1) + u_p @ v.transpose(-2, -1) + u @ v_p.transpose(-2, -1)
        torch.testing.assert_close(xi, _dense_tangent(g, u, v), atol=1e-5, rtol=1e-5)

    def test_blocks_are_perpendicular_to_the_frames(self) -> None:
        u, _, v, g = self._random_point(seed=21)

        _, u_p, v_p = _tangent_factored(g @ v, g.transpose(-2, -1) @ u, u, v)

        zeros = torch.zeros(RANK, RANK)
        torch.testing.assert_close(u.transpose(-2, -1) @ u_p, zeros, atol=1e-5, rtol=0)
        torch.testing.assert_close(v.transpose(-2, -1) @ v_p, zeros, atol=1e-5, rtol=0)

    def test_core_matches_both_frame_contractions(self) -> None:
        """``M = U^T G V`` is also ``(V^T G^T U)^T`` -- the shared core of both blocks."""
        u, _, v, g = self._random_point(seed=22)

        m, _, _ = _tangent_factored(g @ v, g.transpose(-2, -1) @ u, u, v)

        expected = v.transpose(-2, -1) @ g.transpose(-2, -1) @ u
        torch.testing.assert_close(m, expected.transpose(-2, -1), atol=1e-5, rtol=1e-5)

    def test_projection_is_idempotent(self) -> None:
        u, _, v, g = self._random_point(seed=23)
        xi = _dense_tangent(g, u, v)

        m, u_p, v_p = _tangent_factored(xi @ v, xi.transpose(-2, -1) @ u, u, v)

        again = u @ m @ v.transpose(-2, -1) + u_p @ v.transpose(-2, -1) + u @ v_p.transpose(-2, -1)
        torch.testing.assert_close(again, xi, atol=1e-5, rtol=1e-5)

    def test_zero_gradient_gives_zero_tangent(self) -> None:
        u, _, v, _ = self._random_point(seed=24)
        zeros_out = torch.zeros(OUT_FEATURES, RANK)
        zeros_in = torch.zeros(IN_FEATURES, RANK)

        m, u_p, v_p = _tangent_factored(zeros_out, zeros_in, u, v)

        torch.testing.assert_close(m, torch.zeros(RANK, RANK), atol=0.0, rtol=0.0)
        torch.testing.assert_close(u_p, zeros_out, atol=0.0, rtol=0.0)
        torch.testing.assert_close(v_p, zeros_in, atol=0.0, rtol=0.0)


class TestRetraction:
    """The ``2r x 2r`` retraction against a dense truncated SVD."""

    @staticmethod
    def _setup(seed: int, d: list[float] | None = None) -> tuple[Tensor, ...]:
        torch.manual_seed(seed)
        u = torch.linalg.qr(torch.randn(OUT_FEATURES, RANK))[0]
        v = torch.linalg.qr(torch.randn(IN_FEATURES, RANK))[0]
        d_t = torch.tensor(d if d is not None else [1.0, 0.5])
        g = torch.randn(OUT_FEATURES, IN_FEATURES)
        m, u_p, v_p = _tangent_factored(g @ v, g.transpose(-2, -1) @ u, u, v)
        return u, d_t, v, m, u_p, v_p

    def test_matches_dense_truncated_svd(self) -> None:
        u, d, v, m, u_p, v_p = self._setup(seed=30)
        lr = 0.2
        w = (u * d) @ v.transpose(-2, -1)
        xi = u @ m @ v.transpose(-2, -1) + u_p @ v.transpose(-2, -1) + u @ v_p.transpose(-2, -1)

        u_new, d_new, v_new = _retract(u, d, v, m, u_p, v_p, lr=lr, eps=1e-8)

        expected = _truncate(w - lr * xi, RANK)
        got = (u_new * d_new) @ v_new.transpose(-2, -1)
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)

    def test_singular_values_match_dense_truncated_svd(self) -> None:
        u, d, v, m, u_p, v_p = self._setup(seed=31)
        lr = 0.5
        w = (u * d) @ v.transpose(-2, -1)
        xi = u @ m @ v.transpose(-2, -1) + u_p @ v.transpose(-2, -1) + u @ v_p.transpose(-2, -1)

        _, d_new, _ = _retract(u, d, v, m, u_p, v_p, lr=lr, eps=1e-8)

        expected = torch.linalg.svdvals(w - lr * xi)[:RANK]
        torch.testing.assert_close(d_new, expected, atol=1e-5, rtol=1e-5)

    def test_signed_input_returns_equivalent_canonical_svd(self) -> None:
        u, d, v, m, u_p, v_p = self._setup(seed=31, d=[1.0, -0.5])
        lr = 0.2
        w = (u * d) @ v.transpose(-2, -1)
        xi = u @ m @ v.transpose(-2, -1) + u_p @ v.transpose(-2, -1) + u @ v_p.transpose(-2, -1)

        u_new, d_new, v_new = _retract(u, d, v, m, u_p, v_p, lr=lr, eps=1e-8)

        expected = _truncate(w - lr * xi, RANK)
        got = (u_new * d_new) @ v_new.transpose(-2, -1)
        assert bool((d_new >= 0).all())
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)

    def test_frames_stay_orthonormal(self) -> None:
        u, d, v, m, u_p, v_p = self._setup(seed=32)

        u_new, _, v_new = _retract(u, d, v, m, u_p, v_p, lr=0.3, eps=1e-8)

        eye = torch.eye(RANK)
        torch.testing.assert_close(u_new.transpose(-2, -1) @ u_new, eye, atol=1e-5, rtol=0)
        torch.testing.assert_close(v_new.transpose(-2, -1) @ v_new, eye, atol=1e-5, rtol=0)

    def test_reorthogonalizes_noisy_tangent_blocks(self) -> None:
        u, d, v, m, u_p, v_p = self._setup(seed=33)
        torch.manual_seed(34)
        u_noisy = u_p + u @ (1e-3 * torch.randn(RANK, RANK))
        v_noisy = v_p + v @ (1e-3 * torch.randn(RANK, RANK))

        u_new, d_new, v_new = _retract(u, d, v, m, u_noisy, v_noisy, lr=0.2, eps=1e-8)
        u_ref, d_ref, v_ref = _retract(u, d, v, m, u_p, v_p, lr=0.2, eps=1e-8)

        eye = torch.eye(RANK)
        torch.testing.assert_close(u_new.transpose(-2, -1) @ u_new, eye, atol=1e-5, rtol=0)
        torch.testing.assert_close(v_new.transpose(-2, -1) @ v_new, eye, atol=1e-5, rtol=0)
        got = (u_new * d_new) @ v_new.transpose(-2, -1)
        expected = (u_ref * d_ref) @ v_ref.transpose(-2, -1)
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)

    def test_zero_tangent_preserves_the_point(self) -> None:
        u, d, v, _, _, _ = self._setup(seed=33)
        zeros = torch.zeros(RANK, RANK)
        w = (u * d) @ v.transpose(-2, -1)

        u_new, d_new, v_new = _retract(
            u,
            d,
            v,
            zeros,
            torch.zeros(OUT_FEATURES, RANK),
            torch.zeros(IN_FEATURES, RANK),
            lr=0.1,
            eps=1e-8,
        )

        got = (u_new * d_new) @ v_new.transpose(-2, -1)
        torch.testing.assert_close(got, w, atol=1e-5, rtol=1e-5)

    def test_eps_floors_the_singular_values(self) -> None:
        """A collapsing direction is floored instead of leaving the manifold."""
        u, _, v, m, u_p, v_p = self._setup(seed=34, d=[1.0, 0.0])
        eps = 1e-3

        _, d_new, _ = _retract(u, torch.tensor([1.0, 0.0]), v, m * 0, u_p * 0, v_p * 0, 1e-4, eps)

        assert bool((d_new >= eps).all())
        torch.testing.assert_close(d_new[1], torch.tensor(eps), atol=0.0, rtol=0.0)

    def test_step_size_scales_the_displacement(self) -> None:
        """Small ``lr``: the retraction is the tangent step to first order."""
        u, d, v, m, u_p, v_p = self._setup(seed=35)
        lr = 1e-3
        w = (u * d) @ v.transpose(-2, -1)
        xi = u @ m @ v.transpose(-2, -1) + u_p @ v.transpose(-2, -1) + u @ v_p.transpose(-2, -1)

        u_new, d_new, v_new = _retract(u, d, v, m, u_p, v_p, lr=lr, eps=1e-8)

        got = (u_new * d_new) @ v_new.transpose(-2, -1)
        torch.testing.assert_close((got - w) / lr, -xi, atol=2e-2, rtol=2e-2)


class TestTransport:
    def test_transported_velocity_matches_dense_projection(self) -> None:
        u, d, v, m, u_p, v_p = TestRetraction._setup(seed=40)
        u_new, _, v_new = _retract(u, d, v, m, u_p, v_p, lr=0.2, eps=1e-8)
        xi = u @ m @ v.transpose(-2, -1) + u_p @ v.transpose(-2, -1) + u @ v_p.transpose(-2, -1)

        m_t, u_pt, v_pt = _transport(u, v, u_new, v_new, m, u_p, v_p)

        got = (
            u_new @ m_t @ v_new.transpose(-2, -1)
            + u_pt @ v_new.transpose(-2, -1)
            + u_new @ v_pt.transpose(-2, -1)
        )
        torch.testing.assert_close(got, _dense_tangent(xi, u_new, v_new), atol=1e-5, rtol=1e-5)

    def test_transport_to_the_same_point_is_the_identity(self) -> None:
        u, _, v, m, u_p, v_p = TestRetraction._setup(seed=41)

        m_t, u_pt, v_pt = _transport(u, v, u, v, m, u_p, v_p)

        torch.testing.assert_close(m_t, m, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(u_pt, u_p, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(v_pt, v_p, atol=1e-5, rtol=1e-5)

    def test_transported_blocks_are_perpendicular_to_the_new_frames(self) -> None:
        u, d, v, m, u_p, v_p = TestRetraction._setup(seed=42)
        u_new, _, v_new = _retract(u, d, v, m, u_p, v_p, lr=0.4, eps=1e-8)

        _, u_pt, v_pt = _transport(u, v, u_new, v_new, m, u_p, v_p)

        zeros = torch.zeros(RANK, RANK)
        torch.testing.assert_close(u_new.transpose(-2, -1) @ u_pt, zeros, atol=1e-5, rtol=0)
        torch.testing.assert_close(v_new.transpose(-2, -1) @ v_pt, zeros, atol=1e-5, rtol=0)
