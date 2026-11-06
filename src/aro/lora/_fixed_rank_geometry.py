"""Riemannian geometry of the embedded fixed-rank manifold.

Pure tensor helpers for tangent projection, retraction, and vector transport
used by fixed-rank optimizers. Every function here is a free function operating
on the SVD coordinates ``(U, D, V)`` of a point
``W = U diag(D) V^T`` on the fixed-rank manifold ``M_r = {W : rank(W) = r}``
-- as stored by :class:`~aro.lora.LoRAADBLinear` (``U = A``, ``V = B``) --
and never form the ambient ``out x in`` matrix.

The embedded Riemannian gradient of an ambient loss gradient ``G = dL/dW`` is
the tangent projection

    grad = P_T(G) = P_U G + G P_V - P_U G P_V,   P_U = U U^T, P_V = V V^T,

and a gradient step retracts by the rank-``r`` truncated SVD of ``W - lr *
grad`` (the metric projection retraction). Since ``W - lr*grad`` has rank
<= ``2r``, the retraction is a cheap ``2r x 2r`` SVD and the ``out x in``
matrix is never formed.
"""

import torch
from torch import Tensor


def _tangent_factored(
    gv: Tensor,
    gtu: Tensor,
    u: Tensor,
    v: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Decompose the tangent projection ``P_T(G)`` into its three factored blocks.

    A tangent vector of ``M_r`` at ``W = U diag(D) V^T`` is
    ``xi = U M V^T + U_p V^T + U V_p^T`` with ``U_p perp U`` and ``V_p perp V``;
    for ``xi = P_T(G)`` the blocks follow from the frame projections of ``G``::

        M   = U^T G V            = u^T gv               (r, r)
        U_p = (I - U U^T) G V    = gv  - U M            (out, r)
        V_p = (I - V V^T) G^T U  = gtu - V M^T          (in, r)

    Exact (no ``1/D`` recovery, no factor gradients) and never forms ``G``.

    Args:
        gv: ``G V``, shape (out, r).
        gtu: ``G^T U``, shape (in, r).
        u: Left frame ``U``, shape (out, r), orthonormal columns.
        v: Right frame ``V``, shape (in, r), orthonormal columns.

    Returns:
        ``(M, U_p, V_p)`` -- the core (r, r) and the two perpendicular blocks.
    """
    m = u.transpose(-2, -1) @ gv
    u_p = gv - u @ m
    v_p = gtu - v @ m.transpose(-2, -1)
    return m, u_p, v_p


def _retract(
    u: Tensor,
    d: Tensor,
    v: Tensor,
    m: Tensor,
    u_p: Tensor,
    v_p: Tensor,
    lr: float,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Metric-projection retraction of ``W - lr * xi`` back onto ``M_r``.

    ``W = U diag(D) V^T`` and ``xi = U M V^T + U_p V^T + U V_p^T`` both live in
    ``span(U, Qu) x span(V, Qv)`` where ``Qu Ru = U_p`` and ``Qv Rv = V_p`` are
    thin QRs, so ``W - lr*xi = [U, Qu] K [V, Qv]^T`` with the ``2r x 2r`` core::

        K = [[diag(D) - lr * M, -lr * Rv ^ T], [-lr * Ru, 0]]

    The rank-``r`` truncated SVD of ``W - lr*xi`` (the metric projection onto
    ``M_r``) is therefore the truncated SVD of ``K`` mapped back through the
    ``2r``-frames -- the ``out x in`` matrix is never formed.

    Before the thin QRs, ``U_p`` and ``V_p`` are projected against their current
    frames once more. This removes finite-precision leakage from the tangent
    decomposition and keeps ``[U, Qu]`` and ``[V, Qv]`` orthonormal.

    Args:
        u: Left frame (out, r), orthonormal columns.
        d: Signed diagonal coordinates (r,). The retraction accepts either sign.
        v: Right frame (in, r), orthonormal columns.
        m: Tangent core (r, r).
        u_p: Tangent block perpendicular to ``u`` (out, r).
        v_p: Tangent block perpendicular to ``v`` (in, r).
        lr: Step size.
        eps: Floor keeping the retracted singular values strictly positive.

    Returns:
        ``(U_new, D_new, V_new)`` of the retracted point, with orthonormal
        ``U_new`` (out, r) / ``V_new`` (in, r) and canonical nonnegative
        singular values ``D_new >= eps`` (r,). Input signs are absorbed into
        the returned singular frames by the SVD.
    """
    r = u.shape[1]
    # Remove finite-precision leakage into the current frames before constructing
    # the 2r-dimensional retraction bases. The tangent core already represents
    # these frame-parallel components.
    u_p = u_p - u @ (u.transpose(-2, -1) @ u_p)
    v_p = v_p - v @ (v.transpose(-2, -1) @ v_p)
    qu, ru = torch.linalg.qr(u_p)  # qu (out, r) perp U;  ru (r, r)
    qv, rv = torch.linalg.qr(v_p)  # qv (in, r)  perp V;  rv (r, r)

    k = torch.zeros(2 * r, 2 * r, device=u.device, dtype=u.dtype)
    k[:r, :r] = torch.diag(d) - lr * m
    k[:r, r:] = -lr * rv.transpose(-2, -1)
    k[r:, :r] = -lr * ru
    uk, sk, vkh = torch.linalg.svd(k)

    # Truncate to the top-r singular subspace -> retracted point.
    uk_r = uk[:, :r]  # (2r, r)
    vk_r = vkh[:r, :].transpose(-2, -1)  # (2r, r)
    u_new = torch.cat([u, qu], dim=1) @ uk_r  # (out, r), orthonormal
    v_new = torch.cat([v, qv], dim=1) @ vk_r  # (in, r), orthonormal
    d_new = sk[:r].clamp_min(eps)  # keep singular values strictly > 0
    return u_new, d_new, v_new


def _transport(
    u: Tensor,
    v: Tensor,
    u_new: Tensor,
    v_new: Tensor,
    m: Tensor,
    u_p: Tensor,
    v_p: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Transport a tangent vector to the retracted point by tangent projection.

    Given ``xi = U M V^T + U_p V^T + U V_p^T`` at ``(U, V)``, this returns the
    decomposition of ``P_T1(xi)`` at the new frames ``(U1, V1)``. Only the two
    frame projections of ``xi`` are needed (``u_overlap = U^T U1`` and
    ``v_overlap = V^T V1`` measure the frame rotation)::

        xi V1   = U (M v_overlap + V_p^T V1) + U_p v_overlap     (out, r)
        xi^T U1 = V (M^T u_overlap + U_p^T U1) + V_p u_overlap   (in, r)

    then ``M' = U1^T (xi V1)``, ``U_p' = (I - U1 U1^T)(xi V1)`` and
    ``V_p' = (I - V1 V1^T)(xi^T U1)``, i.e. :func:`_tangent_factored` applied to
    the projections. The ``out x in`` matrix is never formed.

    Args:
        u: Old left frame (out, r).
        v: Old right frame (in, r).
        u_new: New left frame (out, r).
        v_new: New right frame (in, r).
        m: Tangent core (r, r) at the old frames.
        u_p: Tangent block perpendicular to ``u`` (out, r).
        v_p: Tangent block perpendicular to ``v`` (in, r).

    Returns:
        ``(M', U_p', V_p')`` -- the decomposition at ``(u_new, v_new)``, with
        ``U_p' perp u_new`` and ``V_p' perp v_new``.
    """
    u_overlap = u.transpose(-2, -1) @ u_new  # U^T U1   (r, r)
    v_overlap = v.transpose(-2, -1) @ v_new  # V^T V1   (r, r)
    xi_v1 = u @ (m @ v_overlap + v_p.transpose(-2, -1) @ v_new) + u_p @ v_overlap
    xi_u1 = v @ (m.transpose(-2, -1) @ u_overlap + u_p.transpose(-2, -1) @ u_new) + v_p @ u_overlap
    return _tangent_factored(xi_v1, xi_u1, u_new, v_new)
