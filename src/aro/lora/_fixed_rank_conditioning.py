"""Spectral conditioning of a fixed-rank tangent step.

Pure tensor helpers that reshape a tangent vector of the fixed-rank manifold
``M_r`` before it is retracted by :func:`~aro.lora._fixed_rank_geometry._retract`.
Like the geometry helpers, every function here works on the factored tangent
decomposition at ``W = U diag(D) V^T``

    xi = U M V^T + U_p V^T + U V_p^T,   U^T U_p = 0,   V^T V_p = 0,

so the ambient ``out x in`` matrix is never formed and everything costs
``O((out + in) r^2 + r^3)``.

**Reduced form.** ``xi = L R^T`` has rank at most ``2r`` with ``L = [U M + U_p, U]``
(out, 2r) and ``R = [V, V_p]`` (in, 2r), so it is fully described by its
coefficients ``F^T xi`` (2r, in) in any orthonormal frame ``F`` (out, 2r) spanning
``col(xi)``. Conditioning applies a cheap map ``f`` to those coefficients and
projects the result back onto the tangent space::

    step = P_T( F f(F^T xi) ).

**Joint QR for fresh frames.** Instead of separate QR of ``U_p`` and ``V_p``,
:func:`_spectral_frame` factors ``[U, U_p]`` and ``[V, V_p]`` jointly.
The perpendicular blocks can be rank deficient, including zero, in a valid
tangent. A separate QR completes their missing directions without regard to
``U`` or ``V``: its columns can overlap the existing basis, so concatenating them
does not give the orthonormal basis required by the small-core SVD.

Joint QR makes the trailing columns perpendicular to the existing basis as well
as mutually orthonormal, up to rounding. We recompute the block coefficients in
that basis to preserve the represented tangent. This uses ``2r``-column QRs
instead of ``r``-column QRs, with more work but the same asymptotic complexity.
The decision concerns fresh frame construction only; supplied frames are used
directly, without a repair QR or an orthogonality check. The normalization
recurrence is independent of the frame construction.

**Whitening (Muon).** In the *spectral* frame ``F = Q`` -- the top-``2r`` left
singular basis of ``xi`` (:func:`_spectral_frame`) -- the coefficients are
``Q^T xi = Sigma V_sv^T``, whose row norms are exactly the singular values.
Normalizing rows therefore whitens the spectrum, i.e. it is Muon's
``sigma_i -> 1`` orthogonalization written in the ``2r``-dimensional tangent
coordinates, up to the additive ``1e-12`` in each norm denominator. The floor
``tau`` leaves rows below ``tau * max_row_norm`` unchanged in that whitening pass.

**Sinkhorn balancing.** With ``sinkhorn_steps > 0`` each sweep applies floored row
normalization followed by L2 normalization of every column (eig-Sinkhorn).
The row floor is reevaluated on each sweep; it does not exclude rows from column
normalization, which can amplify a row left unchanged by the preceding pass.
Both normalizations divide by ``norm + 1e-12``.
There is no final rescale: with nonzero columns and norms well above epsilon,
the result has Frobenius norm approximately ``sqrt(in)`` rather than the
``sqrt(2r)`` of full-rank row whitening. Learning-rate scales need not carry over
between those modes. Rows live in spectral directions and columns in the input
dimension, so Sinkhorn is not transpose-symmetric while fresh spectral whitening
alone is.

**Frame refinement and tracking.** ``frame_update`` replaces ``F`` by one
Procrustes step ``F <- update(xi f(F^T xi)^T)`` before conditioning: ``"qr"`` is
the CholQr-style whitening step (its fixed point without Sinkhorn is just the
spectral basis) and ``"polar"`` is the Stiefel polar factor, which ascends the
alignment ``<xi, F f(F^T xi)>``. Passing the frame returned by the previous call
back in tracks it along the training trajectory instead of re-deriving the
spectral basis every step; carrying the whole frame (rather than a rotation
expressed in the per-step SVD gauge) keeps the tracking gauge-coherent. Supplied
frames must be finite and have orthonormal columns. They are used directly:
the spectral construction and QR/polar refinement already produce orthonormal
frames, so no additional QR repair is applied.
"""

from typing import Literal

import torch
from torch import Tensor

type FrameUpdate = Literal["qr", "polar"]
"""Procrustes step used to refine the conditioning frame before it is applied."""

_EPS = 1e-12
"""Additive epsilon for row/column normalization; absolute norm floor for RMS."""


def _split_complement(frame: Tensor, block: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Split ``block`` (n, r) over an orthonormal ``frame`` (n, k) and a complement.

    QR-ing ``block`` on its own leans on ``block perp frame`` holding exactly. It does
    not where ``block`` is rank deficient: the Householder completion fills the missing
    directions with whatever is orthogonal to the columns it already emitted, which is
    unrelated to ``frame``, so ``[frame, Q]`` stops being orthonormal (exact deficiency)
    or degrades like ``eps / sigma_min`` (near deficiency). QR-ing ``[frame, block]``
    jointly cannot: ``frame`` has full column rank, so the leading ``k`` factor columns
    span it exactly and the trailing ``r`` are orthonormal *and* perpendicular to it
    whatever ``block`` does. The coefficients are then recomputed in that basis, which
    keeps the split exact (and absorbs any ``frame`` component ``block`` should not
    have had) rather than reusing the triangular factor of a different decomposition.

    Requires ``n >= k + r``, which :class:`~aro.lora.LoRAADBLinear` guarantees by
    enforcing ``2 * rank <= min(in_features, out_features)``.

    Args:
        frame: Orthonormal frame (n, k) to split against.
        block: Block (n, r) to decompose.

    Returns:
        ``(Q, R, S)`` with ``block = frame @ S + Q @ R``, ``Q`` (n, r) orthonormal and
        perpendicular to ``frame``, ``R`` (r, r) and ``S`` (k, r).
    """
    k = frame.shape[-1]
    q = torch.linalg.qr(torch.cat([frame, block], dim=-1))[0][..., k:]
    return q, q.transpose(-2, -1) @ block, frame.transpose(-2, -1) @ block


def _spectral_frame(m: Tensor, u_p: Tensor, v_p: Tensor, u: Tensor, v: Tensor) -> Tensor:
    """Top-``2r`` left singular basis ``Q`` (out, 2r) of the tangent vector.

    ``U`` and ``V`` already have orthonormal columns, so only the perpendicular blocks
    need orthonormalizing: with ``U_p = U S_up + Q_up R_up`` and ``V_p = V S_vp +
    Q_vp R_vp`` (:func:`_split_complement`) the frames ``[U, Q_up]`` / ``[V, Q_vp]``
    are orthonormal and

        xi = [U, Q_up] C [V, Q_vp]^T,   C = [[M + S_up + S_vp^T, R_vp^T], [R_up, 0]],

    so ``Q = [U, Q_up] A`` with ``A`` the left factor of the ``2r x 2r`` ``SVD(C)``.
    Each side costs one QR of a ``2r``-column matrix, so the whole frame is still
    ``O((out + in) r^2 + r^3)``, and ``V`` enters only through that complement.
    ``S_up = U^T U_p`` and ``S_vp = V^T V_p`` vanish for an exact tangent
    decomposition; carrying them makes ``C`` -- hence the frame -- exact for the
    ambient ``xi`` the blocks actually describe.

    Because ``[U, Q_up]`` is orthonormal by construction and ``A`` is orthogonal, the
    returned frame is orthonormal and spans ``col(xi)`` even when ``xi`` is exactly
    rank deficient (the surplus directions are then an arbitrary but valid completion).
    The subsequent row-whitening pass applies its relative floor to near-null rows;
    Sinkhorn's column normalization can still amplify them.

    Args:
        m: Tangent core ``M`` (r, r).
        u_p: Tangent block perpendicular to ``u`` (out, r).
        v_p: Tangent block perpendicular to ``v`` (in, r).
        u: Left frame ``U`` (out, r), orthonormal columns.
        v: Right frame ``V`` (in, r), orthonormal columns.

    Returns:
        The orthonormal left singular frame ``Q`` (out, 2r), columns ordered by
        descending singular value.
    """
    r = u.shape[-1]
    q_up, r_up, s_up = _split_complement(u, u_p)
    _, r_vp, s_vp = _split_complement(v, v_p)
    zero = torch.zeros(r, r, device=u.device, dtype=u.dtype)
    core = torch.cat(
        [
            torch.cat([m + s_up + s_vp.transpose(-2, -1), r_vp.transpose(-2, -1)], dim=-1),
            torch.cat([r_up, zero], dim=-1),
        ],
        dim=-2,
    )
    a, _, _ = torch.linalg.svd(core)
    return torch.cat([u, q_up], dim=-1) @ a


def _whiten_rows(red: Tensor, tau: float) -> Tensor:
    """Divide rows by ``norm + 1e-12``, leaving sub-floor rows unchanged.

    Rows below ``tau * max_row_norm`` are returned as they are in this pass. In the
    spectral frame the row norms are singular values, giving spectrum whitening
    with a relative rank floor. Later column normalization can amplify these rows.
    Norms use the input dtype, with its limitations for very large finite values.

    Args:
        red: Reduced coefficients (2r, in).
        tau: Floor as a fraction of the largest row norm, in ``[0, 1)``.

    Returns:
        The whitened coefficients, same shape as ``red``.

    Raises:
        ValueError: If ``tau`` is outside ``[0, 1)`` or non-finite.
    """
    if not 0.0 <= tau < 1.0:
        raise ValueError(f"tau must be finite and in [0, 1), got {tau}")
    norms = red.norm(dim=-1, keepdim=True)
    return torch.where(norms < tau * norms.max(), red, red / (norms + _EPS))


def _condition_reduced(red: Tensor, sinkhorn_steps: int, tau: float) -> Tensor:
    """Apply the reduced conditioner ``f`` to the coefficients ``red``.

    ``sinkhorn_steps == 0`` whitens the spectrum once (Muon); a positive value runs
    that many floored-row/column-normalize sweeps (eig-Sinkhorn). Each row pass
    reevaluates the relative floor on the current coefficients. Every column then
    divides by its L2 norm plus ``1e-12``, including entries in sub-floor rows.
    There is no fixed mask or final Frobenius rescale; Sinkhorn and whitening-only
    outputs need not have the same norm.

    Args:
        red: Reduced coefficients (2r, in).
        sinkhorn_steps: Nonnegative integer number of row/column sweeps
            (``0`` -> whitening only).
        tau: Relative row floor passed to :func:`_whiten_rows`.

    Returns:
        The conditioned coefficients, same shape as ``red``.

    Raises:
        ValueError: If ``sinkhorn_steps`` is not a nonnegative integer, or ``tau``
            is outside ``[0, 1)`` or non-finite.
    """
    if not isinstance(sinkhorn_steps, int) or sinkhorn_steps < 0:
        raise ValueError(f"sinkhorn_steps must be a nonnegative integer, got {sinkhorn_steps}")
    if sinkhorn_steps > 0:
        for _ in range(sinkhorn_steps):
            red = _whiten_rows(red, tau)
            red = red / (red.norm(dim=-2, keepdim=True) + _EPS)
    else:
        red = _whiten_rows(red, tau)
    return red


def _project_factored(
    red: Tensor, frame: Tensor, u: Tensor, v: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Decompose ``P_T(frame @ red)`` into its tangent blocks at ``(u, v)``.

    With ``X = frame @ red`` and ``F_U = frame^T U``::

        M   = U^T X V           = F_U^T (red V)
        U_p = (I - U U^T) X V   = frame (red V) - U M
        V_p = (I - V V^T) X^T U = red^T F_U - V M^T

    The (out, in) product ``X`` is never formed.

    Args:
        red: Conditioned coefficients (2r, in).
        frame: Conditioning frame ``F`` (out, 2r), orthonormal columns.
        u: Left frame ``U`` (out, r). v: Right frame ``V`` (in, r).

    Returns:
        ``(M, U_p, V_p)`` -- the tangent decomposition at ``(u, v)``.
    """
    red_v = red @ v
    ftu = frame.transpose(-2, -1) @ u
    m = ftu.transpose(-2, -1) @ red_v
    u_p = frame @ red_v - u @ m
    v_p = red.transpose(-2, -1) @ ftu - v @ m.transpose(-2, -1)
    return m, u_p, v_p


def _refined_frame(ell: Tensor, rmat: Tensor, red: Tensor, frame_update: FrameUpdate) -> Tensor:
    """One Procrustes refinement ``update(xi red^T)`` of the conditioning frame.

    ``xi red^T = L (red R)^T`` is a (out, 2r) matrix built without forming ``xi``;
    ``"qr"`` returns its orthonormal QR factor and ``"polar"`` its orthogonal polar
    factor. Both land on the Stiefel manifold, so the result is a valid frame.

    Args:
        ell: Left factor ``L = [U M + U_p, U]`` (out, 2r).
        rmat: Right factor ``R = [V, V_p]`` (in, 2r), with ``xi = L R^T``.
        red: Conditioned coefficients (2r, in) at the incoming frame.
        frame_update: ``"qr"`` (whitening step) or ``"polar"`` (alignment step).

    Returns:
        The refined frame (out, 2r), orthonormal columns.
    """
    update = ell @ (red @ rmat).transpose(-2, -1)
    if frame_update == "polar":
        a, _, bt = torch.linalg.svd(update, full_matrices=False)
        return a @ bt
    q, _ = torch.linalg.qr(update)
    return q


def _condition(
    m: Tensor,
    u_p: Tensor,
    v_p: Tensor,
    u: Tensor,
    v: Tensor,
    *,
    frame: Tensor | None = None,
    sinkhorn_steps: int = 0,
    tau: float = 1e-4,
    frame_update: FrameUpdate | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Condition a tangent vector and return it decomposed at the same point.

    Reduces ``xi`` to its coefficients in an orthonormal ``2r``-frame, whitens (and
    optionally Sinkhorn-balances) them, and projects the result back onto the tangent
    space at ``(u, v)`` -- see the module docstring for the math.

    Additive epsilon keeps zero inputs zero. The relative row floor applies within
    each whitening pass, not across subsequent column normalization. If any
    conditioned block or the used frame is non-finite, all blocks fall back to the raw
    input (a non-finite *input* still gives a non-finite output, which the caller is
    expected to have screened for). The returned frame falls back to the base frame.
    A supplied ``frame`` must be finite with orthonormal columns and is used directly,
    without an extra QR repair. Only its shape is validated here, avoiding a host
    synchronization to inspect tensor entries.

    Args:
        m: Tangent core ``M`` (r, r).
        u_p: Tangent block perpendicular to ``u`` (out, r).
        v_p: Tangent block perpendicular to ``v`` (in, r).
        u: Left frame ``U`` (out, r), orthonormal columns.
        v: Right frame ``V`` (in, r), orthonormal columns.
        frame: Finite conditioning frame (out, 2r) with orthonormal columns -- the
            frame returned by the previous call when tracking it across steps.
            ``None`` (the default) derives the fresh spectral basis of ``xi``.
        sinkhorn_steps: Nonnegative integer reduced Sinkhorn sweeps
            (``0`` -> spectrum whitening only).
        tau: Relative row floor in ``[0, 1)``; sub-floor rows are unchanged within
            each row-normalization pass, but still enter column normalization.
        frame_update: ``None`` to condition in ``frame`` directly, otherwise the
            Procrustes step refining it first (``"qr"`` or ``"polar"``).

    Returns:
        ``(M, U_p, V_p, F)`` -- the conditioned tangent decomposition at ``(u, v)``
        (so ``U_p perp u`` and ``V_p perp v``) and the frame it was conditioned in,
        which is what a caller tracking the frame should carry to the next call.

    Raises:
        ValueError: If ``sinkhorn_steps`` is not a nonnegative integer, ``tau`` is
            outside ``[0, 1)`` or non-finite, or ``frame`` has a shape other than
            ``(out, 2r)``.
    """
    out_features, r = u_p.shape
    if frame is not None and frame.shape != (out_features, 2 * r):
        raise ValueError(
            f"frame must have shape {(out_features, 2 * r)} for a rank-{r} adapter "
            + f"with {out_features} output features, got {tuple(frame.shape)}"
        )
    ell = torch.cat([u @ m + u_p, u], dim=-1)  # L, with xi = L R^T
    rmat = torch.cat([v, v_p], dim=-1)  # R
    rt = rmat.transpose(-2, -1)

    base = _spectral_frame(m, u_p, v_p, u, v) if frame is None else frame
    used = base
    if frame_update is not None:
        red_base = _condition_reduced((base.transpose(-2, -1) @ ell) @ rt, sinkhorn_steps, tau)
        used = _refined_frame(ell, rmat, red_base, frame_update)

    red = _condition_reduced((used.transpose(-2, -1) @ ell) @ rt, sinkhorn_steps, tau)
    m_c, u_p_c, v_p_c = _project_factored(red, used, u, v)

    finite = (
        m_c.isfinite().all()
        & u_p_c.isfinite().all()
        & v_p_c.isfinite().all()
        & used.isfinite().all()
    )
    return (
        torch.where(finite, m_c, m),
        torch.where(finite, u_p_c, u_p),
        torch.where(finite, v_p_c, v_p),
        torch.where(finite, used, base),
    )


def _rms_normalized(m: Tensor, u_p: Tensor, v_p: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Rescale a tangent vector to unit entrywise RMS, i.e. ``||xi||_F = sqrt(out * in)``.

    A single per-layer scalar that makes the step size depend on the layer shape rather
    than on the gradient norm. The three blocks are Frobenius-orthogonal, so
    ``||xi||_F^2 = ||M||^2 + ||U_p||^2 + ||V_p||^2`` and no ambient matrix is formed.
    The divisor is floored, so a vanishing vector scales to zero instead of infinity.
    This uses an input-dtype squared-norm reduction and clamped denominator,
    including the possibility of overflow for very large finite inputs.

    **The resulting ``lr`` is not comparable to a full-rank Muon / AdamW ``lr``.** Those
    take a step of rank ``min(out, in)``, so spreading ``||step||_F = sqrt(out * in)``
    over it leaves singular values of order ``sqrt(max(out, in))``; a tangent vector of
    ``M_r`` has rank at most ``2r``, so the same Frobenius budget concentrates on those
    few directions and each singular value comes out around ``sqrt(out * in / rank)``,
    i.e. amplified by ``sqrt(min(out, in) / rank) >= sqrt(min(out, in) / 2r)`` relative
    to a full-rank step of the same norm -- a factor set by the layer shape and the
    adapter rank, growing as the rank shrinks. Treat ``lr`` as a per-shape, per-rank
    quantity and tune it separately; do not carry one over from a full-rank optimizer.

    Args:
        m: Tangent core ``M`` (r, r).
        u_p: Tangent block perpendicular to the left frame (out, r).
        v_p: Tangent block perpendicular to the right frame (in, r).

    Returns:
        The rescaled ``(M, U_p, V_p)``.
    """
    fnorm = (m.pow(2).sum() + u_p.pow(2).sum() + v_p.pow(2).sum()).sqrt()
    scale = (u_p.shape[0] * v_p.shape[0]) ** 0.5 / fnorm.clamp_min(_EPS)
    return scale * m, scale * u_p, scale * v_p
