import math
from copy import deepcopy
from typing import Any

import torch
from torch import Tensor

from aro.processors import GradientProcessor

from ._defs import UProcessor

_scqr_safe_stats: dict[str, int] = {
    "calls": 0,
    "calls_fallback": 0,  # at least one batch item used QR fallback
    "calls_no_fallback": 0,  # all batch items used SCQR path
    "items_total": 0,  # total batch items processed across calls
    "items_fallback": 0,  # batch items that used QR fallback
    "items_no_fallback": 0,  # batch items that used SCQR path
}


def scqr_safe_take_stats(*, reset: bool = True) -> dict[str, int]:
    """Return cumulative ``scqr_safe`` stats since last call (optionally resetting)."""
    out = dict(_scqr_safe_stats)
    if reset:
        for k in _scqr_safe_stats:
            _scqr_safe_stats[k] = 0
    return out


def scqr_safe(A: torch.Tensor, jitter: float = 1e-7, cond_max: float = 1e12) -> torch.Tensor:
    """Shifted Cholesky QR with a per-batch-item QR fallback.

    Given an input tensor ``A`` with shape ``(*, m, n)`` (treating all leading
    dimensions as batch), this routine returns orthonormal ``Q``. It attempts a fast SCQR-based
    QR and only falls back to a standard QR factorization on the specific
    batch items where SCQR is numerically unsafe.

    Args:
        A (torch.Tensor):
            Real-valued tensor with shape ``(*, m, n)`` and ``A.ndim >= 2``.
            All leading dimensions are treated as batch.
        jitter (float): default=1e-7
            Relative regularization added.
            Increase if you frequently see SCQR failures.
        cond_max (float): default=1e12
            Threshold on a cheap condition proxy computed from ``diag(L)``, where L is obtained from the internal triangular factorization. If
            ``max(diag(L)) / min(diag(L))`` exceeds this value, the batch item is
            considered ill-conditioned and the QR fallback is used. Under the
            default setting this is a no-op; use smaller values to activate
            condition-based fallback.

    """
    if A.ndim < 2:
        raise ValueError("Input tensor must be at least 2D")
    *batch, m, n = A.shape
    B = math.prod(batch) if batch else 1
    A_flat = A.reshape(B, m, n)

    gram = A_flat.transpose(-2, -1).matmul(A_flat)  # (B, n, n)
    gram = 0.5 * (gram + gram.transpose(-2, -1))
    eps = jitter * gram.norm(dim=(-2, -1), keepdim=True)  # (B,1,1)
    eye_n = torch.eye(n, device=gram.device, dtype=gram.dtype).unsqueeze(0)
    gram += eps * eye_n

    L, info = torch.linalg.cholesky_ex(gram)  # L=(B,n,n), info=(B,)
    ok_scqr = info == 0
    ok_finite = torch.isfinite(L).all(dim=(-2, -1))
    diag_L = L.diagonal(dim1=-2, dim2=-1)  # (B, n)
    cond = diag_L.max(-1).values / diag_L.min(-1).values
    ok_cond = cond < cond_max
    use_scqr = ok_scqr & ok_finite & ok_cond  # (B,)

    _scqr_safe_stats["calls"] += 1
    _scqr_safe_stats["items_total"] += B

    Q_flat = torch.empty_like(A_flat)
    if use_scqr.all().item():
        Y = torch.linalg.solve_triangular(L, A_flat.transpose(-2, -1), upper=False)
        Q_scqr = Y.transpose(-2, -1).contiguous()  # (B, m, n)
        _scqr_safe_stats["calls_no_fallback"] += 1
        _scqr_safe_stats["items_no_fallback"] += B
        Q_flat.copy_(Q_scqr)
    else:
        _scqr_safe_stats["calls_fallback"] += 1
        idx_fail = torch.nonzero(~use_scqr, as_tuple=False).squeeze(1)
        num_fail = int(idx_fail.numel())
        _scqr_safe_stats["items_fallback"] += num_fail
        _scqr_safe_stats["items_no_fallback"] += B - num_fail

        A_fail = A_flat[idx_fail]  # (F, m, n)
        Q_fail, _ = torch.linalg.qr(A_fail)  # (F, m, n)
        Q_flat[idx_fail] = Q_fail.contiguous()

        idx_scqr = torch.nonzero(use_scqr, as_tuple=False).squeeze(1)
        if idx_scqr.numel() > 0:
            Y_scqr = torch.linalg.solve_triangular(
                L[idx_scqr], A_flat[idx_scqr].transpose(-2, -1), upper=False
            )
            Q_flat[idx_scqr] = Y_scqr.transpose(-2, -1).contiguous()

    # restore batch dims
    Q = Q_flat.reshape(*batch, m, n) if batch else Q_flat.squeeze(0)
    return Q.contiguous()


class SCQRUpdate(UProcessor):
    """Update function for rotation matrix U using SCQR decomposition."""

    def __init__(self, rotation_jitter: float = 1e-7) -> None:
        self._rotation_jitter = rotation_jitter

    def __call__(
        self,
        u: Tensor,
        k: Tensor,
        transpose_k: bool,
        base_processor: GradientProcessor,
        *,
        state_dict: dict[str, Any],
    ) -> Tensor:
        sd = deepcopy(state_dict)
        utk = k.transpose(-1, -2).matmul(u) if transpose_k else u.transpose(-1, -2).matmul(k)
        f = base_processor.process_grad(utk, state_dict=sd)
        kf = k.matmul(f) if transpose_k else k.matmul(f.transpose(-1, -2))
        return scqr_safe(kf, jitter=self._rotation_jitter)


# Compatibility names for the internal Cholesky-QR API.
CholQrUpdate = SCQRUpdate
chol_qr_safe = scqr_safe
chol_qr_safe_take_stats = scqr_safe_take_stats
