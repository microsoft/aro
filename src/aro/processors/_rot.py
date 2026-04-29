from typing import Any

import torch
from torch import Tensor

from ._defs import GradientProcessor
from .u import UProcessor


class RotationProcessor:
    """This implements the processor that will apply the optimal rotation to the input base processor This is the core processing function for ARO."""

    def __init__(self, base_processor: GradientProcessor, update_u: UProcessor) -> None:
        """Init function.

        Args:
            base_processor: The base gradient processing function, which represents the base optimizer.
            update_u: A function that takes in the gradient and current rotation matrix, and returns the updated rotation matrix.
        """
        self._base_processor = base_processor
        self._update_u = update_u

    def process_grad(
        self, G: Tensor, u: Tensor, update_subspace: bool, *, state_dict: dict[str, Any]
    ) -> tuple[Tensor, Tensor]:
        """This implements the core ARO update.

        The update is computed as Uf(U^TG) (left) or f(GU)U^T (right), where f is the base processing function, and U is the rotation matrix. The orientation of matmul (i.e. left or right) depends on
        the size of G. For G with shape mxn, and m<n, the orientation is left with U shape mxm; otherwise, right is the default choice with U shape nxn.
        For updating rotation U, U = qr(Gf(G^TU)) (left) or U = qr(G^Tf(GU)) (right). qr is replaced by SCQR factorization.

        Args:
            G: Input gradient or momentum.
            u: Rotation matrix.
            update_subspace: whether the rotation U should be updated
            state_dict: passed on to self._base_processor.process_grad and must be modified in-place.

        Returns:
            tuple[Tensor, Tensor]: The processed gradient and the updated rotation matrix.

        """
        work_dtype = torch.float32
        k = G.clone()
        original_dtype = k.dtype
        k = k.to(work_dtype)
        u = u.to(work_dtype)
        transpose_k = False

        if k.shape[-1] <= k.shape[-2]:
            k = k.transpose(-1, -2)
            transpose_k = True

        if update_subspace:
            u = self._update_u(u, k, transpose_k, self._base_processor, state_dict=state_dict)

        utk = k.transpose(-1, -2).matmul(u) if transpose_k else u.transpose(-1, -2).matmul(k)
        f = self._base_processor.process_grad(utk, state_dict=state_dict)
        x = f.matmul(u.transpose(-1, -2)) if transpose_k else u.matmul(f)

        # convert back
        x = x.to(original_dtype)
        u = u.to(original_dtype)

        return x, u
