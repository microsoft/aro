from copy import deepcopy
from typing import Any

import torch
from torch import Tensor

from aro.processors import GradientProcessor

from ._defs import UProcessor


class PolarUpdate(UProcessor):
    """Update function for rotation matrix U using QR decomposition."""

    def __init__(self, steps: int) -> None:
        self._steps = steps

    def __call__(
        self,
        u: Tensor,
        k: Tensor,
        transpose_k: bool,
        base_processor: GradientProcessor,
        *,
        state_dict: dict[str, Any],
    ) -> Tensor:
        for _ in range(self._steps):
            sd = deepcopy(state_dict)
            utk = k.transpose(-1, -2).matmul(u) if transpose_k else u.transpose(-1, -2).matmul(k)
            f = base_processor.process_grad(utk, state_dict=sd)
            kf = k.matmul(f) if transpose_k else k.matmul(f.transpose(-1, -2))
            X, _, Yt = torch.linalg.svd(kf, full_matrices=False)
            u = X @ Yt
        return u
