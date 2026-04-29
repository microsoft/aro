from typing import Any

import torch
from torch import Tensor

from ._defs import GradientProcessor


class SignProcessor(GradientProcessor):
    """This implements the Sign processor that executes the Sign operations on gradients."""

    def __init__(self, eps: float):
        """Init method.

        Args:
            eps (float): The epsilon value to prevent division by zero.
        """
        self._eps = eps

    def process_grad(self, grad: Tensor, *, state_dict: dict[str, Any] | None = None) -> Tensor:
        """The processing function to apply sign normalization to gradients.

        This applies a soft-sign operation: grad / (|grad| + eps), which normalizes
        gradients while preserving their sign and smoothly handling small values.

        Args:
            grad (Tensor): Gradients or momentum.

        Returns:
            Tensor: The processed gradients.
        """
        return grad / (torch.abs(grad) + self._eps)  # soft version
