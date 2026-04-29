from typing import Any

from torch import Tensor

from ._defs import GradientProcessor


class SinkhornProcessor(GradientProcessor):
    """This implements the Sinkhorn processor that executes the Sinkhorn operations on gradients."""

    def __init__(self, sinkhorn_steps: int, eps: float):
        """Init method.

        Args:
            sinkhorn_steps (int): The number of Sinkhorn iterations steps.
            eps (float): The epsilon value to prevent division by zero.
        """
        self._eps = eps
        self._sinkhorn_steps = sinkhorn_steps

    def process_grad(self, grad: Tensor, *, state_dict: dict[str, Any] | None = None) -> Tensor:
        """The processing function to apply the Sinkhorn operation to gradients.

        Args:
            grad (Tensor): Gradients or momentum.

        Returns:
            Tensor: The processed gradients.
        """
        original_shape = grad.shape

        assert grad.ndim >= 2
        batch_dims = grad.ndim - 2
        if batch_dims > 0:
            grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])

        for _ in range(self._sinkhorn_steps):
            row_norms = grad.norm(dim=-1, keepdim=True) + self._eps
            grad = grad / row_norms
            col_norms = grad.norm(dim=-2, keepdim=True) + self._eps
            grad = grad / col_norms

        if grad.shape != original_shape:
            grad = grad.reshape(original_shape)

        return grad
