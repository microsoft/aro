import math

import torch

from ._defs import GradientProcessor


class ConstantNormProcessor(GradientProcessor):
    """This processor will normalize the norm of input gradient with two options.

    When re-normalize is set to True. This processor will first normalize the input norm to be 1 and then scale it with user-defined norm. This ensures
    the processed gradient will have a constant norm, equal to the user-defined norm.

    When re-normalize is False, it will simply scale the gradient with the user-defined norm. This does not guarantee the constant norm.

    """

    def __init__(self, eps: float):
        """Init method.

        Args:
            eps: The eps added to prevent divide 0 error.
        """
        self._eps = eps

    def process_grad(self, grad: torch.Tensor) -> torch.Tensor:
        """The processing function that normalize the norm of the input gradient based on the re-normalize flag and user-defined norm.

        Args:
            grad (torch.Tensor): Input gradient.

        Returns:
            torch.Tensor: Processed gradient.
        """
        original_dtype = grad.dtype
        original_device = grad.device
        grad = grad.float()
        norm = math.sqrt(float(grad.shape[-1] * grad.shape[-2]))

        cur_norm = (
            grad.norm(dim=(-1, -2), keepdim=True)
            if grad.ndim >= 2
            else grad.norm(dim=-1, keepdim=True)
        )
        normalized_grad = grad / (cur_norm + self._eps) * norm
        normalized_grad = normalized_grad.to(dtype=original_dtype, device=original_device)
        return normalized_grad
