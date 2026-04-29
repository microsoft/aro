from abc import abstractmethod
from typing import Any, Protocol

from torch import Tensor


class GradientProcessor(Protocol):
    """Protocol for gradient processors."""

    @abstractmethod
    def process_grad(self, grad: Tensor, *, state_dict: dict[str, Any] | None = None) -> Tensor:
        """Process the gradient tensor.

        state_dict is modified in place.

        Returns:
            Tensor: The processed gradient tensor.
        """

    def init_state_dict(self, grad: Tensor) -> dict[str, Any]:
        """Initialize the state dict of the optimizer.

        Returns:
            dict[str, Any]: The state dict.
        """
        return {}
