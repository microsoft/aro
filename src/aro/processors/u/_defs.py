from abc import abstractmethod
from typing import Any, Protocol

from torch import Tensor

from aro.processors import GradientProcessor


class UProcessor(Protocol):
    """Protocol for rotation matrix U update processors."""

    @abstractmethod
    def __call__(
        self,
        u: Tensor,
        k: Tensor,
        transpose_k: bool,
        base_processor: GradientProcessor,
        *,
        state_dict: dict[str, Any],
    ) -> Tensor:
        """Update the rotation matrix U.

        Returns:
            Tensor: The updated rotation matrix U.
        """
