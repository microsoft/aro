from typing import Any

from torch import Tensor

from aro.processors import GradientProcessor

from ._defs import UProcessor


class IdentityUpdate(UProcessor):
    """Run no update on U matrix and return."""

    def __call__(
        self,
        u: Tensor,
        k: Tensor,
        transpose_k: bool,
        base_processor: GradientProcessor,
        *,
        state_dict: dict[str, Any],
    ) -> Tensor:
        return u
