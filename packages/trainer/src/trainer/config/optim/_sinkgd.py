from typing import Literal

from pydantic.dataclasses import dataclass

from ._base import BaseOptimizerConfig


@dataclass(frozen=True)
class SinkGDConfig(BaseOptimizerConfig):
    """Source-compatible SinkGD defaults with an independent AdamW fallback.

    ``sinkgd`` selects all matrices; ``mixed-sinkgd`` excludes token embeddings
    and the prediction head. Alternating steps count individual passes, while
    simultaneous steps count joint row/column passes.
    """

    name: Literal["sinkgd", "mixed-sinkgd"] = "sinkgd"
    momentum: float = 0.95
    nesterov: bool = True
    sinkhorn_steps: int = 11
    normalization: Literal["alternating", "simultaneous"] = "alternating"
    final_row_pass: bool = False
    eps: float = 1e-20
    row_filter_threshold: float = 1e-3
    alpha: float = 0.2
    constant_norm: bool = True
    weight_decay: float = 0.0
    start_axis: Literal["row", "col"] = "row"
    adamw_betas: tuple[float, float] = (0.9, 0.95)
    adamw_eps: float = 1e-8
    adamw_weight_decay: float = 0.1

    def __post_init__(self) -> None:
        if self.final_row_pass and self.normalization != "simultaneous":
            raise ValueError("final_row_pass only applies to simultaneous normalization")
        if self.sinkhorn_steps < 1 or (
            self.normalization == "alternating" and self.sinkhorn_steps % 2 == 0
        ):
            raise ValueError(
                "sinkhorn_steps must be positive and odd for alternating normalization"
            )
