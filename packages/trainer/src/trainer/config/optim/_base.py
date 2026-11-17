from abc import ABC
from typing import Literal

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


# Unknown fields are rejected (and inherited by every optimizer config) so that switching
# `optimizer.name` across optimizer families cannot silently keep another optimizer's settings.
@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class BaseOptimizerConfig(ABC):
    name: str
    lr: float = 1e-3
    weight_decay: float = 0.0
    eps: float = 1e-8


@dataclass(frozen=True)
class SinkhornConfig:
    sinkhorn_steps: int = 5
    eps: float = 1e-8


@dataclass(frozen=True)
class AdamConfig:
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    eps: float = 1e-8
    correct_bias: bool = True


@dataclass(frozen=True)
class AdamWConfig(BaseOptimizerConfig, AdamConfig):
    name: Literal["adamw"] = "adamw"
