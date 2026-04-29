from abc import ABC
from typing import Literal

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
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
