from abc import ABC
from typing import Literal

from pydantic.dataclasses import dataclass

SCHEDULERS = Literal["linear", "linear-cosine"]


@dataclass(frozen=True)
class SchedulerConfig(ABC):
    name: SCHEDULERS


@dataclass(frozen=True)
class LinearCosineSchConfig(SchedulerConfig):
    name: Literal["linear-cosine"]
    warmup_iters: int  # number of iterations for the linear warmup
    lr_decay_iters: int  # total number of iterations
    min_lr: float = 0.0  # minimum learning rate at the end of cosine annealing
