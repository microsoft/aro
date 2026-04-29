from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, Self

import torch.nn as nn
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class TrainingState(Stateful, Protocol):
    """This object contains all the state that needs to be checkpointed to resume training."""

    step: int
    model: nn.Module
    optimizers: Iterable[Optimizer]
    schedulers: Iterable[LRScheduler]

    def save_checkpoint(self, checkpoint_path: Path) -> None: ...
    def resume_from_checkpoint_if_exists(self, checkpoint_path: Path) -> Self: ...
