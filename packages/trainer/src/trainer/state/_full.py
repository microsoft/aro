import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler

from trainer.utils.device import get_device
from trainer.utils.randomness import rng_state_capture, rng_state_restore

from ._defs import TrainingState

logger = logging.getLogger(__name__)


def log_on_main_worker(message: str, *args: Any, level: int = logging.INFO):
    """Logs a message only on the worker with number (rank) 0."""
    if int(os.getenv("RANK", "0")) == 0:
        logger.log(level, message, *args)


@dataclass
class FullState(TrainingState):
    step: int
    model: nn.Module
    optimizers: Iterable[torch.optim.Optimizer]
    schedulers: Iterable[LRScheduler]

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "schedulers": [sch.state_dict() for sch in self.schedulers],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.step = state_dict["step"]
        self.model.load_state_dict(state_dict["model"])
        for optim, optim_state in zip(self.optimizers, state_dict["optimizers"], strict=True):
            optim.load_state_dict(optim_state)
        for sch, sch_state in zip(self.schedulers, state_dict["schedulers"], strict=True):
            sch.load_state_dict(sch_state)

    def save_checkpoint(self, checkpoint_path: Path) -> None:
        """Saves a checkpoint to the given path."""
        logger.info(f"Checkpointing at step {self.step}.")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        rng_state = rng_state_capture()
        torch.save({"state_dict": self.state_dict(), "rng_state": rng_state}, checkpoint_path)

    def resume_from_checkpoint_if_exists(self, checkpoint_path: Path) -> Self:
        """Resumes training from a checkpoint if it exists."""
        if checkpoint_path.exists():
            log_on_main_worker("Resuming training from %s.", checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=get_device())
            self.load_state_dict(checkpoint["state_dict"])
            rng_state_restore(checkpoint["rng_state"])
            log_on_main_worker("Starting from step %d.", self.step)
        return self
