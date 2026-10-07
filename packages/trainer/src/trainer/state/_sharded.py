import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from trainer.utils.randomness import rng_state_capture, rng_state_restore

from ._defs import TrainingState

logger = logging.getLogger(__name__)


def log_on_main_worker(message: str, *args: Any, level: int = logging.INFO):
    """Logs a message only on the worker with number (rank) 0."""
    if int(os.getenv("RANK", "0")) == 0:
        logger.log(level, message, *args)


@dataclass
class ShardedState(TrainingState):
    step: int
    model: nn.Module
    optimizers: Iterable[Optimizer]
    schedulers: Iterable[LRScheduler]

    def _initialize_optimizer_states(self) -> None:
        for optimizer in self.optimizers:
            initialize = getattr(optimizer, "ensure_state_initialized", None)
            if callable(initialize):
                initialize()

    def state_dict(self) -> dict[str, Any]:
        self._initialize_optimizer_states()
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizers)
        return {
            "step": self.step,
            "model": model_state_dict,
            "optimizers": optimizer_state_dict,
            "schedulers": [sch.state_dict() for sch in self.schedulers] if self.schedulers else [],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.step = state_dict["step"]
        set_state_dict(
            self.model,
            self.optimizers,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizers"],
        )
        for sch, sch_state in zip(self.schedulers, state_dict["schedulers"], strict=True):
            sch.load_state_dict(sch_state)

    def save_checkpoint(self, checkpoint_path: Path) -> None:
        """Saves a checkpoint to the given path."""
        logger.info(f"Checkpointing at step {self.step}.")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        state = {"state_dict": self.state_dict(), "rng_state": rng_state_capture()}
        dcp.save(state, checkpoint_id=checkpoint_path)

    def resume_from_checkpoint_if_exists(self, checkpoint_path: Path) -> Self:
        """Resumes training from a checkpoint if it exists."""
        if checkpoint_path.exists():
            log_on_main_worker("Resuming training from %s.", checkpoint_path)
            state = {"state_dict": self.state_dict(), "rng_state": rng_state_capture()}
            dcp.load(state_dict=state, checkpoint_id=checkpoint_path)
            self.load_state_dict(state["state_dict"])
            rng_state_restore(state["rng_state"])
            log_on_main_worker("Starting from step %d.", self.step)
        return self
