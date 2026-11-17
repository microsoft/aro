import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler

from trainer.utils.device import get_device
from trainer.utils.randomness import rng_state_capture, rng_state_restore

from ._defs import TrainingState

logger = logging.getLogger(__name__)
SHARED_FILE = "shared.pt"


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
        """Saves a checkpoint directory; every rank must call this.

        Rank 0 writes the replicated model and scheduler state; each rank writes its own optimizer
        and RNG state, because under DDP ARO optimizers keep each matrix's rotation state only on
        the rank that owns the matrix.
        """
        rank, world_size = _rank_and_world_size()
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            logger.info(f"Checkpointing at step {self.step}.")
            shared = {
                "step": self.step,
                "world_size": world_size,
                "model": self.model.state_dict(),
                "schedulers": [sch.state_dict() for sch in self.schedulers],
            }
            torch.save(shared, checkpoint_path / SHARED_FILE)
        local = {
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "rng_state": rng_state_capture(),
        }
        torch.save(local, checkpoint_path / _rank_file(rank))

    def resume_from_checkpoint_if_exists(self, checkpoint_path: Path) -> Self:
        """Resumes training from a checkpoint directory if it exists."""
        if not checkpoint_path.exists():
            return self
        rank, world_size = _rank_and_world_size()
        log_on_main_worker("Resuming training from %s.", checkpoint_path)
        device = get_device()
        shared = torch.load(checkpoint_path / SHARED_FILE, map_location=device)
        if shared["world_size"] != world_size:
            raise ValueError(
                f"{checkpoint_path} was saved with {shared['world_size']} processes; resume it "
                + f"with the same number of GPUs instead of {world_size}."
            )
        local = torch.load(checkpoint_path / _rank_file(rank), map_location=device)
        self.load_state_dict(
            {
                "step": shared["step"],
                "model": shared["model"],
                "optimizers": local["optimizers"],
                "schedulers": shared["schedulers"],
            }
        )
        rng_state_restore(local["rng_state"])
        log_on_main_worker("Starting from step %d.", self.step)
        return self


def _rank_and_world_size() -> tuple[int, int]:
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _rank_file(rank: int) -> str:
    return f"rank{rank:05d}.pt"
