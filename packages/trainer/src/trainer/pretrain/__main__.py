import logging
import os
from typing import Any

import click
from devtools import debug
from torch.distributed import destroy_process_group, init_process_group, is_initialized

from trainer.config import ConfigContainer, ConfigServices
from trainer.datasets import TrainIteratorContainer
from trainer.datasets._services import ValDataloaderContainer
from trainer.models import ModelContainer
from trainer.optim import OptimContainer
from trainer.state import StateContainer
from trainer.utils import list_yaml_files

from ._train import TrainContainer

logger = logging.getLogger(__name__)
yaml_files = click.Choice(list_yaml_files("config"))


def log_on_main_worker(message: str, *args: Any, level: int = logging.INFO):
    """Logs a message only on the worker with number (rank) 0."""
    if os.getenv("RANK", "0") == "0":
        logger.log(level, message, *args)


def setup_logging():
    """Sets up logging for the training script."""
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


@click.command()
@click.option("--cfg", default="nanogpt", type=yaml_files)
@click.argument("overrides", nargs=-1)
def main(cfg: str, overrides: tuple[str, ...]) -> None:
    setup_logging()
    config_container = ConfigContainer("config", cfg, overrides)
    config = config_container.get(ConfigServices.BASE_CONFIG)
    wandb_run = config_container.get(ConfigServices.WANDB_RUN)
    log_on_main_worker(debug(config))  # displays config for logging

    model_container = ModelContainer(config)
    optim_container = OptimContainer(config, model_container)
    state_container = StateContainer(config, model_container, optim_container)
    train_container = TrainIteratorContainer(config, state_container)
    val_container = ValDataloaderContainer(config)
    try:
        if all(os.getenv(e) for e in ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]):
            init_process_group(backend="nccl", device_id=int(os.getenv("LOCAL_RANK", 0)))
        TrainContainer(
            config,
            model_container,
            state_container,
            train_container,
            val_container,
            wandb_run,
        ).train()
    finally:
        destroy_process_group() if is_initialized() else None
        wandb_run.finish() if wandb_run is not None else None


if __name__ == "__main__":
    main()
