import os
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import rats.apps as apps
import torch.distributed as dist
import wandb
from hydra import compose, initialize_config_dir
from omegaconf import DictKeyType, OmegaConf
from pydantic import ValidationError
from wandb.wandb_run import Run

from ._base import BaseConfig

DICT_TYPE = dict[DictKeyType, Any] | list[Any] | str | Any | None


@apps.autoscope
class ConfigServices:
    PROJECT_ROOT = apps.ServiceId[Path]("project-root")
    DICT_TYPE_CONFIG = apps.ServiceId[DICT_TYPE]("dict-type-config")
    BASE_CONFIG = apps.ServiceId[BaseConfig]("base-config")
    WANDB_RUN = apps.ServiceId[Run | None]("wandb-run")


class ConfigContainer(apps.Container):
    def __init__(self, config_dir: str, config_name: str, overrides: Sequence[str]) -> None:
        self._config_dir = config_dir
        self._config_name = config_name
        self._overrides = list(overrides)

    @apps.service(ConfigServices.PROJECT_ROOT)
    def _project_root(self) -> Path:
        current_path = Path(__file__)
        for parent in [current_path, *current_path.parents]:
            if (parent / "uv.lock").exists():
                return parent
        raise FileNotFoundError(
            "Could not find 'uv.lock' in the current directory or any parent directories."
        )

    @apps.service(ConfigServices.DICT_TYPE_CONFIG)
    def _dict_config(self) -> DICT_TYPE:
        project_root = self.get(ConfigServices.PROJECT_ROOT)
        config_dir = str(project_root / self._config_dir)
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name=self._config_name, overrides=list(self._overrides))
            return OmegaConf.to_container(cfg, resolve=True)

    @apps.service(ConfigServices.BASE_CONFIG)
    def _base_config(self) -> BaseConfig:
        dict_config = self.get(ConfigServices.DICT_TYPE_CONFIG)
        try:
            return BaseConfig(**cast(dict[str, Any], dict_config))
        except ValidationError as error:
            if any(err["loc"][:1] == ("optimizer",) for err in error.errors()):
                error.add_note(
                    "Hint: select an optimizer recipe with `optimizer=<name>` (see "
                    + "config/optimizer/); `optimizer.name=<name>` only switches between variants "
                    + "of the same optimizer, e.g. arosinkhorn and mixed-arosinkhorn."
                )
            raise

    @apps.service(ConfigServices.WANDB_RUN)
    def _wandb_run(self) -> Run | None:
        config = self.get(ConfigServices.BASE_CONFIG)
        flag = (dist.is_initialized() and dist.get_rank() == 0) or int(os.getenv("RANK", 0)) == 0
        if config.logger and flag:
            project = config.project.name
            entity = config.logger.entity
            mode = config.logger.mode
            run_name = config.logger.run_name
            wandb.login(host=config.logger.host)
            run = wandb.init(entity, project, config=asdict(config), mode=mode, name=run_name)
            return run
        return None
