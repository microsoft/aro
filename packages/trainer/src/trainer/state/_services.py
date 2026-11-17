import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from rats import apps

from trainer.config import BaseConfig
from trainer.models import ModelContainer
from trainer.models._services import ModelServices
from trainer.optim import OptimContainer, OptimServices

from ._defs import TrainingState
from ._full import FullState
from ._sharded import ShardedState

STRATEGY = Literal["none", "ddp", "fsdp", "fsdp2"]


@apps.autoscope
class _PrivateServices:
    @staticmethod
    def strategy(strategy: STRATEGY) -> apps.ServiceId[TrainingState]:
        return apps.ServiceId[TrainingState](f"{strategy}-strategy")


@apps.autoscope
class StateServices:
    TRAINING_STATE = apps.ServiceId[TrainingState]("training-state")
    CHECKPOINT_DIR = apps.ServiceId[Path]("checkpoint-dir")


class StateContainer(apps.Container):
    _base_config: BaseConfig
    _model_container: ModelContainer
    _optim_container: OptimContainer

    def __init__(
        self,
        config: BaseConfig,
        model_container: ModelContainer,
        optim_container: OptimContainer,
    ) -> None:
        self._base_config = config
        self._model_container = model_container
        self._optim_container = optim_container

    @apps.service(StateServices.TRAINING_STATE)
    def _training_state(self) -> TrainingState:
        """Freshly initialized state; the trainer resumes it from CHECKPOINT_DIR if possible."""
        return self.get(_PrivateServices.strategy(self._base_config.training.strategy))

    @apps.service(StateServices.CHECKPOINT_DIR)
    def _checkpoint_dir(self) -> Path:
        """Directory with this run's checkpoints and its `latest` pointer.

        Unless training.checkpoint_path is set, the directory is named after the optimizer and a
        hash of the settings that affect training, so relaunching the same command resumes it,
        while runs with different settings never resume from each other's checkpoints.
        """
        config = self._base_config
        if config.training.checkpoint_path is not None:
            return config.training.checkpoint_path
        name = f"{config.optimizer.name}-{self._run_fingerprint()}"
        return config.compute_env.output_dir / "checkpoints" / name

    def _run_fingerprint(self) -> str:
        settings = asdict(self._base_config)
        # Logging, evaluation and checkpointing knobs do not change the training trajectory.
        settings.pop("logger")
        settings["compute_env"].pop("output_dir")
        for key in ("log_interval", "eval_interval", "checkpoint_interval", "checkpoint_path"):
            settings["training"].pop(key)
        settings["data"].pop("eval_size")
        payload = json.dumps(settings, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:8]

    @apps.service(_PrivateServices.strategy("none"))
    @apps.service(_PrivateServices.strategy("ddp"))
    def _full_state(self) -> FullState:
        model = self._model_container.get(ModelServices.DISTRIBUTED_MODEL)
        optimizers = self._optim_container.get(OptimServices.OPTIMIZERS)
        schedulers = self._optim_container.get(OptimServices.SCHEDULERS)
        return FullState(0, model, optimizers, schedulers)

    @apps.service(_PrivateServices.strategy("fsdp"))
    @apps.service(_PrivateServices.strategy("fsdp2"))
    def _sharded_state(self) -> ShardedState:
        model = self._model_container.get(ModelServices.DISTRIBUTED_MODEL)
        optimizers = self._optim_container.get(OptimServices.OPTIMIZERS)
        schedulers = self._optim_container.get(OptimServices.SCHEDULERS)
        return ShardedState(0, model, optimizers, schedulers)
