from typing import Literal

from rats import apps

from trainer.config import BaseConfig, TrainingConfig
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


class StateContainer(apps.Container):
    _config: TrainingConfig
    _model_container: ModelContainer
    _optim_container: OptimContainer

    def __init__(
        self,
        config: BaseConfig,
        model_container: ModelContainer,
        optim_container: OptimContainer,
    ) -> None:
        self._config = config.training
        self._model_container = model_container
        self._optim_container = optim_container

    @apps.service(StateServices.TRAINING_STATE)
    def training_state(self) -> TrainingState:
        return self.get(_PrivateServices.strategy(self._config.strategy))

    @apps.service(_PrivateServices.strategy("none"))
    @apps.service(_PrivateServices.strategy("ddp"))
    def _full_state(self) -> FullState:
        step = 0
        model = self._model_container.get(ModelServices.DISTRIBUTED_MODEL)
        optimizers = self._optim_container.get(OptimServices.OPTIMIZERS)
        schedulers = self._optim_container.get(OptimServices.SCHEDULERS)
        state = FullState(step, model, optimizers, schedulers)
        return state.resume_from_checkpoint_if_exists(self._config.checkpoint_path / "latest.pt")

    @apps.service(_PrivateServices.strategy("fsdp"))
    @apps.service(_PrivateServices.strategy("fsdp2"))
    def _sharded_state(self) -> ShardedState:
        step = 0
        model = self._model_container.get(ModelServices.DISTRIBUTED_MODEL)
        optimizers = self._optim_container.get(OptimServices.OPTIMIZERS)
        schedulers = self._optim_container.get(OptimServices.SCHEDULERS)
        state = ShardedState(step, model, optimizers, schedulers)
        return state.resume_from_checkpoint_if_exists(self._config.checkpoint_path / "latest.pt")
