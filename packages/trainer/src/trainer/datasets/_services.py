import os
from collections.abc import Iterator, Sized
from typing import Any, Literal

import rats.apps as apps
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import Sampler

from trainer.config._base import BaseConfig
from trainer.config.datasets import OWTConf
from trainer.state import StateContainer, StateServices

from ._owt import EvalOpenWebText as EvalOWT
from ._owt import TrainOpenWebText as TrainOWT

MODE = Literal["train", "validation"]
DATASET_NAMES = Literal["owt"]
DATASET_CONF = OWTConf


@apps.autoscope
class _PrivateServices:
    DISTRIBUTED_SAMPLER = apps.ServiceId[DistributedSampler[Any]]("distributed-sampler")
    EVAL_SAMPLER = apps.ServiceId[Sampler[Any] | None]("eval-sampler")

    @staticmethod
    def dataset(name: DATASET_NAMES) -> apps.ServiceId[Dataset[Any]]:
        return apps.ServiceId[Dataset[Any]](f"{name}-dataset")


class DatasetServices:
    DATASET = apps.ServiceId[Dataset[Any]]("dataset")
    TRAIN_ITERATOR = apps.ServiceId[Iterator[Any]]("train-iterator")
    VAL_DATALOADER = apps.ServiceId[DataLoader[Any]]("val-dataloader")


class TrainIteratorContainer(apps.Container):
    _config: BaseConfig

    def __init__(self, config: BaseConfig, state_container: StateContainer) -> None:
        self._config = config
        self._state_container = state_container

    @apps.service(DatasetServices.DATASET)
    def _dataset(self) -> Dataset[Any]:
        """Returns the dataset based on the configuration name."""
        return self.get(_PrivateServices.dataset(self._config.data.name))

    @apps.service(_PrivateServices.dataset("owt"))
    def _train_owt(self) -> Dataset[Any]:
        assert isinstance(self._config.data, OWTConf)
        return TrainOWT(
            path=self._config.data.path,
            batch_size=self._config.training.batch_size,
            max_length=self._config.model.sequence_len,
            seed=self._config.seed,
            worker_id=int(os.getenv("RANK", "0")),
        )

    @apps.service(DatasetServices.TRAIN_ITERATOR)
    def _iterator_from_iterable_ds(self) -> Iterator[Any]:
        data_cfg = self._config.data
        batch_size = self._config.training.batch_size if data_cfg.enable_batch_dl else None
        iterable_ds = self.get(DatasetServices.DATASET)
        num_workers = data_cfg.num_workers
        prefetch_factor = data_cfg.prefetch_factor if num_workers > 0 else None
        dl = DataLoader(
            iterable_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=data_cfg.pin_memory and torch.cuda.is_available(),
        )
        i = 0
        iterable_dl = iter(dl)
        start_at_step = self._state_container.get(StateServices.TRAINING_STATE).step
        while i < start_at_step:
            next(iterable_dl)
            i += 1
        return iterable_dl


class ValDataloaderContainer(apps.Container):
    _config: BaseConfig

    def __init__(self, config: BaseConfig) -> None:
        self._config = config

    @apps.service(DatasetServices.DATASET)
    def _dataset(self) -> Dataset[Any]:
        """Returns the dataset based on the configuration name."""
        return self.get(_PrivateServices.dataset(self._config.data.name))

    @apps.service(_PrivateServices.dataset("owt"))
    def _val_owt(self) -> Dataset[Any]:
        assert isinstance(self._config.data, OWTConf)
        path = self._config.data.path
        batch_size = self._config.training.batch_size
        max_length = self._config.model.sequence_len
        eval_size = self._config.data.eval_size
        return EvalOWT(path, batch_size, eval_size=eval_size, max_length=max_length)

    @apps.service(_PrivateServices.DISTRIBUTED_SAMPLER)
    def _distributed_sampler(self) -> DistributedSampler[Any]:
        """Returns sampler for validation datasets for distributed settings."""
        ds = self.get(DatasetServices.DATASET)
        assert isinstance(ds, Sized)  # for finite datasets only
        return DistributedSampler(
            ds,
            num_replicas=int(os.getenv("WORLD_SIZE", "1")),
            rank=int(os.getenv("RANK", "0")),
            shuffle=False,
            drop_last=False,
        )

    @apps.service(_PrivateServices.EVAL_SAMPLER)
    def _sampler(self) -> Sampler[Any] | None:
        """Returns the appropriate sampler based on the configuration and distributed setting."""
        if torch.distributed.is_initialized():
            return self.get(_PrivateServices.DISTRIBUTED_SAMPLER)

    @apps.service(DatasetServices.VAL_DATALOADER)
    def _val_dataloader(self) -> DataLoader[Any]:
        """Returns a validation dataloader wrapped in a distributed sampler if needed."""
        data_cfg = self._config.data
        batch_size = self._config.training.batch_size if data_cfg.enable_batch_dl else None
        num_workers = data_cfg.num_workers
        prefetch_factor = data_cfg.prefetch_factor if num_workers > 0 else None
        dataset = self.get(DatasetServices.DATASET)
        sampler = self.get(_PrivateServices.EVAL_SAMPLER)
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=data_cfg.pin_memory and torch.cuda.is_available(),
        )
