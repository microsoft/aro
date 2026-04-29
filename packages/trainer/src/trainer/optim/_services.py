import logging
import os
from dataclasses import asdict
from typing import Any, Literal, cast

import torch
import torch.nn as nn
from rats import apps
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, SequentialLR

from aro.optim import AroAdam, AroSign, AroSinkhorn, Muon
from aro.processors.u import PolarUpdate, SCQRUpdate
from gpt import NanoChatGPT, NanoGPT
from trainer.config import BaseConfig
from trainer.config.optim import (
    AdamWConfig,
    AroAdamConfig,
    AroSignConfig,
    AroSinkhornConfig,
    MuonConfig,
    PolarConfig,
    SCQRConfig,
)
from trainer.config.schedulers import SCHEDULERS, LinearCosineSchConfig, SchedulerConfig
from trainer.models import ModelContainer, ModelServices

from ._scheduler import WarmupLR

logger = logging.getLogger(__name__)


def log_on_main_worker(message: str, *args: Any, level: int = logging.INFO):
    """Logs a message only on the worker with number (rank) 0."""
    if int(os.getenv("RANK", "0")) == 0:
        logger.log(level, message, *args)


UPDATE_U = Literal["scqr", "polar"]
MODEL_NAME = Literal["nanogpt", "nanochat"]
ARO_CONFIGS = AroSinkhornConfig | AroAdamConfig | AroSignConfig
OPT_CONFIGS = ARO_CONFIGS | AdamWConfig | MuonConfig
OPTIMIZERS = Literal[
    "adamw",
    "arosinkhorn",
    "aroadam",
    "arosign",
    "muon",
    "mixed-arosinkhorn",
    "mixed-aroadam",
    "mixed-arosign",
    "mixed-muon",
]


def _exclude_name(d: list[tuple[str, Any]]) -> dict[str, Any]:
    return {k: v for k, v in d if k not in ["name", "u_processor", "rank", "mode"]}


class _PrivateServices:
    @staticmethod
    def optimizers(optimizer: OPTIMIZERS) -> apps.ServiceId[list[torch.optim.Optimizer]]:
        return apps.ServiceId[list[torch.optim.Optimizer]](f"{optimizer}-optimizer")

    @staticmethod
    def schedulers(scheduler: SCHEDULERS) -> apps.ServiceId[list[LRScheduler]]:
        return apps.ServiceId[list[LRScheduler]](f"{scheduler}-scheduler")

    @staticmethod
    def u_processor(name: UPDATE_U) -> apps.ServiceId[Any]:
        return apps.ServiceId[Any](f"{name}-u-processor")

    @staticmethod
    def head_params(name: MODEL_NAME) -> apps.ServiceId[list[nn.Parameter]]:
        return apps.ServiceId[list[nn.Parameter]](f"{name}-head-params")


@apps.autoscope
class OptimServices:
    OPTIMIZERS = apps.ServiceId[list[torch.optim.Optimizer]]("optimizers")
    SCHEDULERS = apps.ServiceId[list[LRScheduler]]("schedulers")


class OptimContainer(apps.Container):
    _config: BaseConfig
    _optim_cfg: OPT_CONFIGS
    _model_container: ModelContainer

    def __init__(self, config: BaseConfig, model_container: ModelContainer):
        self._config = config
        self._optim_cfg = config.optimizer
        self._model_container = model_container

    @apps.service(OptimServices.OPTIMIZERS)
    def _optimizers(self) -> list[torch.optim.Optimizer]:
        return self.get(_PrivateServices.optimizers(self._optim_cfg.name))

    @apps.container()
    def _fulloptimizer_container(self) -> "_FullOptimizerContainer":
        return _FullOptimizerContainer(self._config, self._model_container)

    @apps.container()
    def _mixedoptimizer_container(self) -> "_MixedOptimizerContainer":
        return _MixedOptimizerContainer(self._config, self._model_container)

    @apps.container()
    def _schedulers_container(self) -> "SchedulersContainer":
        return SchedulersContainer(self._config, self)


class _FullOptimizerContainer(apps.Container):
    _config: BaseConfig
    _optim_cfg: OPT_CONFIGS
    _model_container: ModelContainer

    def __init__(self, config: BaseConfig, model_container: ModelContainer):
        self._config = config
        self._optim_cfg = config.optimizer
        self._model_container = model_container

    def _get_full_params_and_adamw(self) -> tuple[list[nn.Parameter], AdamW]:
        model = self._model_container.get(ModelServices.DISTRIBUTED_MODEL)
        model = cast(nn.Module, model.module if isinstance(model, DDP) else model)
        oned_params = [p for p in model.parameters() if p.requires_grad and p.ndim == 1]
        params = [
            p
            for p in model.parameters()
            if p.requires_grad and all(p is not q for q in oned_params)
        ]
        adamw_opt = AdamW(
            oned_params,
            lr=self._optim_cfg.lr,
            betas=(0.9, 0.95),
            weight_decay=self._optim_cfg.weight_decay,
            eps=self._optim_cfg.eps,
        )
        return params, adamw_opt

    @apps.container()
    def _common_container(self) -> "_CommonContainer":
        return _CommonContainer(self._config, self._model_container)

    @apps.service(_PrivateServices.optimizers("adamw"))
    def _adamw_optimizer(self) -> list[torch.optim.Optimizer]:
        model = self._model_container.get(ModelServices.DISTRIBUTED_MODEL)
        model = cast(nn.Module, model.module if isinstance(model, DDP) else model)
        params = [p for p in model.parameters() if p.requires_grad]
        lr = self._optim_cfg.lr
        optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
        return [optimizer]

    @apps.service(_PrivateServices.optimizers("arosinkhorn"))
    def _arosinkhorn_optimizer(self) -> list[torch.optim.Optimizer]:
        assert isinstance(self._optim_cfg, AroSinkhornConfig)
        aro_params, adamw_opt = self._get_full_params_and_adamw()
        u_processor = self.get(_PrivateServices.u_processor(self._optim_cfg.u_processor.name))
        kwargs = asdict(self._optim_cfg, dict_factory=_exclude_name)
        sinkhorn_opt = AroSinkhorn(aro_params, u_processor=u_processor, **kwargs)
        return [sinkhorn_opt, adamw_opt]

    @apps.service(_PrivateServices.optimizers("aroadam"))
    def _aroadam_optimizer(self) -> list[torch.optim.Optimizer]:
        assert isinstance(self._optim_cfg, AroAdamConfig)
        aro_params, adamw_opt = self._get_full_params_and_adamw()
        u_processor = self.get(_PrivateServices.u_processor(self._optim_cfg.u_processor.name))
        kwargs = asdict(self._optim_cfg, dict_factory=_exclude_name)
        aroadam_opt = AroAdam(aro_params, u_processor=u_processor, **kwargs)
        return [aroadam_opt, adamw_opt]

    @apps.service(_PrivateServices.optimizers("muon"))
    def _muon_optimizer(self) -> list[torch.optim.Optimizer]:
        assert isinstance(self._optim_cfg, MuonConfig)
        muon_params, adamw_opt = self._get_full_params_and_adamw()
        kwargs = asdict(self._optim_cfg, dict_factory=_exclude_name)
        muon_opt = Muon(muon_params, **kwargs)
        return [muon_opt, adamw_opt]


class _MixedOptimizerContainer(apps.Container):
    _config: BaseConfig
    _optim_cfg: OPT_CONFIGS
    _model_container: ModelContainer

    def __init__(self, config: BaseConfig, model_container: ModelContainer):
        self._config = config
        self._optim_cfg = config.optimizer
        self._model_container = model_container

    def _get_mixed_params_and_adamw(self) -> tuple[list[nn.Parameter], AdamW]:
        model = self._model_container.get(ModelServices.DISTRIBUTED_MODEL)
        model = cast(nn.Module, model.module if isinstance(model, (DDP)) else model)
        head_params = self.get(_PrivateServices.head_params(self._config.model.name))
        oned_params = [p for p in model.parameters() if p.requires_grad and p.ndim == 1]
        adamw_params = head_params + oned_params
        params = [
            p
            for p in model.parameters()
            if p.requires_grad and all(p is not q for q in adamw_params)
        ]
        adamw_opt = AdamW(
            adamw_params,
            lr=self._optim_cfg.lr,
            betas=(0.9, 0.95),
            weight_decay=self._optim_cfg.weight_decay,
            eps=self._optim_cfg.eps,
        )
        return params, adamw_opt

    @apps.container()
    def _common_container(self) -> "_CommonContainer":
        return _CommonContainer(self._config, self._model_container)

    @apps.service(_PrivateServices.optimizers("mixed-arosinkhorn"))
    def _mixed_arosinkhorn_optimizer(self) -> list[torch.optim.Optimizer]:
        assert isinstance(self._optim_cfg, AroSinkhornConfig)
        aro_params, adamw_opt = self._get_mixed_params_and_adamw()
        u_processor = self.get(_PrivateServices.u_processor(self._optim_cfg.u_processor.name))
        kwargs = asdict(self._optim_cfg, dict_factory=_exclude_name)
        sinkhorn_opt = AroSinkhorn(aro_params, u_processor=u_processor, **kwargs)
        return [adamw_opt, sinkhorn_opt]

    @apps.service(_PrivateServices.optimizers("mixed-aroadam"))
    def _mixed_aroadam_optimizer(self) -> list[torch.optim.Optimizer]:
        assert isinstance(self._optim_cfg, AroAdamConfig)
        aro_params, adamw_opt = self._get_mixed_params_and_adamw()
        u_processor = self.get(_PrivateServices.u_processor(self._optim_cfg.u_processor.name))
        kwargs = asdict(self._optim_cfg, dict_factory=_exclude_name)
        aroadam_opt = AroAdam(aro_params, u_processor=u_processor, **kwargs)
        return [adamw_opt, aroadam_opt]

    @apps.service(_PrivateServices.optimizers("mixed-arosign"))
    def _mixed_arosign_optimizer(self) -> list[torch.optim.Optimizer]:
        assert isinstance(self._optim_cfg, AroSignConfig)
        aro_params, adamw_opt = self._get_mixed_params_and_adamw()
        u_processor = self.get(_PrivateServices.u_processor(self._optim_cfg.u_processor.name))
        kwargs = asdict(self._optim_cfg, dict_factory=_exclude_name)
        arosign_opt = AroSign(aro_params, u_processor=u_processor, **kwargs)
        return [adamw_opt, arosign_opt]

    @apps.service(_PrivateServices.optimizers("mixed-muon"))
    def _mixed_muon_optimizer(self) -> list[torch.optim.Optimizer]:
        assert isinstance(self._optim_cfg, MuonConfig)
        muon_params, adamw_opt = self._get_mixed_params_and_adamw()
        kwargs = asdict(self._optim_cfg, dict_factory=_exclude_name)
        muon_opt = Muon(muon_params, **kwargs)
        return [muon_opt, adamw_opt]


class _CommonContainer(apps.Container):
    _config: BaseConfig
    _optim_cfg: OPT_CONFIGS
    _model_container: ModelContainer

    def __init__(self, config: BaseConfig, model_container: ModelContainer):
        self._config = config
        self._optim_cfg = config.optimizer
        self._model_container = model_container

    @apps.service(_PrivateServices.u_processor("scqr"))
    def _scqr_update(self) -> Any:
        assert isinstance(self._optim_cfg, ARO_CONFIGS)
        assert isinstance(self._optim_cfg.u_processor, SCQRConfig)
        return SCQRUpdate(self._optim_cfg.u_processor.rotation_jitter)

    @apps.service(_PrivateServices.u_processor("polar"))
    def _polar_update(self) -> Any:
        assert isinstance(self._optim_cfg, ARO_CONFIGS)
        assert isinstance(self._optim_cfg.u_processor, PolarConfig)
        return PolarUpdate(self._optim_cfg.u_processor.steps)

    @apps.service(_PrivateServices.head_params("nanogpt"))
    @apps.service(_PrivateServices.head_params("nanochat"))
    def _gpt2_head_and_wte_params(self) -> list[nn.Parameter]:
        model = self._model_container.get(ModelServices.DISTRIBUTED_MODEL)
        model = model.module if isinstance(model, DDP) else model
        model = cast(NanoGPT | NanoChatGPT, model)  # OptimizedModel type
        params = list(model.lm_head.parameters())
        if model.transformer.wte.weight is not model.lm_head.weight:  # type: ignore
            params.append(model.transformer.wte.weight)  # type: ignore
        return params


class SchedulersContainer(apps.Container):
    _scheduler_cfg: SchedulerConfig | None
    _opt_container: OptimContainer

    def __init__(self, config: BaseConfig, opt_container: OptimContainer):
        self._scheduler_cfg = config.lr_scheduler
        self._opt_container = opt_container

    @apps.service(OptimServices.SCHEDULERS)
    def _schedulers(self) -> list[LRScheduler]:
        if self._scheduler_cfg is None:
            return []
        return self.get(_PrivateServices.schedulers(self._scheduler_cfg.name))

    @apps.service(_PrivateServices.schedulers("linear-cosine"))
    def _linear_cosine_scheduler(self) -> list[LRScheduler]:
        assert isinstance(self._scheduler_cfg, LinearCosineSchConfig)
        optimizers = self._opt_container.get(OptimServices.OPTIMIZERS)
        warmup_iters = self._scheduler_cfg.warmup_iters
        total_iters = self._scheduler_cfg.lr_decay_iters
        lr_min = self._scheduler_cfg.min_lr
        schedulers = []
        for opt in optimizers:
            warmup = WarmupLR(opt, warmup_iters)
            T_max = max(0, total_iters - warmup_iters)
            cosine = CosineAnnealingLR(opt, T_max=T_max, eta_min=lr_min)
            sch = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_iters])
            schedulers.append(sch)
        return schedulers
