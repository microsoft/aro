import logging
import os
from typing import Any, Literal, cast

import torch
import torch.nn as nn
from rats import apps
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.nn.parallel import DistributedDataParallel as DDP

from gpt import NanoChatGPT, NanoGPT
from gpt.config import GPT_CONF, NanoChatConfig, NanoGptConfig
from trainer.config import BaseConfig, TrainingConfig
from trainer.utils.device import get_device

from ._backend import sdpa_probe

MODEL_TYPE = NanoChatGPT | NanoGPT
MODEL_CONF = GPT_CONF
MODEL_NAME = Literal["nanochat", "nanogpt"]
STRATEGY = Literal["none", "ddp", "fsdp", "fsdp2"]


class FSDP2(nn.Module, FSDPModule): ...


logger = logging.getLogger(__name__)


def log_on_main_worker(message: str, *args: Any, level: int = logging.INFO):
    """Logs a message only on the worker with number (rank) 0."""
    if int(os.getenv("RANK", "0")) == 0:
        logger.log(level, message, *args)


class _PrivateServices:
    @staticmethod
    def model(name: MODEL_NAME) -> apps.ServiceId[MODEL_TYPE]:
        return apps.ServiceId[MODEL_TYPE](f"{name}-net")

    @staticmethod
    def model_shard(name: MODEL_NAME) -> apps.ServiceId[FSDP2]:
        return apps.ServiceId[FSDP2](f"{name}-fsdp-net")

    @staticmethod
    def strategy(name: STRATEGY) -> apps.ServiceId[MODEL_TYPE | DDP | FSDP2]:
        return apps.ServiceId[MODEL_TYPE | DDP | FSDP2](f"{name}-wrapped-model")

    @staticmethod
    def flops_per_token(name: MODEL_NAME) -> apps.ServiceId[float]:
        return apps.ServiceId[float](f"{name}-flops-per-token")

    COMPILED_MODEL = apps.ServiceId[nn.Module]("compiled-model")


@apps.autoscope
class ModelServices:
    RAW_MODEL = apps.ServiceId[MODEL_TYPE]("raw-model")
    DISTRIBUTED_MODEL = apps.ServiceId[MODEL_TYPE | DDP | FSDP2]("distributed-model")
    NUM_PARAMETERS = apps.ServiceId[int]("num-parameters")
    FLOPS_PER_TOKEN = apps.ServiceId[float]("flops-per-token")
    ATTENTION_BACKEND = apps.ServiceId[None]("attention-backend")


class ModelContainer(apps.Container):
    _base_config: BaseConfig
    _model_conf: MODEL_CONF
    _training_conf: TrainingConfig

    def __init__(self, config: BaseConfig) -> None:
        self._base_config = config
        self._model_conf = config.model
        self._training_conf = config.training

    @apps.service(ModelServices.RAW_MODEL)
    def _raw_model(self) -> MODEL_TYPE:
        model = self.get(_PrivateServices.model(self._model_conf.name))
        return model

    @apps.service(ModelServices.DISTRIBUTED_MODEL)
    def _distributed_model(self) -> MODEL_TYPE | DDP | FSDP2:
        return self.get(_PrivateServices.strategy(self._training_conf.strategy))

    @apps.service(_PrivateServices.model("nanogpt"))
    def _nanogpt_model(self) -> NanoGPT:
        assert isinstance(self._model_conf, NanoGptConfig)
        model = NanoGPT(self._model_conf)
        model.to(get_device())
        return model

    @apps.service(_PrivateServices.model("nanochat"))
    def _nanochat_model(self) -> NanoChatGPT:
        assert isinstance(self._model_conf, NanoChatConfig)
        model = NanoChatGPT(self._model_conf)
        model.to(get_device())
        return model

    @apps.service(_PrivateServices.model_shard("nanogpt"))
    @apps.service(_PrivateServices.model_shard("nanochat"))
    def _fsdp2_nano_models(self) -> FSDP2:
        model = self.get(ModelServices.RAW_MODEL)
        mp_policy = self._mixed_precision_policy()
        for layer in model.transformer.h:  # type: ignore
            fully_shard(layer, mp_policy=mp_policy)
        if hasattr(model.transformer, "wpe"):
            fully_shard(model.transformer.wpe, mp_policy=mp_policy)  # type: ignore
        fully_shard(model.lm_head, mp_policy=mp_policy)
        fully_shard(model, mp_policy=mp_policy)
        return cast(FSDP2, model)

    @apps.service(_PrivateServices.strategy("none"))
    def _non_distributed_model(self) -> MODEL_TYPE:
        model = self.get(ModelServices.RAW_MODEL)
        if self._training_conf.compile_model:
            log_on_main_worker("Compiling model... (takes ~1 minute)")
            model = cast(MODEL_TYPE, torch.compile(model, dynamic=False))
        return model

    @apps.service(_PrivateServices.strategy("ddp"))
    def _ddp_wrapped_model(self) -> DDP:
        model = self.get(ModelServices.RAW_MODEL)
        if self._training_conf.compile_model:
            log_on_main_worker("Compiling DDP model... (takes ~1 minute)")
            model = torch.compile(model, dynamic=False)
        return DDP(model)

    @apps.service(_PrivateServices.strategy("fsdp"))
    @apps.service(_PrivateServices.strategy("fsdp2"))
    def _fsdp2_wrapped_model(self) -> FSDP2:
        model = self.get(_PrivateServices.model_shard(self._model_conf.name))
        if self._training_conf.compile_model:
            log_on_main_worker("Compiling FSDP2 model... (takes ~1 minute)")
            model = torch.compile(model, dynamic=False)
        return cast(FSDP2, model)

    def _mixed_precision_policy(self) -> MixedPrecisionPolicy:
        precision = self._training_conf.precision
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[precision]
        return MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)

    @apps.container()
    def _model_info_container(self) -> "InfoModelContainer":
        return InfoModelContainer(self._model_conf, self._training_conf, self)


class InfoModelContainer(apps.Container):
    _model_conf: MODEL_CONF
    _model_container: ModelContainer
    _training_conf: TrainingConfig

    def __init__(
        self,
        model_conf: MODEL_CONF,
        training_conf: TrainingConfig,
        model_container: ModelContainer,
    ) -> None:
        self._model_conf = model_conf
        self._training_conf = training_conf
        self._model_container = model_container

    @apps.service(ModelServices.NUM_PARAMETERS)
    def _num_parameters(self) -> int:
        model = self._model_container.get(ModelServices.RAW_MODEL)
        if isinstance(model, NanoGPT):
            n_params = model.get_num_params()
        else:
            n_params = sum(p.numel() for p in model.parameters())
        return n_params

    @apps.service(ModelServices.FLOPS_PER_TOKEN)
    def _flops_per_token(self) -> float:
        return self._model_container.get(_PrivateServices.flops_per_token(self._model_conf.name))

    @apps.service(_PrivateServices.flops_per_token("nanogpt"))
    def _nanogpt_flops_per_token(self) -> int:
        cfg = self._model_conf
        assert isinstance(cfg, NanoGptConfig)
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.sequence_len
        N = self.get(ModelServices.NUM_PARAMETERS)
        num_flops_per_token = 6 * N + 12 * L * H * Q * T
        return num_flops_per_token

    @apps.service(_PrivateServices.flops_per_token("nanochat"))
    def _nanochat_flops_per_token(self) -> int:
        model = self._model_container.get(ModelServices.RAW_MODEL)
        assert isinstance(model, NanoChatGPT)
        return model.estimate_flops()

    @apps.service(ModelServices.ATTENTION_BACKEND)
    def _attention_backend(self) -> None:
        head_dim = self._model_conf.n_embd // self._model_conf.n_head
        precision = self._training_conf.precision
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[precision]
        sdpa_probe(
            tag="nanogpt",
            device=get_device(),
            dtype=dtype,
            batch_size=self._training_conf.batch_size,
            seq_len=self._model_conf.sequence_len,
            num_heads=self._model_conf.n_head,
            num_kv_heads=self._model_conf.n_head,
            head_dim=head_dim,
        )
