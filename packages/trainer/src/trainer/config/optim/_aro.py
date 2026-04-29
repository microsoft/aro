from typing import Literal

from pydantic import Field
from pydantic.dataclasses import dataclass

from ._base import AdamConfig, BaseOptimizerConfig, SinkhornConfig
from ._u import U_PROCESSOR, SCQRConfig


@dataclass(frozen=True)
class AroConfig:
    weight_decay: float = 0.0
    momentum: float = 0.95
    eps: float = 1e-8
    alpha: float = 0.2
    update_freq: int = 1
    constant_norm: bool = True
    u_processor: U_PROCESSOR = Field(default_factory=SCQRConfig)


@dataclass(frozen=True)
class AroSinkhornConfig(BaseOptimizerConfig, SinkhornConfig, AroConfig):
    name: Literal["mixed-arosinkhorn", "arosinkhorn"] = "mixed-arosinkhorn"


@dataclass(frozen=True)
class AroAdamConfig(BaseOptimizerConfig, AdamConfig, AroConfig):
    name: Literal["mixed-aroadam", "aroadam"] = "mixed-aroadam"


@dataclass(frozen=True)
class AroSignConfig(BaseOptimizerConfig, AroConfig):
    name: Literal["mixed-arosign", "arosign"] = "mixed-arosign"


@dataclass(frozen=True)
class MuonConfig(BaseOptimizerConfig, AroConfig):
    name: Literal["muon", "mixed-muon"] = "muon"
    ns_steps: int = 5
