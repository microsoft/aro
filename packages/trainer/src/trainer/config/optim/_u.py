from typing import Annotated, Literal

from pydantic import Discriminator
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class SCQRConfig:
    name: Literal["scqr"] = "scqr"
    rotation_jitter: float = 1e-9


@dataclass(frozen=True)
class PolarConfig:
    name: Literal["polar"] = "polar"
    steps: int = 5


U_PROCESSOR = Annotated[SCQRConfig | PolarConfig, Discriminator("name")]
