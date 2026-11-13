from typing import Annotated, Literal

from pydantic import Discriminator
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class SCQRConfig:
    name: Literal["scqr"] = "scqr"
    rotation_jitter: float = 1e-7


@dataclass(frozen=True)
class CholQrConfig:
    name: Literal["chol-qr"] = "chol-qr"
    rotation_jitter: float = 1e-7


@dataclass(frozen=True)
class PolarConfig:
    name: Literal["polar"] = "polar"
    steps: int = 5


U_PROCESSOR = Annotated[SCQRConfig | CholQrConfig | PolarConfig, Discriminator("name")]
