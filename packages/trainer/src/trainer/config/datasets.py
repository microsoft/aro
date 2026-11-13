from dataclasses import field
from pathlib import Path
from typing import Literal

from pydantic.dataclasses import dataclass

DATASET_NAMES = Literal["owt"]


@dataclass(frozen=True)
class DatasetConfig:
    name: DATASET_NAMES
    path: Path
    iterable: bool = field(init=False)
    enable_batch_dl: bool = field(init=False)  # enables dataloader batching or relies on Dataset
    num_workers: int = 0
    prefetch_factor: int | None = None
    pin_memory: bool = True


@dataclass(frozen=True)
class OWTConf(DatasetConfig):
    name: Literal["owt"]
    path: Path = Path("data/owt")
    eval_size: int = 200

    def __post_init__(self) -> None:
        object.__setattr__(self, "iterable", True)
        object.__setattr__(self, "enable_batch_dl", False)  # owt uses its own batching
