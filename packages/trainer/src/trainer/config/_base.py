from pathlib import Path
from typing import Annotated, Literal

from pydantic import Discriminator
from pydantic.dataclasses import dataclass

from gpt.config import GPT_CONF

from .datasets import OWTConf
from .optim import (
    AdamWConfig,
    AroAdamConfig,
    AroSignConfig,
    AroSinkhornConfig,
    MuonConfig,
)
from .schedulers import LinearCosineSchConfig

ACCELERATOR = Literal["cpu", "cuda"]
PRECISION = Literal["bfloat16", "float16"]
STRATEGY = Literal["none", "ddp"]
MODEL_CONF = Annotated[GPT_CONF, Discriminator("name")]
ARO_CONF = AroAdamConfig | AroSinkhornConfig | AroSignConfig
OPTIM_CONF = Annotated[ARO_CONF | AdamWConfig | MuonConfig, Discriminator("name")]
DATASET_CONF = OWTConf
SCHEDULER_CONF = LinearCosineSchConfig | None


@dataclass(frozen=True)
class ComputeEnvironmentConfig:
    nodes: int = 1
    accelerator: ACCELERATOR = "cuda"
    output_dir: Path = Path("outputs")


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    run_id: str | None = None


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    max_iters: int
    gradient_accumulation_steps: int
    grad_clip: float
    precision: PRECISION
    strategy: STRATEGY
    compile_model: bool = True
    log_interval: int = 10
    eval_interval: int = 1000
    checkpoint_interval: int = 1000  # 0 for skipping saving
    checkpoint_path: Path = Path("outputs/checkpoints")


@dataclass(frozen=True)
class TrainingHF:
    batch_size: int
    max_iters: int
    gradient_accumulation_steps: int
    grad_clip: float
    precision: PRECISION
    log_interval: int = 1
    eval_interval: int = 0  # 0 for skipping evaluation
    compile_model: bool = True
    eval_batch_size: int = 10
    eval_tokens: int = 14_000_000  # 14M
    save_interval: int = 0  # 0 for skipping saving
    resume_path: str | None = None  # Path to the checkpoint to resume from


@dataclass(frozen=True)
class WandBConfig:
    name: Literal["wandb"] = "wandb"
    entity: str | None = None
    host: str | None = None
    mode: Literal["online", "offline", "disabled"] | None = None
    log: Literal["parameters", "gradients", "all"] = "all"
    run_name: str | None = None


@dataclass(frozen=True)
class BaseConfig:
    project: ProjectConfig
    model: MODEL_CONF
    optimizer: OPTIM_CONF
    lr_scheduler: SCHEDULER_CONF
    training: TrainingConfig
    data: DATASET_CONF
    compute_env: ComputeEnvironmentConfig
    logger: WandBConfig | None = None
    seed: int = 1337
