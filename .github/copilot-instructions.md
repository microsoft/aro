# Copilot Instructions for ARO

## Build & Run

```bash
# Install dependencies (requires uv). The trainer/gpt packages depend on flash-attn; its build
# downloads a matching prebuilt wheel when one exists and otherwise compiles from source (needs
# the CUDA toolkit/nvcc and can take a long time). No prebuilt wheel exists for the locked
# torch/CUDA versions, so this compiles flash-attn.
uv sync
# flash-attn is not imported at runtime (the models load FA3 via `kernels` or fall back to SDPA),
# so it can be skipped; then use `uv run --no-sync` so uv does not try to install it again.
uv sync --no-install-package flash-attn

# Lint
uv run --no-sync ruff check --fix
uv run --no-sync ruff format

# Type check
uv run --no-sync pyright

# Tests
uv run --no-sync pytest

# Run pretrain training example (single GPU, W&B disabled); run from the repository root
uv run --no-sync torchrun --standalone --nproc_per_node=1 -m trainer.pretrain --cfg nanogpt -- \
  training.strategy=none logger=null

# Training config overrides use Hydra syntax; select optimizer recipes with `optimizer=<name>`
uv run --no-sync torchrun --standalone --nproc_per_node=1 -m trainer.pretrain --cfg nanogpt -- \
  training.strategy=none logger=null optimizer=muon optimizer.lr=0.001
```

## Architecture

ARO is a PyTorch optimizer library for LLM training. It is structured as a **uv workspace monorepo**:

- **`src/aro/`** — The core library (published as the `aro` package). Contains optimizer implementations in `optim/` and `processors/`, and RoM low-rank adaptation in `lora/`. Only depends on `torch`.
- **`packages/gpt/`** — Model definitions (NanoGPT, NanoChat) used for training experiments and dataset preparation utilities (e.g., `python -m dataprep.openwebtext`).
- **`packages/trainer/`** — Pretrain training loop (`trainer.pretrain`). Uses Hydra for config loading with pydantic validation. Depends on `aro` and `gpt` as workspace references. Supports non-distributed (`none`), DDP and FSDP2 strategies specified in config.
- **`config/`** — Hydra YAML config files for training runs (`nanogpt`, `nanochat`, `sinkgd`), plus the `optimizer/` config group with one recipe per optimizer name.
- **`test/`** — Tests for the core library (`test_aro/`, `test_lora/`) and the workspace packages (`test_gpt/`, `test_trainer/`).

### Optimizer pattern

Each optimizer in `src/aro/optim/` follows the same structure:

1. A `GradientProcessor` (protocol in `defs.py`) transforms gradients (e.g., Sinkhorn normalization, Adam moments).
2. A `UProcessor` (protocol in `defs.py`) updates rotation matrices applied to gradients.
3. The optimizer class (e.g., `AroSinkhorn`) extends `AroBase` in `_rot.py`, which handles the specific initialization of the corresponding `GradientProcessor`.

## Dependency injection

The project uses **`rats-apps`** (`rats.apps`) as its DI framework. All builder/factory functions **must** be methods on a container class — never standalone functions. Services are lazy-instantiated and cached (singleton per container).

### Core concepts

| Concept | Usage |
|---|---|
| `apps.Container` | Base class for all DI containers |
| `apps.ServiceId[T]("name")` | Typed token that identifies a service |
| `@apps.service(ServiceId)` | Marks a method as the factory for a service |
| `@apps.autoscope` | Namespaces a `ServiceId` class to avoid collisions |
| `@apps.autoid_service` | Auto-generates a `ServiceId` from the method name |
| `@apps.group(ServiceId)` | Contributes to a multi-value service group |
| `@apps.container()` | Embeds a sub-container whose services become available in the parent |
| `self.get(ServiceId)` | Resolves a service (lazy, cached) |

### File conventions

Each sub-package in `trainer/` follows a consistent layout:

- **`_services.py`** — Contains the container class(es) and `ServiceId` definitions.
- **`__init__.py`** — Re-exports only the public container and services class (e.g., `ModelContainer`, `ModelServices`).
- **`XxxServices`** (decorated with `@apps.autoscope`) — Public `ServiceId` registry for a sub-package.
- **`_PrivateServices`** (no autoscope) — Module-internal `ServiceId`s, often with `@staticmethod` factories for parameterized IDs.
- **`XxxContainer(apps.Container)`** — The public container class.

### How to define services (pattern to follow)

**1. Declare service IDs in a scoped class:**

```python
@apps.autoscope
class ModelServices:
    RAW_MODEL = apps.ServiceId[MODEL_TYPE]("raw-model")
    DISTRIBUTED_MODEL = apps.ServiceId[MODEL_TYPE | DDP]("distributed-model")
    NUM_PARAMETERS = apps.ServiceId[int]("num-parameters")
```

**2. Use `_PrivateServices` with static methods for dynamic/parameterized IDs:**

```python
class _PrivateServices:
    @staticmethod
    def model(name: MODEL_NAME) -> apps.ServiceId[MODEL_TYPE]:
        return apps.ServiceId[MODEL_TYPE](f"{name}-net")

    @staticmethod
    def strategy(name: STRATEGY) -> apps.ServiceId[nn.Module | DDP]:
        return apps.ServiceId[nn.Module | DDP](f"{name}-wrapped-model")
```

**3. Group all builder methods inside a container class:**

```python
class ModelContainer(apps.Container):
    def __init__(self, config: BaseConfig) -> None:
        self._base_config = config

    @apps.service(ModelServices.RAW_MODEL)
    def _raw_model(self) -> MODEL_TYPE:
        return self.get(_PrivateServices.model(self._base_config.model.name))

    @apps.service(_PrivateServices.model("nanogpt"))
    def _nanogpt_model(self) -> NanoGPT:
        return NanoGPT(self._base_config.model)

    @apps.container()
    def _model_info_container(self) -> "InfoModelContainer":
        return InfoModelContainer(self._base_config.model, self)
```

**4. Embed sub-containers with `@apps.container()` to compose services:**

```python
class OptimContainer(apps.Container):
    def __init__(self, config: BaseConfig, model_container: ModelContainer):
        self._config = config
        self._model_container = model_container

    @apps.container()
    def _fulloptimizer_container(self) -> "_FullOptimizerContainer":
        return _FullOptimizerContainer(self._config, self._model_container)

    @apps.container()
    def _schedulers_container(self) -> "SchedulersContainer":
        return SchedulersContainer(self._config, self)
```

**5. Wire containers together via constructor injection at the entry point:**

```python
# packages/trainer/src/trainer/pretrain/__main__.py
config_container = ConfigContainer("config", cfg, overrides)
config = config_container.get(ConfigServices.BASE_CONFIG)
model_container = ModelContainer(config)
optim_container = OptimContainer(config, model_container)
state_container = StateContainer(config, model_container, optim_container)
train_container = TrainIteratorContainer(config, state_container)
val_container = ValDataloaderContainer(config)
TrainContainer(
    config, model_container, state_container, train_container, val_container, wandb_run
).train()
```

### Key rules

- **All factory/builder functions must live inside an `apps.Container` subclass.** No standalone builder functions.
- **Services are injected via constructor arguments** (passing one container to another).
- **Use `self.get(ServiceId)` to resolve dependencies** within a container — never call other containers' private methods directly.
- **Mark service methods as private** (prefix with `_`) since they are called by the DI framework, not by external code.
- **Use `@apps.container()` for composition** — embed sub-containers to break large containers into focused units.

### Container overview (trainer package)

| Container | Responsibility |
|---|---|
| `ConfigContainer` | Hydra/OmegaConf config loading, wandb init |
| `ModelContainer` | Model creation (NanoGPT/NanoChatGPT), DDP/FSDP2 wrapping |
| `InfoModelContainer` | Parameter count, FLOPs estimation (embedded in `ModelContainer`) |
| `OptimContainer` | Optimizer selection (AdamW, Muon, SinkGD, ARO variants) |
| `SchedulersContainer` | LR schedulers (embedded in `OptimContainer`) |
| `StateContainer` | Training state and the per-run checkpoint directory (resumed by `TrainContainer`) |
| `TrainIteratorContainer` | Training dataset + infinite sampler |
| `ValDataloaderContainer` | Validation dataset + distributed sampler |
| `TrainContainer` | Training loop orchestration |

### Configuration pattern

Optimizer and model configs are **frozen pydantic dataclasses** with a `name` literal discriminator field. Unions use `Annotated[... , Discriminator("name")]` for polymorphic deserialization from Hydra YAML. Optimizer configs reject unknown fields (`extra="forbid"`), and each optimizer name has a recipe in `config/optimizer/<name>.yaml` that runs select with `optimizer=<name>`; keep YAML configs listing every field users may override, since Hydra rejects plain `key=value` overrides of keys missing from the YAML.

## Conventions

- **Type annotations**: All code must pass `pyright` in strict-ish mode (see `pyrightconfig.json`). Use modern Python typing (`X | Y`, not `Union[X, Y]`).
- **Formatting**: Ruff with Black-compatible style. Double quotes, 4-space indent, 99-char line length. Google-style docstrings.
- **Imports**: isort via ruff. First-party packages: `aro`, `gpt`, `trainer`, `dataprep`.
- **Config dataclasses**: Always `frozen=True`, use `pydantic.dataclasses.dataclass` (not stdlib or attrs).
