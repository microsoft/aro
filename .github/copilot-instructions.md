# Copilot Instructions for ARO

## Build & Run

```bash
# Install dependencies (requires uv)
uv sync

# Lint
uv run ruff check --fix
uv run ruff format

# Type check
uv run pyright

# Run pretrain training example (single GPU)
uv run torchrun --standalone --nproc_per_node=1 -m trainer.pretrain --cfg default

# Training config overrides use Hydra syntax
uv run torchrun --standalone --nproc_per_node=1 -m trainer.pretrain --cfg default -- optimizer.lr=0.001
```

## Architecture

ARO is a PyTorch optimizer library for LLM training. It is structured as a **uv workspace monorepo**:

- **`src/aro/`** — The core library (published as the `aro` package). Contains optimizer implementations in `optim/` and `processors/`. Only depends on `torch`.
- **`packages/gpt/`** — Model definitions (NanoGPT, NanoChat) used for training experiments and dataset preparation utilities (e.g., `python -m dataprep.openwebtext`).
- **`packages/trainer/`** — Pretrain training loop (`trainer.pretrain`). Uses Hydra for config loading with pydantic validation. Depends on `aro` and `gpt` as workspace references. Supports both non-distributed and DDP strategies specified in config.
- **`config/`** — Hydra YAML config files for training runs.
- **`test/`** — Tests for the core `aro` library (not for workspace packages).

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
| `ModelContainer` | Model creation (NanoGPT/NanoChatGPT), DDP wrapping |
| `InfoModelContainer` | Parameter count, FLOPs estimation (embedded in `ModelContainer`) |
| `OptimContainer` | Optimizer selection (AdamW, Muon, ARO variants) |
| `SchedulersContainer` | LR schedulers (embedded in `OptimContainer`) |
| `StateContainer` | Training state management, checkpoint resume |
| `TrainIteratorContainer` | Training dataset + infinite sampler |
| `ValDataloaderContainer` | Validation dataset + distributed sampler |
| `TrainContainer` | Training loop orchestration |

### Configuration pattern

Optimizer and model configs are **frozen pydantic dataclasses** with a `name` literal discriminator field. Unions use `Annotated[... , Discriminator("name")]` for polymorphic deserialization from Hydra YAML.

## Conventions

- **Type annotations**: All code must pass `pyright` in strict-ish mode (see `pyrightconfig.json`). Use modern Python typing (`X | Y`, not `Union[X, Y]`).
- **Formatting**: Ruff with Black-compatible style. Double quotes, 4-space indent, 99-char line length. Google-style docstrings.
- **Imports**: isort via ruff. First-party packages: `aro`, `gpt`, `trainer`, `dataprep`.
- **Config dataclasses**: Always `frozen=True`, use `pydantic.dataclasses.dataclass` (not stdlib or attrs).
