# :sparkles: ARO

[![main](https://github.com/microsoft/aro/actions/workflows/main.yml/badge.svg)](https://github.com/microsoft/aro/actions/workflows/main.yml)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Adaptive Rotation Optimizers

ARO is a library of PyTorch optimizers for LLM training. For more details, see our report:

**[ARO: A New Lens On Matrix Optimization For Large Models](https://arxiv.org/abs/2602.09006)**

For low-rank adaptation (LoRA), the library also includes
[RoM (Rotated Manifold Optimization)](#f-rom-rotated-manifold-optimization-for-low-rank-adaptation),
introduced in **[Rotated Manifold Optimization for Low-Rank Adaptation](https://arxiv.org/abs/2609.34264)**.

It also includes [SinkGD (Sinkhorn Gradient Descent)](#e-sinkgd), a rotation-free matrix
optimizer based on repeated row/column normalization, introduced in
**[Gradient Multi-Normalization for Stateless and Scalable LLM Training](https://arxiv.org/abs/2502.06742)**
(implemented here as a momentum-based variant).

It includes a range of advanced optimizers designed to improve training efficiency and model performance.
These optimizers process and rotate gradients prior to updating model parameters, leveraging
structures from the model architecture.

ARO includes the following optimizers (all in `aro.optim`):

* AroAdam: An adaptive learning rate optimizer that tracks first, second moments of gradients and incorporates gradient rotation.
* AroSinkhorn: A variant that applies Sinkhorn-style row/column L2 normalization to gradients in a rotated basis and then rotates the update back.
* AroSign: An optimizer that applies rotation to the sign of the gradients, inspired by signSGD.
* Muon: Momentum orthogonalized via Newton–Schulz iterations (Moonlight-style, with update RMS aligned to AdamW), implemented as the rotation-free special case of the ARO framework.
* [SinkGD](#e-sinkgd): Rotation-free Nesterov momentum with alternating or simultaneous row/column normalization and shard-local FSDP2 updates.

Our default in the paper uses AroSinkhorn, which provides robust overall performance. In certain cases, AroAdam can outperform AroSinkhorn.

-----------------------------------------------------------------

## :rocket: Quick start

This section gets you running quickly; the [Appendix](#appendix) covers requirements,
configuration and advanced usage in detail.

### 1. Use ARO in your own training loop

The `aro` library needs Python ≥ 3.12 and PyTorch (its only dependency) and runs on CPU or CUDA.
Add it to your project, e.g. with `uv` or `pip`:

```bash
uv add "aro @ git+https://github.com/microsoft/aro"
# or: pip install "aro @ git+https://github.com/microsoft/aro"
```

ARO optimizes 2D weight matrices; keep the other parameters on AdamW. The split and
hyperparameters below mirror our NanoGPT example (the `mixed-arosinkhorn` recipe):

```python
import torch
from torch.optim import AdamW

from aro.optim import AroSinkhorn

model = ...  # your model, moved to CUDA before constructing the optimizers
lr = 6e-4
weight_decay = 0.1
grad_clip = 1.0
max_iters = 100000
loss_fn = torch.nn.CrossEntropyLoss()
train_iterator = ...  # your data iterator

# AdamW: output head, token embedding and every non-2D parameter (biases, norms).
# ARO: all remaining 2D weight matrices. Adapt the module names to your model.
trainable = [p for p in model.parameters() if p.requires_grad]
adamw_modules = [model.lm_head, model.tok_embeddings]
adamw_ids = {id(p) for module in adamw_modules for p in module.parameters()}
aro_params = [p for p in trainable if p.ndim == 2 and id(p) not in adamw_ids]
aro_ids = {id(p) for p in aro_params}
adamw_params = [p for p in trainable if id(p) not in aro_ids]

optimizers = [
    AdamW(adamw_params, lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay),
    AroSinkhorn(aro_params, lr=lr, weight_decay=weight_decay),
]
```

The training loop is the same as with any PyTorch optimizer (shown with gradient clipping and
bf16 mixed precision):

```python
step = 0
while True:
    inputs, labels = next(train_iterator)
    inputs, labels = inputs.to("cuda"), labels.to("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(inputs)
        loss = loss_fn(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))
    loss.backward()

    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    for opt in optimizers:
        opt.step()
        opt.zero_grad()

    step += 1
    if step >= max_iters:
        break
```

The examples also use a learning-rate schedule (200 warmup iterations, then cosine decay to
`lr / 10`). See [Appendix B](#b-using-the-optimizers-in-your-own-code) for the other optimizers,
multi-GPU training and checkpointing.

### 2. Run the bundled NanoGPT example

The example pretrains GPT-2 (124M) on OpenWebText with the default `mixed-arosinkhorn` recipe.
You need Linux, NVIDIA GPU(s) with a driver that supports CUDA 13.0,
[uv](https://docs.astral.sh/uv/getting-started/installation/) and roughly 100 GB of free disk for
the dataset (see [Appendix A](#a-requirements-and-installation)). Run all commands from the
repository root.

```bash
# 1. Install. This skips the optional flash-attn package; a plain `uv sync` also compiles it from
#    source, which needs the CUDA 13 toolkit (nvcc), see Appendix A.
git clone https://github.com/microsoft/aro.git
cd aro
uv sync --no-install-package flash-attn

# 2. Download (~24 GB) and tokenize OpenWebText into data/openwebtext/ (see Appendix C)
uv run --no-sync python -m dataprep.openwebtext

# 3. Log in to Weights & Biases (or add `-- logger=null` to the training command to disable it)
uv run --no-sync wandb login

# 4. Train on 4 GPUs (~1-1.5 hours on 4x A100 80GB). Runs log to your default W&B entity;
#    set `logger.entity=<your W&B user or team>` to log elsewhere.
uv run --no-sync torchrun --standalone --nproc_per_node=4 -m trainer.pretrain --cfg nanogpt
```

Common variations (append overrides after `--`, e.g. `... --cfg nanogpt -- optimizer=muon`):

* **Another optimizer:** `optimizer=<name>` with `adamw`, `arosinkhorn`, `aroadam`, `arosign`,
  `muon`, `sinkgd`, or their `mixed-` variants (see [Appendix D](#optimizer-recipes)).
* **Single GPU:** use `--nproc_per_node=1` and append `training.strategy=none`. Other GPU counts
  must divide `training.gradient_accumulation_steps` (80) evenly
  (see [Appendix D](#gpus-batch-size-and-gradient-accumulation)).
* **Other examples:** `--cfg nanochat` (NanoChat-style model) or `--cfg sinkgd` (small SinkGD run).
* **Resuming:** relaunch the same command; training continues from the latest checkpoint
  (see [Appendix D](#outputs-checkpoints-and-resuming)).

-----------------------------------------------------------------

# :mountain: Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on contributing to this project.

# :triangular_ruler: Conventions

Consider reading [.github/copilot-instructions.md](.github/copilot-instructions.md) for a summary of the repository structure and coding style.

# :balance_scale: Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.


# :bookmark_tabs: Citation

If you use ARO in your research, please cite:

```bibtex
@article{gong2026aro,
  title={ARO: A New Lens On Matrix Optimization For Large Models},
  author={Gong, Wenbo and Zazo, Javier and Luo, Qijun and Wang, Puqian and Hensman, James and Ma, Chao},
  journal={arXiv preprint arXiv:2602.09006},
  year={2026}
}
```

If you use RoM for low-rank adaptation, please cite:

```bibtex
@article{ding2026rom,
  title={Rotated Manifold Optimization for Low-Rank Adaptation},
  author={Ding, Yuhui and Zazo, Javier and Hensman, James},
  journal={arXiv preprint arXiv:2609.34264},
  year={2026},
  doi={10.48550/arXiv.2609.34264},
  url={https://arxiv.org/abs/2609.34264}
}
```

For the SinkGD algorithm, please also cite:

```bibtex
@article{scetbon2025gradient,
  title={Gradient multi-normalization for stateless and scalable llm training},
  author={Scetbon, Meyer and Ma, Chao and Gong, Wenbo and Meeds, Edward},
  journal={arXiv preprint arXiv:2502.06742},
  year={2025}
}
```

-----------------------------------------------------------------

# Appendix

## A. Requirements and installation

**Library (`aro`).** Python ≥ 3.12 and PyTorch, its only dependency (tested with PyTorch 2.14).
The optimizers run on CPU or CUDA; multi-GPU training uses `torch.distributed` (DDP or FSDP2).

**Bundled examples (`gpt` and `trainer` packages).**

* **OS and GPUs:** Linux with one or more NVIDIA GPUs. The trainer is CUDA-only. The default
  NanoGPT configuration was tested on 4x A100 80GB, at about 1.5 s per iteration (2600 iterations
  take about 1-1.5 hours including evaluation). With less GPU memory, lower
  `training.batch_size` (see [Appendix D](#gpus-batch-size-and-gradient-accumulation)).
* **NVIDIA driver:** `uv sync` installs PyTorch wheels built for CUDA 13.0 (the `torchcuda` index
  in `pyproject.toml`), which need a driver that supports CUDA 13.0 (R580 or newer). For another
  CUDA version, point that index to another CUDA build of PyTorch and run `uv lock`.
* **uv and Python:** install [uv](https://docs.astral.sh/uv/getting-started/installation/); it
  downloads a suitable Python (≥ 3.12) automatically if needed.
* **Disk:** roughly 100 GB of free space for data preparation (see
  [Appendix C](#c-data-preparation)), plus about 3 GB per training checkpoint.
* **Network:** data preparation downloads from the Hugging Face Hub. On Hopper or newer GPUs,
  importing the `gpt` package also downloads Flash Attention 3 kernels from the Hub (used by
  NanoChat; other GPUs use PyTorch SDPA).

**flash-attn.** The `gpt` and `trainer` packages list `flash-attn` as a dependency, but no code
imports it at runtime (NanoGPT uses PyTorch SDPA; NanoChat loads Flash Attention 3 through
`kernels` or falls back to SDPA). `uv sync` installs it from a pinned source revision: its build
first tries to download a matching prebuilt wheel and compiles from source only when none exists.
No prebuilt wheel exists for the locked PyTorch 2.14 / CUDA 13 combination, so `uv sync` compiles
it, which needs the CUDA 13 toolkit (`nvcc`), a lot of RAM and can take a long time (set e.g.
`MAX_JOBS=4` to limit parallel compile jobs). To skip it:

```bash
uv sync --no-install-package flash-attn
```

Then always use `uv run --no-sync ...`, since a plain `uv run` would try to install flash-attn
again.

## B. Using the optimizers in your own code

**Which parameters.** AroSinkhorn, AroAdam, AroSign and Muon accept only 2D parameters; put
biases, normalization weights and other shapes on a separate optimizer such as AdamW. The
examples use two parameter splits:

* **Full** (`arosinkhorn`, `aroadam`, `arosign`, `muon` in the trainer): every trainable 2D
  matrix, including the token embedding and the output head, uses the matrix optimizer.
* **Mixed** (`mixed-arosinkhorn`, the default, and the other `mixed-` recipes): the token
  embedding and the output head use AdamW as well, as in the [Quick start](#1-use-aro-in-your-own-training-loop).

**Constructor defaults.** All arguments are keyword-only:

| Optimizer | Defaults |
|---|---|
| `AroSinkhorn` | `lr=1e-3, sinkhorn_steps=5, momentum=0.95, alpha=0.2, weight_decay=0.0, eps=1e-8, update_freq=1, constant_norm=True, u_processor=SCQRUpdate(rotation_jitter=1e-9)` |
| `AroAdam` | `lr=1e-3, betas=(0.0, 0.99), momentum=0.95, alpha=1.0, weight_decay=0.0, eps=1e-8, correct_bias=True, update_freq=1, constant_norm=True, u_processor=SCQRUpdate(rotation_jitter=1e-9)` |
| `AroSign` | `lr=1e-3, momentum=0.95, alpha=0.2, weight_decay=0.0, eps=1e-8, update_freq=1, constant_norm=True, u_processor=SCQRUpdate(rotation_jitter=1e-9)` |
| `Muon` | `lr=1e-3, ns_steps=5, momentum=0.95, alpha=0.2, weight_decay=0.0, eps=1e-8, constant_norm=True` |
| `SinkGD` | see [Appendix E](#e-sinkgd) |

* `momentum` is applied to the raw gradient before rotation. AroAdam's `betas` drive the Adam
  moments in the rotated basis.
* With `constant_norm=True`, each update is normalized to roughly unit RMS (aligning its scale
  with AdamW), and the parameter step is `lr * alpha * update`. Weight decay is decoupled
  (`p -= lr * weight_decay * p`).
* `update_freq` controls how often the rotation is refreshed. `u_processor` (from
  `aro.processors.u`) selects the rotation update: `SCQRUpdate(rotation_jitter)` (default),
  `PolarUpdate(steps)`, or `CholQrUpdate` (an alias of `SCQRUpdate`).
* The examples use `lr=6e-4` (NanoGPT) or `2e-3` (NanoChat), `weight_decay=0.1`, `alpha=0.2`,
  200 warmup iterations and cosine decay to `lr / 10`. AroAdam's constructor defaults differ
  from the examples: pass `alpha=0.2, betas=(0.0, 0.95)` to match them. The recipes are listed
  in [`config/optimizer/`](config/optimizer).

**Multi-GPU training.** On CUDA, ARO optimizers (including Muon) assign matrices to ranks
round-robin with both DDP and FSDP2: each rank computes rotation updates for its assigned
matrices and distributes the updates to the other ranks. Every rank must construct the optimizer
with the same parameters in the same order and call `step()`.

* **DDP:** initialize `torch.distributed` and wrap the model in `DistributedDataParallel`, so
  gradients are synchronized before `step()`.
* **FSDP2:** apply `fully_shard` before constructing the optimizer. FSDP2 updates support
  uneven and empty trailing row shards and one-dimensional submeshes.
* **CPU:** execution uses local updates.

Both execution paths leave the processed update in each active parameter's `.grad`. Read
gradients (e.g. for logging) before `step()`, and call `zero_grad()` before accumulating the next
step.

**Checkpointing.** Use the standard `state_dict()` / `load_state_dict()`, with one caveat for
DDP: each matrix's rotation state (and AroAdam's second moments) lives only on the rank that owns
the matrix. Save `optimizer.state_dict()` on every rank (e.g. one file per rank) and load each
rank's file with the same number of GPUs, as the bundled trainer does. With FSDP2, use
`torch.distributed.checkpoint` (`get_state_dict` / `set_state_dict`), and call
`optimizer.ensure_state_initialized()` before loading into a freshly constructed ARO optimizer.

## C. Data preparation

```bash
uv run --no-sync python -m dataprep.openwebtext
```

* **Run it from the repository root.** It writes `data/openwebtext/train.bin` and `val.bin`
  (GPT-2 BPE tokens as `uint16`), the default `data.path` of the trainer configs, which is
  resolved relative to the working directory.
* **Download:** the [Skylion007/openwebtext](https://huggingface.co/datasets/Skylion007/openwebtext)
  parquet files (~24 GB, about 8M documents) go to the Hugging Face cache. Set `HF_HOME` to put
  the cache on a large disk. The whole dataset is downloaded even when only a subset is
  tokenized.
* **Subset:** the `SUBSET_FRACTION` environment variable (default `0.29`) selects the fraction of
  documents to tokenize. 0.29 gives about 2.6B training tokens, enough for the default 2600
  iterations of ~1M tokens each; set `SUBSET_FRACTION=1.0` for the full ~9B tokens. 0.05% of the
  selected documents are held out for validation.
* **Disk and time:** at the default fraction, plan for about 90 GB of free disk: the ~24 GB
  download, a ~40 GB Hugging Face Arrow cache, a ~21 GB tokenized cache and the ~5 GB
  `train.bin`. With `SUBSET_FRACTION=1.0`, plan for about 150 GB (`train.bin` is then about
  17 GB). Tokenization uses 8 processes. Once the `.bin` files exist, the trainer no longer
  needs the Hugging Face cache, so you can delete it.

## D. Bundled trainer

`trainer.pretrain` is an example harness for the optimizers. It trains the NanoGPT (GPT-2 124M)
and NanoChat-style models from the `gpt` package on the tokenized OpenWebText files, on CUDA
GPUs only. To use other models or datasets, use the optimizers directly
([Appendix B](#b-using-the-optimizers-in-your-own-code)) or extend `packages/gpt` and
`packages/trainer`.

### Configs and overrides

Launch with `--cfg <name>` for one of the configs in [`config/`](config): `nanogpt` (default),
`nanochat` or `sinkgd`. Append [Hydra overrides](https://hydra.cc/docs/advanced/override_grammar/basic/)
after `--`, e.g. `-- optimizer.lr=0.001 training.max_iters=5000`. Each config lists every field
you can override, with comments, and pydantic validates the values at startup.

### Optimizer recipes

`optimizer=<name>` selects a complete recipe from [`config/optimizer/`](config/optimizer), and
`optimizer.<field>=<value>` tweaks it (e.g. `optimizer=aroadam optimizer.alpha=0.1`).

| Recipe | Matrix optimizer handles | AdamW handles |
|---|---|---|
| `adamw` | – | all trainable parameters |
| `arosinkhorn`, `aroadam`, `arosign`, `muon` | all trainable 2D matrices, including the token embedding and LM head | 1D parameters |
| `mixed-arosinkhorn` (default), `mixed-aroadam`, `mixed-arosign`, `mixed-muon` | 2D matrices except the token embedding and LM head | the token embedding, LM head and 1D parameters |
| `sinkgd`, `mixed-sinkgd` | as above (see [Appendix E](#e-sinkgd)) | as above, with separate `adamw_*` settings |

* **AdamW fallback:** for the ARO and Muon recipes it uses betas `(0.9, 0.95)` and the recipe's
  `lr`, `weight_decay` and `eps`.
* **Learning rate:** recipes default to `lr=6e-4` (SinkGD: `1e-3`), and `nanochat.yaml`
  overrides it to `2e-3` for whichever recipe is selected.
* **Switching optimizers:** use `optimizer=<name>` rather than `optimizer.name=<name>`.
  `optimizer.name` can only switch between a recipe and its `mixed-` variant; other changes fail
  validation with a hint.

### GPUs, batch size and gradient accumulation

* **Tokens per iteration:** `batch_size * sequence_len * gradient_accumulation_steps`
  (12 x 1024 x 80 ≈ 1M by default), independent of the number of GPUs.
* **Splitting across GPUs:** `training.gradient_accumulation_steps` is global, so each GPU runs
  `gradient_accumulation_steps / num_gpus` micro-batches of `training.batch_size` sequences per
  iteration. The GPU count must divide it evenly; the default 80 works with 1, 2, 4, 5, 8, 10 or
  16 GPUs.
* **Single GPU:** use `--nproc_per_node=1` and `training.strategy=none`. All 80 micro-batches
  then run on one GPU, so each iteration takes about 4x longer than on 4 GPUs.
* **Out of memory:** lower `training.batch_size` and raise `training.gradient_accumulation_steps`
  by the same factor to keep the tokens per iteration unchanged.

### Strategies

`training.strategy` is `none` (single process), `ddp` (default for `nanogpt`/`nanochat`) or
`fsdp2` (default for `sinkgd`). With FSDP2, NanoGPT's tied token embedding and language-model head
are sharded together in one FSDP group to preserve weight sharing. Model compilation
(`training.compile_model`) takes about a minute at startup.

### Weights & Biases

The `logger` section configures W&B:

* **Logging in:** log in once with `uv run --no-sync wandb login` (or set `WANDB_API_KEY`).
  Without credentials, startup fails.
* **Entity:** runs log to your default entity; set `logger.entity=<your W&B user or team>` (in
  the YAML or on the command line) to change it.
* **Mode and server:** `logger.mode=offline` logs locally, and `logger.host` points to a custom
  W&B server.
* **Disabling:** `logger=null` turns W&B off.

### Outputs, checkpoints and resuming

**Logs.** Rank 0 prints the training loss every `training.log_interval` iterations, together
with throughput, MFU and the estimated time left. It also prints the validation loss every
`training.eval_interval` iterations, computed over `data.eval_size` batches. The same metrics go
to W&B when enabled.

**Checkpoints.** Every `training.checkpoint_interval` iterations (default 1000; 0 disables
saving), the trainer writes `ckpt-<step>/` into the run's checkpoint directory. It is about 3 GB
for the default 4-GPU NanoGPT run. With `training.checkpoint_path=null` (the default), the
directory is `<compute_env.output_dir>/checkpoints/<optimizer>-<hash>/`, where the hash covers
every setting that affects training (logging, evaluation and checkpoint settings excluded). The
directory also contains `config.json` and a `latest` file naming the newest complete checkpoint.
Checkpoints are only written at multiples of the interval, so the default 2600-iteration run
saves at iterations 1000 and 2000. The trainer keeps all checkpoints.

**Resuming.** Relaunching the same command resumes from `latest`, restoring the model,
optimizer and scheduler states, the RNG state and the position in the training data. For DDP and
single-GPU runs, resume with the same number of GPUs.

* **Changing settings:** changing any training setting starts a fresh run in a new directory.
  To continue a run with different settings (e.g. more iterations), pass its directory
  explicitly: `training.checkpoint_path=outputs/checkpoints/<optimizer>-<hash>`.
* **Multi-node:** the checkpoint directory must be on storage shared by all nodes.
* **W&B:** each launch creates a new W&B run.

## E. SinkGD

SinkGD (Sinkhorn Gradient Descent) is a matrix optimizer based on repeated
row-wise and column-wise L2 normalization. It balances gradient magnitudes
without computing rotation matrices or a QR/SVD decomposition. See
**[Gradient Multi-Normalization for Stateless and Scalable LLM Training](https://arxiv.org/abs/2502.06742)**,
by Meyer Scetbon, Chao Ma, Wenbo Gong, and Edward Meeds (2025).

**Implementation in this library.** [`aro.optim.SinkGD`](src/aro/optim/_sinkgd.py)
is a **momentum-based variant**, not an exact reproduction of the paper's
stateless algorithm. It combines Nesterov-style momentum, near-zero row
filtering, row/column normalization, and global update-RMS alignment. It stores
one momentum tensor per active parameter, but no rotation matrices or Adam
second-moment tensors.

**Where to use it.** Apply SinkGD to trainable, real, nonempty 2D matrices:
linear projections, dense-gradient token embeddings, and language-model heads.
It can be used in ordinary PyTorch models on CPU or CUDA, or through this
repository's NanoGPT/NanoChat trainer. Keep biases, normalization vectors, and
other parameter shapes on a separate optimizer such as AdamW. Sparse gradients
are not supported. For RoM's frozen ADB factors, use `FixedRankRGD` instead.

**Example.** This self-contained CPU example applies SinkGD to matrices and
AdamW to biases. Move the model to CUDA before constructing optimizers if desired,
and move the batches to the same device.

```python
import torch
from torch import nn
from torch.optim import AdamW

from aro.optim import SinkGD

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4))
trainable = [p for p in model.parameters() if p.requires_grad]
matrix_params = [p for p in trainable if p.ndim == 2]
other_params = [p for p in trainable if p.ndim != 2]
optimizers = [SinkGD(matrix_params, lr=1e-3)]
if other_params:
    optimizers.append(
        AdamW(other_params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    )
inputs, targets = torch.randn(16, 8), torch.randn(16, 4)

for _ in range(10):
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    for optimizer in optimizers:
        optimizer.step()
```

**Defaults and normalization modes.** Defaults are `lr=1e-3`, Nesterov momentum
`0.95`, `sinkhorn_steps=11`, `eps=1e-20`, and zero weight decay. Rows whose norm
is at most `1e-3` times the global mean row norm are filtered from the update.
With `constant_norm=True`, a nonzero update is normalized to approximately unit
RMS before `alpha=0.2` scaling; its parameter-step scale therefore depends on
both `lr` and `alpha`. Retune the learning rate when changing normalization.

| Mode | Meaning of `sinkhorn_steps` | Additional options |
|---|---|---|
| `alternating` (default) | A positive odd number of individual row/column passes; 11 means 6 row and 5 column passes by default | `start_axis="row"` or `"col"` selects the first axis |
| `simultaneous` | Any positive number of joint passes, with both norms computed from the same pre-pass matrix | `final_row_pass=True` adds a final row pass; `start_axis` does not affect the update |

Five simultaneous passes plus a final row pass have the same row/column pass
counts as eleven row-first alternating passes, but are **not numerically
equivalent**. `final_row_pass` applies only to simultaneous mode.

**Distributed training and state.** Apply `fully_shard(model)` before constructing
an FSDP2 optimizer. Supported DTensors use a one-dimensional mesh with `Shard(0)`
or `Replicate()`. For row-sharded parameters, momentum and updates remain local;
only column-norm vectors and scalar statistics are all-reduced, with no full
matrix gathers. Uneven and empty local shards are supported. Every rank in a
shard group must step the same ordered active parameters. DDP callers must
synchronize gradients before `step()`.

Momentum is stored in float32, or float64 for double parameters. Incoming
gradients are not modified. Save and load optimizer state with the standard
`state_dict()` / `load_state_dict()` methods when resuming.

**Bundled trainer.** [`config/sinkgd.yaml`](config/sinkgd.yaml) uses NanoGPT,
context length 1024, FSDP2, and disables model compilation, W&B, and checkpoint
saving. It expects `data/openwebtext/train.bin` and `val.bin` from
[data preparation](#c-data-preparation). Select the parameter split with `optimizer=sinkgd`
(default) or `optimizer=mixed-sinkgd`:

| Optimizer name | Parameters handled by SinkGD | Parameters handled by AdamW |
|---|---|---|
| `sinkgd` | All trainable 2D matrices, including embeddings and the head | Remaining trainable parameters |
| `mixed-sinkgd` | Trainable 2D matrices excluding token embeddings and the prediction head | Excluded embeddings/head and remaining trainable parameters |

The AdamW fallback has independent `adamw_betas=(0.9, 0.95)`, `adamw_eps=1e-8`,
and `adamw_weight_decay=0.1`, sharing `optimizer.lr` with SinkGD. Changing
`optimizer.weight_decay` affects SinkGD, not the fallback.

```bash
uv run --no-sync torchrun --standalone --nproc_per_node=4 -m trainer.pretrain --cfg sinkgd

uv run --no-sync torchrun --standalone --nproc_per_node=4 -m trainer.pretrain --cfg sinkgd -- \
  optimizer=mixed-sinkgd optimizer.normalization=simultaneous \
  optimizer.sinkhorn_steps=5 optimizer.final_row_pass=true
```

Use `training.strategy=ddp` for DDP, or one process with
`training.strategy=none` for single-GPU training.

## F. RoM: Rotated Manifold Optimization for low-rank adaptation

RoM extends matrix optimizers to the manifold of fixed-rank matrices by combining
rotation and normalization, accounting for the gauge symmetry of low-rank
factorization. See the paper,
**[Rotated Manifold Optimization for Low-Rank Adaptation](https://arxiv.org/abs/2609.34264)**,
for the method and experiments.

The implementation is available through `aro.lora.FixedRankRGD`. It optimizes the adapter product
`A @ diag(D) @ B.T` on the embedded fixed-rank manifold, rather than differentiating
the individual factors. The implementation uses low-rank gradient projections and
a small SVD retraction, without materializing a dense adapter gradient.

**Quick start.** Install `aro` as in the [Quick start](#1-use-aro-in-your-own-training-loop), freeze the base model, replace its linear
layers with `apply_adb_lora`, and pass the adapter modules (not their parameters)
to `FixedRankRGD`. Enable input gradients, then use `zero_grad()`, `backward()`,
and `step()` in your training loop. The example below uses plain Riemannian gradient
descent; see **Conditioning** below to enable rotation and normalization options.

This self-contained CPU example also works on CUDA after moving the model and inputs:

```python
import torch
from torch import nn

from aro.lora import FixedRankRGD, LoRAADBLinear, apply_adb_lora, merged_adapters

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 4))
model.requires_grad_(False)
apply_adb_lora(model, rank=2)
adapters = [module for module in model.modules() if isinstance(module, LoRAADBLinear)]
optimizer = FixedRankRGD(adapters, lr=0.01, momentum=0.9)
inputs, targets = torch.randn(16, 8), torch.randn(16, 4)

try:
    for _ in range(20):
        optimizer.zero_grad()
        # Factors are frozen: inputs must require grad so the capture hooks run.
        predictions = model(inputs.detach().requires_grad_(True))
        loss = nn.functional.mse_loss(predictions, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad(), merged_adapters(model):
        predictions = model(inputs)
finally:
    optimizer.remove_hooks()
```

For pretrained models, `apply_adb_lora` replaces linear descendants in place and
copies their frozen weights and biases. Use its `exclude(name, module)` callback
to skip layers such as the output head. Other parameters are not frozen by the
helper; freeze the model first for adapter-only training. For a single linear layer,
use `LoRAADBLinear.from_linear`. Every adapted layer requires
`0 < 2 * rank <= min(in_features, out_features)`. Adapters start near, not exactly at,
the pretrained function (`d_init=1e-3` by default).

**Conditioning.** The default is plain Riemannian gradient descent. Opt into
spectral whitening with `conditioning="spectral"`; add `sinkhorn_steps=2` for reduced
Sinkhorn balancing. Optional `frame_update="qr"` or `"polar"` refines the frame;
`track_frame=True` carries it across steps (tracking with `"qr"` requires positive
Sinkhorn steps). `rms_normalize=True` independently normalizes the step. Retune the
learning rate when changing conditioning, normalization, or adapter rank.

**Training integration.** With frozen embeddings, enable input gradients before
training (for example, `model.enable_input_require_grads()` in compatible model
libraries). Accumulated gradients are consumed by `step()` and cleared by
`zero_grad()`; average microbatch losses or set `gradient_divisor` explicitly.
Factor `.grad` values are always `None`, so standard factor-gradient clipping and
`GradScaler.step(optimizer)` do not apply. For loss-scaled backward passes, supply
`optimizer.step(grad_scale=...)` and use `optimizer.found_non_finite` to manage the
loss scale explicitly. Call `remove_hooks()` when discarding the optimizer.

**Evaluation and checkpoints.** `disabled_adapters(model)` temporarily evaluates
the frozen base model; `merged_adapters(model)` caches effective weights for
evaluation only. Both restore entry states, including nested contexts. Save both
`model.state_dict()` and `optimizer.state_dict()`, then reconstruct the same adapter
layout and optimizer options before loading both. If you load only model weights
into an existing optimizer, call `resync_from_modules()` to reset its fp32 master
state, momentum, and tracked frames.

**Distributed training.** Factors must be replicated and initialized identically
across ranks. RoM all-reduces its captured projections and averages over the process
group by default; it does not rely on DDP factor gradients. FSDP2 may shard frozen
base weights, but factors must be excluded from sharding at every level, placed on
the compute device, and synchronized before constructing the optimizer. Sharded
adapter factors are not supported.

## G. Practical usage and benchmarking guidelines

For fair and informative benchmarks, we recommend aligning optimization settings across methods as closely as possible.

- **RMSNorm alignment**: in our paper, we tune the AdamW baseline and align RMSNorms of Muon and ARO to AdamW, similar to Muon Moonlight. To reproduce results/behavior in the paper, as well as for easier hyperparameter transfer, we recommend keeping RMSNorm aligned when comparing methods.
- **When benchmarking without RMSNorm**: optimization dynamics may differ, so the learning rate should be re-tuned accordingly. In practice, this often requires a substantially different (often larger) learning rate than in RMSNorm-based settings.
- **Variance adaptation parity**: when comparing against optimizers that explicitly use variance adaptation, we recommend applying the corresponding variance-adaptation mechanism to ARO as well, so the comparison better isolates optimizer behavior rather than differences in adaptation features.
- **SCQR for faster rotation updates**: in our work, we proposed to use SCQR instead of a standard QR to accelerate computation. The ARO constructors and example configs use `rotation_jitter=1e-9`; standalone `SCQRUpdate` and `SCQRConfig` default to `1e-7`. If you encounter very frequent fallbacks to QR or other numerical issues, especially for larger models, we recommend trying `1e-7` instead, e.g. with the trainer override `optimizer.u_processor.rotation_jitter=1e-7` or `SCQRUpdate(rotation_jitter=1e-7)` in your own code. `chol-qr` / `CholQrUpdate` are aliases of SCQR.

    For readers interested in related follow-up discussions and extensions, Kexue.fm has a nice series, for example https://kexue.fm/archives/11654 and https://kexue.fm/archives/11710.
