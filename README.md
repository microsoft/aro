# :sparkles: ARO

[![main](https://github.com/microsoft/aro/actions/workflows/main.yml/badge.svg)](https://github.com/microsoft/aro/actions/workflows/main.yml)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.io/pyright/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Adaptive Rotation Optimizers

ARO is a library of Pytorch optimizers for LLM training. For more details, see our report:

**[ARO: A New Lens On Matrix Optimization For Large Models](https://arxiv.org/abs/2602.09006)**

For low-rank adaptation (LoRA), the library also includes
[RoM (Rotated Manifold Optimization)](#rom-rotated-manifold-optimization-for-low-rank-adaptation),
introduced in **[Rotated Manifold Optimization for Low-Rank Adaptation](https://arxiv.org/abs/2609.34264)**.

It includes a range of advanced optimizers designed to improve training efficiency and model performance.
These optimizers process and rotate gradients prior to updating model parameters, leveraging
structures from the model architecture.

ARO includes the following optimizers:

* AroAdam: An adaptive learning rate optimizer that tracks first, second moments of gradients and incorporates gradient rotation.
* AroSinkhorn: A variant that applies Sinkhorn-style row/column L2 normalization to gradients in a rotated basis and then rotates the update back.
* AroSign: An optimizer that applies rotation to the sign of the gradients, inspired by signSGD.
* [SinkGD](#sinkgd): Rotation-free Nesterov momentum with alternating or simultaneous row/column normalization and shard-local FSDP2 updates.

Our default in the paper uses AroSinkhorn, which provides robust overall performance. In certain cases, AroAdam can outperform AroSinkhorn.

-----------------------------------------------------------------

## :rocket: Getting started with ARO

Add `aro` to your project dependencies, e.g., if using `uv`:

```bash
uv add "aro @ git+https://github.com/microsoft/aro"
```

## :bulb: Training example

Training with ARO optimizers is not different than training with Adam or other standard optimizers:

1. Load your model and optimizer

```python
import torch

from aro.optim import AroSinkhorn
from torch.optim import AdamW

model = ...  # your model, moved to CUDA before constructing the optimizers
lr = 6e-4
grad_clip = 1.0
max_iters = 100000
loss_fn = torch.nn.CrossEntropyLoss()
train_iterator = ...  # your data iterator

# ARO accepts 2D parameters only. Keep the head and all other shapes on AdamW.
trainable = [p for p in model.parameters() if p.requires_grad]
head_ids = {id(p) for p in model.lm_head.parameters()}
aro_params = [p for p in trainable if p.ndim == 2 and id(p) not in head_ids]
aro_ids = {id(p) for p in aro_params}
adamw_params = [p for p in trainable if id(p) not in aro_ids]

optimizers = []
if adamw_params:
    optimizers.append(AdamW(adamw_params, lr=lr, betas=(0.9, 0.95)))
if aro_params:
    optimizers.append(AroSinkhorn(aro_params, lr=lr))
```

2. Training loop (with clipping and bf16 mixed precision)

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


## Running provided examples in this repo

1. Clone and install the repository:

```bash
git clone https://github.com/microsoft/aro.git
cd aro
uv sync
```

2. Download the data:

```bash
uv run --no-sync python -m dataprep.openwebtext
```

3. Run a training example:

```bash
uv run --no-sync torchrun --standalone --nproc_per_node=4 -m trainer.pretrain --cfg nanogpt -- <optional-overrides>
```

Here, `<optional-overrides>` are command-line [hydra overrides](https://hydra.cc/docs/advanced/override_grammar/basic/).
For example, `optimizer.lr=0.001` to change the learning rate:
```bash
uv run --no-sync torchrun --standalone --nproc_per_node=4 -m trainer.pretrain --cfg nanogpt -- optimizer.lr=0.001
```

Configuration values are validated with pydantic at runtime.
Use `logger=null` to disable Weights & Biases, or configure your own logger settings.

Use `training.strategy=fsdp2` to run with FSDP2 instead of DDP. NanoGPT's tied
token embedding and language-model head are sharded together in one FSDP group
to preserve weight sharing.

CUDA ARO optimizers use round-robin matrix ownership with both DDP and FSDP2:
each rank computes rotation updates for its assigned matrices and distributes
the updates to the other ranks. FSDP2 updates support uneven and empty trailing
row shards and one-dimensional submeshes. CPU execution uses local updates.
Both execution paths leave the processed update in each active parameter's
`.grad`; call `zero_grad()` before accumulating the next step.

The NanoGPT example uses learning rate `6e-4`, minimum learning rate `6e-5`, and
decays over `training.max_iters`. Both model examples use Adam betas `(0.0, 0.95)`
when selecting AroAdam. These settings supersede the earlier OSS example defaults.

`SCQRUpdate` and `SCQRConfig` use standalone rotation jitter `1e-7`. The ARO
optimizer constructors and example YAMLs explicitly retain `1e-9`. The legacy
`CholQrUpdate`, `chol_qr_safe_take_stats`, `CholQrConfig`, and `chol-qr` configuration
name remain supported alongside the SCQR names.

## SinkGD

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

**Quick start.** This self-contained CPU example applies SinkGD to matrices and
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
saving. It expects `data/openwebtext/train.bin` and `val.bin` from the preparation
helper above.

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
  optimizer.name=mixed-sinkgd optimizer.normalization=simultaneous \
  optimizer.sinkhorn_steps=5 optimizer.final_row_pass=true
```

Use `training.strategy=ddp` for DDP, or one process with
`training.strategy=none` for single-GPU training.

## :compass: Practical usage and benchmarking guidelines


For fair and informative benchmarks, we recommend aligning optimization settings across methods as closely as possible.

- **RMSNorm alignment**: in our paper, we tune the AdamW baseline and align RMSNorms of Muon and ARO to AdamW, similar to Muon Moonlight. To reproduce results/behavior in the paper, as well as for easier hyperparameter transfer, we recommend keeping RMSNorm aligned when comparing methods.
- **When benchmarking without RMSNorm**: optimization dynamics may differ, so the learning rate should be re-tuned accordingly. In practice, this often requires a substantially different (often larger) learning rate than in RMSNorm-based settings.
- **Variance adaptation parity**: when comparing against optimizers that explicitly use variance adaptation, we recommend applying the corresponding variance-adaptation mechanism to ARO as well, so the comparison better isolates optimizer behavior rather than differences in adaptation features.
- **SCQR for faster rotation updates**: in our work, we proposed to use SCQR instead of a standard QR to accelerate computation. The ARO constructors and example YAMLs use `rotation_jitter=1e-9`; standalone `SCQRUpdate` and `SCQRConfig` default to `1e-7`. If you encounter very frequent fallbacks to QR or other numerical issues, especially for larger models, we recommend trying `1e-7` instead.

    You can adjust this directly in the provided configs:

    ```yaml
    optimizer:
        u_processor:
            name: scqr
            rotation_jitter: 1e-7
    ```

    For readers interested in related follow-up discussions and extensions, Kexue.fm has a nice series, for example https://kexue.fm/archives/11654 and https://kexue.fm/archives/11710.

## RoM: Rotated Manifold Optimization for low-rank adaptation

RoM extends matrix optimizers to the manifold of fixed-rank matrices by combining
rotation and normalization, accounting for the gauge symmetry of low-rank
factorization. See the paper,
**[Rotated Manifold Optimization for Low-Rank Adaptation](https://arxiv.org/abs/2609.34264)**,
for the method and experiments.

The implementation is available through `aro.lora.FixedRankRGD`. It optimizes the adapter product
`A @ diag(D) @ B.T` on the embedded fixed-rank manifold, rather than differentiating
the individual factors. The implementation uses low-rank gradient projections and
a small SVD retraction, without materializing a dense adapter gradient.

**Quick start.** Install `aro` as above, freeze the base model, replace its linear
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

## :package: Dependencies

The only direct python dependency of `aro` library is `torch`.

--------------------------------------------

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
