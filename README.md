# :sparkles: ARO

[![main](https://github.com/microsoft/aro/actions/workflows/main.yml/badge.svg)](https://github.com/microsoft/aro/actions/workflows/main.yml)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.io/pyright/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Adaptive Rotation Optimizers

ARO is a library of Pytorch optimizers for LLM training. For more details, see our report:

**[ARO: A New Lens On Matrix Optimization For Large Models](https://arxiv.org/abs/2602.090062)**

It includes a range of advanced optimizers designed to improve training efficiency and model performance.
These optimizers process and rotate gradients prior to updating model parameters, leveraging
structures from the model architecture.

ARO includes the following optimizers:

* AroAdam: An adaptive learning rate optimizer that tracks first, second moments of gradients and incorporates gradient rotation.
* AroSinkhorn: A variant that applies Sinkhorn normalization to the gradients (transforms matrices to doubly stochastic) and then applies rotation.
* AroSign: An optimizer that applies rotation to the sign of the gradients, inspired by signSGD.

Our default in the paper uses AroSinkhorn, which provides robust overall performance. In certain cases, AroAdam can outperform AroSinkho

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

model = ...  # your model
lr = 6e-4
grad_clip = 1.0
max_iters = 100000
loss_fn = torch.nn.CrossEntropyLoss()
train_iterator = ...  # your data iterator

# separate parameters into 2 groups:
# 1) AdamW params: lm_head + 1d params (LayerNorm,  biases)
# 2) ARO params: 2D params (weights of Linear, Conv, Embedding layers)
head_params = list(model.lm_head.parameters())
oned_params = [p for p in model.parameters() if p.requires_grad and p.ndim == 1]
adamw_params = head_params + oned_params
aro_params = [
    p
    for p in model.parameters()
    if p.requires_grad and all(p is not q for q in adamw_params)
]

adamw_opt = AdamW(adamw_params, lr=lr, betas=(0.9, 0.95))
aro_opt = AroSinkhorn(aro_params, lr=lr)
optimizers = [adamw_opt, aro_opt]
```

2. Training loop (with clipping and bf16 mixed precision)

```python
step = 0
while True:
    inputs, labels = next(train_iterator)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
    loss.backward()

    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    for opt in optimizers:
        opt.step()
        opt.zero_grad()

    step += 1
    if step > max_iters:
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
python -m dataprep.openwebtext
```

3. Run a training example:

```bash
torchrun --standalone --nproc_per_node=4 -m trainer.pretrain --cfg nanogpt -- <optional-overrides>
```

Here, `<optional-overrides>` are command-line [hydra overrides](https://hydra.cc/docs/advanced/override_grammar/basic/).
For example, `optimizer.lr=0.001` to change the learning rate:
```bash
torchrun --standalone --nproc_per_node=4 -m trainer.pretrain --cfg nanogpt -- optimizer.lr=0.001
```

Configuration values are validated with pydantic at runtime.

Use `training.strategy=fsdp2` to run with FSDP2 instead of DDP. NanoGPT's tied
token embedding and language-model head are sharded together in one FSDP group
to preserve weight sharing.

## :compass: Practical usage and benchmarking guidelines


For fair and informative benchmarks, we recommend aligning optimization settings across methods as closely as possible.

- **RMSNorm alignment**: in our paper, we tune the AdamW baseline and align RMSNorms of Muon and ARO to AdamW, similar to Muon Moonlight. To reproduce results/behavior in the paper, as well as for easier hyperparameter transfer, we recommend keeping RMSNorm aligned when comparing methods.
- **When benchmarking without RMSNorm**: optimization dynamics may differ, so the learning rate should be re-tuned accordingly. In practice, this often requires a substantially different (often larger) learning rate than in RMSNorm-based settings.
- **Variance adaptation parity**: when comparing against optimizers that explicitly use variance adaptation, we recommend applying the corresponding variance-adaptation mechanism to ARO as well, so the comparison better isolates optimizer behavior rather than differences in adaptation features.
- **SCQR for faster rotation updates**: in our work, we proposed to use SCQR instead of a standard QR to accelerate computation. The default `rotation_jitter` is `1e-9`. If you encounter very frequent fallbacks to QR or other numerical issues, especially for larger models, we recommend trying `1e-7` instead.

    You can adjust this directly in the provided configs:

    ```yaml
    optimizer:
        u_processor:
            name: scqr
            rotation_jitter: 1e-7
    ```

    For readers interested in related follow-up discussions and extensions, Kexue.fm has a nice series, for example https://kexue.fm/archives/11654 and https://kexue.fm/archives/11710.

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
