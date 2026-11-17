from typing import Any

import pytest
import torch

from aro.optim import SinkGD
from gpt import NanoGPT
from trainer.config import ConfigContainer, ConfigServices
from trainer.config.optim import SinkGDConfig
from trainer.models import ModelContainer, ModelServices
from trainer.optim import OptimContainer, OptimServices


@pytest.mark.parametrize("name", ["sinkgd", "mixed-sinkgd"])
@pytest.mark.parametrize("mode,steps", [("alternating", 11), ("simultaneous", 5)])
def test_native_factory_routes_parameters_and_preserves_defaults(
    monkeypatch: pytest.MonkeyPatch, name: str, mode: str, steps: int
) -> None:
    monkeypatch.setattr("trainer.models._services.get_device", lambda: torch.device("cpu"))
    config = ConfigContainer(
        "config",
        "sinkgd",
        [
            "model.n_layer=1",
            "model.n_head=2",
            "model.n_embd=8",
            "model.vocab_size=32",
            "model.sequence_len=8",
            "model.bias=true",
            "training.strategy=none",
            "training.compile_model=false",
            f"optimizer.name={name}",
            f"optimizer.normalization={mode}",
            f"optimizer.sinkhorn_steps={steps}",
        ],
    ).get(ConfigServices.BASE_CONFIG)
    assert isinstance(config.optimizer, SinkGDConfig)
    models = ModelContainer(config)
    container = OptimContainer(config, models)
    optimizers = container.get(OptimServices.OPTIMIZERS)
    assert container.get(OptimServices.OPTIMIZERS) is optimizers
    model = models.get(ModelServices.RAW_MODEL)
    assert isinstance(model, NanoGPT)
    owners = {
        id(parameter): optimizer
        for optimizer in optimizers
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert set(owners) == {id(parameter) for parameter in model.parameters()}
    assert sum(len(group["params"]) for opt in optimizers for group in opt.param_groups) == len(
        owners
    )
    head_ids = {id(parameter) for parameter in model.lm_head.parameters()}
    for parameter in model.parameters():
        expected = parameter.ndim == 2 and (name == "sinkgd" or id(parameter) not in head_ids)
        assert isinstance(owners[id(parameter)], SinkGD) is expected
    for optimizer in optimizers:
        group = optimizer.param_groups[0]
        if isinstance(optimizer, SinkGD):
            assert group["normalization"] == mode and group["sinkhorn_steps"] == steps
            assert group["weight_decay"] == 0 and group["eps"] == 1e-20
        else:
            assert group["weight_decay"] == 0.1 and group["eps"] == 1e-8
            assert group["betas"] == (0.9, 0.95)
    tokens = torch.randint(0, 32, (2, 8))
    _, loss = model(tokens, tokens)
    loss.backward()
    for optimizer in optimizers:
        optimizer.step()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


@pytest.mark.parametrize(
    "options",
    [
        {"sinkhorn_steps": 2},
        {"sinkhorn_steps": 0},
        {"normalization": "invalid"},
        {"final_row_pass": True},
    ],
)
def test_config_rejects_invalid_normalization(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SinkGDConfig(**options)
