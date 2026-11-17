from dataclasses import replace

import pytest
import torch
from pydantic import ValidationError

from aro.optim import AroSign
from aro.processors.u import CholQrUpdate
from trainer.config import ConfigContainer, ConfigServices
from trainer.config.optim import (
    AroAdamConfig,
    AroSignConfig,
    AroSinkhornConfig,
    CholQrConfig,
    PolarConfig,
    SCQRConfig,
    SinkGDConfig,
)
from trainer.models import ModelContainer, ModelServices
from trainer.optim import OptimContainer, OptimServices


def test_polar_steps_accept_the_internal_override_syntax() -> None:
    config = ConfigContainer(
        "config",
        "nanogpt",
        ["optimizer.u_processor.name=polar", "optimizer.u_processor.steps=2"],
    ).get(ConfigServices.BASE_CONFIG)
    assert isinstance(config.optimizer, AroSinkhornConfig)
    assert config.optimizer.u_processor == PolarConfig(steps=2)


def test_optimizer_recipe_replaces_the_whole_optimizer_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ConfigContainer("config", "nanogpt", ["optimizer=sinkgd"]).get(
        ConfigServices.BASE_CONFIG
    )
    assert config.optimizer == SinkGDConfig()

    with pytest.raises(ValidationError, match="sinkhorn_steps"):
        ConfigContainer("config", "nanogpt", ["optimizer.name=muon"]).get(
            ConfigServices.BASE_CONFIG
        )

    monkeypatch.setattr("trainer.models._services.get_device", lambda: torch.device("cpu"))
    overrides = ["model.n_layer=1", "model.n_head=2", "model.n_embd=8", "model.vocab_size=32"]
    overrides += ["training.strategy=none", "training.compile_model=false", "optimizer=adamw"]
    overrides += ["optimizer.betas=[0.8,0.9]", "optimizer.weight_decay=0.2"]
    config = ConfigContainer("config", "nanogpt", overrides).get(ConfigServices.BASE_CONFIG)
    (adamw,) = OptimContainer(config, ModelContainer(config)).get(OptimServices.OPTIMIZERS)
    assert adamw.param_groups[0]["betas"] == (0.8, 0.9)
    assert adamw.param_groups[0]["weight_decay"] == 0.2


@pytest.mark.parametrize("model_name", ["nanogpt", "nanochat"])
def test_shipped_settings_match_reference_updates_and_decay(
    monkeypatch: pytest.MonkeyPatch, model_name: str
) -> None:
    monkeypatch.setattr("trainer.models._services.get_device", lambda: torch.device("cpu"))
    overrides = [
        "model.n_layer=1",
        "model.n_head=2",
        "model.n_embd=8",
        "model.vocab_size=32",
        "model.sequence_len=8",
        "training.strategy=none",
        "training.compile_model=false",
        "training.max_iters=6",
        "lr_scheduler.warmup_iters=1",
        "optimizer=mixed-aroadam",
        "logger=null",
    ]
    if model_name == "nanochat":
        overrides.append("model.n_kv_head=2")
    config = ConfigContainer("config", model_name, overrides).get(ConfigServices.BASE_CONFIG)
    assert isinstance(config.optimizer, AroAdamConfig)
    assert config.lr_scheduler is not None
    reference_config = replace(
        config,
        optimizer=replace(
            config.optimizer,
            lr=6e-4 if model_name == "nanogpt" else 2e-3,
            betas=(0.0, 0.95),
            u_processor=CholQrConfig(rotation_jitter=1e-9),
        ),
        lr_scheduler=replace(
            config.lr_scheduler,
            min_lr=6e-5 if model_name == "nanogpt" else 2e-4,
            lr_decay_iters=config.training.max_iters,
        ),
    )
    models = ModelContainer(config)
    references = ModelContainer(reference_config)
    torch.manual_seed(531)
    model = models.get(ModelServices.DISTRIBUTED_MODEL)
    torch.manual_seed(531)
    reference = references.get(ModelServices.DISTRIBUTED_MODEL)
    containers = [OptimContainer(config, models), OptimContainer(reference_config, references)]
    optimizers = [container.get(OptimServices.OPTIMIZERS) for container in containers]
    schedulers = [container.get(OptimServices.SCHEDULERS) for container in containers]
    for _ in range(config.training.max_iters):
        tokens = torch.randint(0, 32, (2, 8))
        for network, opts, schs in zip((model, reference), optimizers, schedulers, strict=True):
            result = network(tokens, tokens)
            loss = result[1] if isinstance(result, tuple) else result
            loss.backward()
            for optimizer in opts:
                optimizer.step()
                optimizer.zero_grad()
            for scheduler in schs:
                scheduler.step()
        for parameter, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
        assert [opt.param_groups[0]["lr"] for opt in optimizers[0]] == [
            opt.param_groups[0]["lr"] for opt in optimizers[1]
        ]


@pytest.mark.parametrize("rotation", ["scqr", "chol-qr"])
def test_full_arosign_factory_accepts_rotation_names(
    monkeypatch: pytest.MonkeyPatch, rotation: str
) -> None:
    monkeypatch.setattr("trainer.models._services.get_device", lambda: torch.device("cpu"))
    config = ConfigContainer(
        "config",
        "nanogpt",
        [
            "model.n_layer=1",
            "model.n_head=2",
            "model.n_embd=8",
            "model.vocab_size=32",
            "model.sequence_len=8",
            "model.bias=true",
            "training.strategy=none",
            "training.compile_model=false",
            "optimizer=arosign",
            f"optimizer.u_processor.name={rotation}",
            "~optimizer.u_processor.rotation_jitter",
        ],
    ).get(ConfigServices.BASE_CONFIG)
    assert isinstance(config.optimizer, AroSignConfig)
    assert isinstance(config.optimizer.u_processor, SCQRConfig | CholQrConfig)
    models = ModelContainer(config)
    model = models.get(ModelServices.DISTRIBUTED_MODEL)
    optimizers = OptimContainer(config, models).get(OptimServices.OPTIMIZERS)
    matrix_optimizer, _ = optimizers
    assert isinstance(matrix_optimizer, AroSign)
    owners = [
        parameter
        for optimizer in optimizers
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len(owners) == len({id(parameter) for parameter in model.parameters()})
    assert {id(parameter) for parameter in owners} == {
        id(parameter) for parameter in model.parameters()
    }

    parameter = matrix_optimizer.param_groups[0]["params"][0]
    reference = torch.nn.Parameter(parameter.detach().clone())
    expected = AroSign(
        [reference],
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
        u_processor=CholQrUpdate(),
    )
    gradient = torch.randn_like(parameter)
    parameter.grad = gradient.clone()
    reference.grad = gradient.clone()
    matrix_optimizer.step()
    expected.step()
    torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
