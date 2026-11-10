"""Public ADB model conversion, evaluation contexts, and training integration."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.testing

from aro.lora import (
    FixedRankRGD,
    LoRAADBLinear,
    apply_adb_lora,
    disabled_adapters,
    merged_adapters,
)


def test_conversion_preserves_base_weights_aliases_and_exclusions() -> None:
    shared = nn.Linear(8, 8).eval()
    head = nn.Linear(8, 2).eval()
    model = nn.ModuleDict({"left": shared, "right": shared, "head": head})

    result = apply_adb_lora(
        model, rank=2, exclude=lambda name, _: name == "head", manual_factors=True
    )

    assert result is model
    assert isinstance(model["left"], LoRAADBLinear)
    layer = model["left"]
    assert layer is model["right"]
    assert model["head"] is head
    assert not layer.training
    torch.testing.assert_close(layer.weight, shared.weight, atol=0, rtol=0)
    assert layer.bias is not None
    torch.testing.assert_close(layer.bias, shared.bias, atol=0, rtol=0)
    assert {"A", "B", "d_raw"} <= dict(layer.named_buffers()).keys()
    assert all(not parameter.requires_grad for parameter in layer.parameters())


def test_conversion_failure_leaves_model_unchanged() -> None:
    first, second = nn.Linear(8, 8), nn.Linear(8, 2)
    model = nn.Sequential(first, second)

    with pytest.raises(ValueError, match=r"2 \* rank"):
        apply_adb_lora(model, rank=2)

    assert model[0] is first
    assert model[1] is second


def test_bare_linear_requires_explicit_conversion() -> None:
    with pytest.raises(ValueError, match="cannot replace a bare"):
        apply_adb_lora(nn.Linear(8, 8), rank=2)


def test_disabled_context_restores_mixed_states_after_nested_exception() -> None:
    first, second = LoRAADBLinear(8, 8, rank=2), LoRAADBLinear(8, 8, rank=2)
    model = nn.Sequential(first, second)
    first.set_adapter_enabled(False)
    x = torch.randn(3, 8)
    expected = model(x)
    base = F.linear(F.linear(x, first.weight), second.weight)

    with pytest.raises(RuntimeError, match="evaluation failed"), disabled_adapters(model):
        torch.testing.assert_close(model(x), base)
        with disabled_adapters(model):
            assert not first._adapter_enabled and not second._adapter_enabled
        assert not first._adapter_enabled and not second._adapter_enabled
        raise RuntimeError("evaluation failed")

    assert not first._adapter_enabled and second._adapter_enabled
    torch.testing.assert_close(model(x), expected)


def test_merged_context_restores_mixed_states_after_nested_exception_and_move() -> None:
    first, second = LoRAADBLinear(8, 8, rank=2), LoRAADBLinear(8, 8, rank=2)
    model = nn.Sequential(first, second).eval()
    x = torch.randn(3, 8)
    expected = model(x)
    first.merge_weight()

    with pytest.raises(RuntimeError, match="evaluation failed"), merged_adapters(model):
        torch.testing.assert_close(model(x), expected, atol=1e-5, rtol=1e-5)
        with merged_adapters(model):
            model.to(torch.float64)
        assert first._merged_weight is not None and second._merged_weight is not None
        raise RuntimeError("evaluation failed")

    assert first._merged_weight is not None
    assert first._merged_weight.dtype == torch.float64
    assert second._merged_weight is None
    torch.testing.assert_close(model(x.double()), expected.double(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("spectral", [False, True])
def test_public_api_training_updates_only_adapters_and_resumes(spectral: bool) -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 4))
    model.requires_grad_(False)
    apply_adb_lora(model, rank=2, d_init=0.1)
    adapters = [module for module in model.modules() if isinstance(module, LoRAADBLinear)]
    base_weights = [layer.weight.detach().clone() for layer in adapters]
    original = [layer._adapter().clone() for layer in adapters]
    optimizer = FixedRankRGD(
        adapters,
        lr=0.01,
        momentum=0.9,
        conditioning="spectral" if spectral else "none",
        sinkhorn_steps=2 if spectral else 0,
    )
    x, target = torch.randn(16, 8), torch.randn(16, 4)
    try:
        initial_loss = F.mse_loss(model(x), target).item()
        for _ in range(10):
            optimizer.zero_grad()
            F.mse_loss(model(x.detach().requires_grad_(True)), target).backward()
            optimizer.step()

        assert F.mse_loss(model(x), target).item() < initial_loss
        for layer, base, before in zip(adapters, base_weights, original, strict=True):
            torch.testing.assert_close(layer.weight, base, atol=0, rtol=0)
            assert not torch.equal(layer._adapter(), before)
            assert layer.A.grad is None and layer.B.grad is None and layer.d_raw.grad is None

        resumed = nn.Sequential(nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 4))
        resumed.requires_grad_(False)
        apply_adb_lora(resumed, rank=2)
        resumed.load_state_dict(model.state_dict())
        resumed_optimizer = FixedRankRGD(
            [module for module in resumed.modules() if isinstance(module, LoRAADBLinear)],
            conditioning="spectral" if spectral else "none",
            sinkhorn_steps=2 if spectral else 0,
        )
        try:
            resumed_optimizer.load_state_dict(optimizer.state_dict())
            for net, opt in ((model, optimizer), (resumed, resumed_optimizer)):
                opt.zero_grad()
                F.mse_loss(net(x.detach().requires_grad_(True)), target).backward()
                opt.step()
            torch.testing.assert_close(model(x), resumed(x), atol=0, rtol=0)
        finally:
            resumed_optimizer.remove_hooks()
    finally:
        optimizer.remove_hooks()
