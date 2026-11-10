"""Model conversion and evaluation helpers for ADB adapters."""

from collections.abc import Callable, Generator
from contextlib import contextmanager

import torch.nn as nn

from ._adb_linear import LoRAADBLinear


@contextmanager
def disabled_adapters(model: nn.Module) -> Generator[None, None, None]:
    """Temporarily disable adapters, restoring each layer's entry state on exit."""
    layers = [module for module in model.modules() if isinstance(module, LoRAADBLinear)]
    previous = [module._adapter_enabled for module in layers]
    for module in layers:
        module.set_adapter_enabled(False)
    try:
        yield
    finally:
        for module, enabled in zip(layers, previous, strict=True):
            module.set_adapter_enabled(enabled)


@contextmanager
def merged_adapters(model: nn.Module) -> Generator[None, None, None]:
    """Cache effective weights for evaluation, restoring the entry merge states.

    Use with ``model.eval()`` and keep adapter factors unchanged during the block.
    Merged and unmerged outputs can differ within floating-point rounding; use
    one path consistently when comparing evaluation results. Previously merged
    caches are rebuilt on exit to preserve dtype/device moves inside the block.
    """
    layers = [module for module in model.modules() if isinstance(module, LoRAADBLinear)]
    previously_merged = [module._merged_weight is not None for module in layers]
    for module in layers:
        module.merge_weight()
    try:
        yield
    finally:
        for module, was_merged in zip(layers, previously_merged, strict=True):
            if was_merged:
                module.merge_weight()
            else:
                module.unmerge_weight()


def apply_adb_lora(
    model: nn.Module,
    rank: int,
    exclude: Callable[[str, nn.Module], bool] | None = None,
    d_init: float = 1e-3,
    manual_factors: bool = False,
) -> nn.Module:
    """Replace linear descendants in place, preserving shared-module aliases.

    Args:
        model: Model to modify. Nonlinear modules are left unchanged, including
            their ``requires_grad`` flags; freeze them separately if needed.
        rank: Adapter rank; requires ``0 < 2 * rank <= min(in, out)`` per layer.
        exclude: Callback ``(full_name, module) -> bool``; return True to skip a
            layer. Excluding one alias does not exclude other paths to that layer.
        d_init: Initial signed singular value (default ``1e-3``).
        manual_factors: Store the frozen factors as buffers instead of parameters.

    Returns:
        The same model reference.

    Raises:
        ValueError: If model is a bare ``nn.Linear`` (use
            ``LoRAADBLinear.from_linear`` instead), or a selected rank is invalid.
    """
    if isinstance(model, nn.Linear):
        raise ValueError("apply_adb_lora cannot replace a bare nn.Linear in place")

    converted: dict[int, LoRAADBLinear] = {}
    replacements: list[tuple[str, LoRAADBLinear]] = []
    for name, module in model.named_modules(remove_duplicate=False):
        if not isinstance(module, nn.Linear):
            continue
        if exclude is not None and exclude(name, module):
            continue
        replacement = converted.get(id(module))
        if replacement is None:
            replacement = LoRAADBLinear.from_linear(
                module, rank, d_init=d_init, manual_factors=manual_factors
            )
            converted[id(module)] = replacement
        replacements.append((name, replacement))

    for name, replacement in replacements:
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], replacement)
    return model
