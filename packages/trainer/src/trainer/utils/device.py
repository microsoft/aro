import contextlib
import logging
import os
from typing import Any

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from torch import memory_format
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP

type TS = dict[str, TS] | list[TS] | torch.Tensor | float | int | None
DeviceLikeType = str | torch.device | int

logger = logging.getLogger(__name__)


def log_on_main_worker(message: str, *args: Any, level: int = logging.INFO):
    """Logs a message only on the worker with number (rank) 0."""
    if int(os.getenv("RANK", "0")) == 0:
        logger.log(level, message, *args)


def to_device[T: TS](
    batch: T,
    device: DeviceLikeType,
    non_blocking: bool = False,
    copy: bool = False,
    *,
    memory_format: memory_format | None = None,
) -> T:
    """Recursively move tensors to the given device."""
    return pytree.tree_map_only(
        torch.is_tensor,
        lambda x: x.to(device, non_blocking=non_blocking, copy=copy, memory_format=memory_format),
        batch,
    )


def get_device() -> torch.device:
    """Returns the device for the current worker."""
    if torch.cuda.is_available():
        local_rank = os.getenv("LOCAL_RANK", "0")
        return torch.device(f"cuda:{local_rank}")
    else:
        return torch.device("cpu")


@contextlib.contextmanager
def distributed_grad_sync(model: nn.Module | DDP, sync: bool):
    """Context manager that disables gradient sync on the model if requested.

    Works with non-distributed and DDP/FSDP classic.
    """
    if hasattr(model, "require_backward_grad_sync"):
        assert isinstance(model, DDP) or isinstance(model, FSDP)
        model.require_backward_grad_sync = sync  # type: ignore
        yield
        return

    yield  # fallback no-op


def get_peak_flops(device_name: str) -> float:
    name = device_name.lower()

    # --- NVIDIA Blackwell ---
    if "gb200" in name or "grace blackwell" in name:
        return 2.5e15
    if "b200" in name:
        return 2.25e15
    if "b100" in name:
        return 1.8e15

    # --- NVIDIA Hopper (H100/H200/H800) ---
    if "h200" in name:
        if "nvl" in name or "pcie" in name:
            return 836e12
        return 989e12  # H200 SXM
    if "h100" in name:
        if "nvl" in name:
            return 835e12
        if "pcie" in name:
            return 756e12
        return 989e12  # H100 SXM
    if "h800" in name:
        if "nvl" in name:
            return 989e12
        return 756e12  # H800 PCIe

    # --- NVIDIA Ampere data center ---
    if "a100" in name or "a800" in name:
        return 312e12
    if "a40" in name:
        return 149.7e12
    if "a30" in name:
        return 165e12

    # --- NVIDIA Ada data center ---
    if "l40s" in name or "l40-s" in name or "l40 s" in name:
        return 362e12
    if "l4" in name:
        return 121e12

    # --- AMD CDNA accelerators ---
    if "mi355" in name:
        return 2.5e15
    if "mi325" in name or "mi300x" in name:
        return 1.3074e15
    if "mi300a" in name:
        return 980.6e12
    if "mi250x" in name:
        return 383e12
    if "mi250" in name:
        return 362.1e12

    # --- Intel ---
    if "data center gpu max 1550" in name:
        # Ponte Vecchio (PVC) - dynamic based on compute units
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6

    # --- Consumer RTX (for hobbyists) ---
    if "5090" in name:
        return 209.5e12
    if "4090" in name:
        return 165.2e12
    if "3090" in name:
        return 71e12

    # Unknown GPU - return inf so MFU shows as 0% rather than a wrong guess
    msg = f"Peak flops undefined for: {device_name}, MFU will show as 0%"
    log_on_main_worker(msg, level=logging.WARNING)
    return float("inf")
