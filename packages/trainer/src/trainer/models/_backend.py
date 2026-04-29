import logging
import os
from typing import Any

import torch

from trainer.utils.device import DeviceLikeType

logger = logging.getLogger(__name__)


def log_on_main_worker(message: str, *args: Any, level: int = logging.INFO):
    """Logs a message only on the worker with number (rank) 0."""
    if int(os.getenv("RANK", "0")) == 0:
        logger.log(level, message, *args)


def sdpa_probe(
    tag: str,
    device: DeviceLikeType,
    dtype: torch.dtype,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    if not torch.cuda.is_available():
        return
    try:
        from torch.backends.cuda import (
            SDPAParams,
            can_use_cudnn_attention,
            can_use_efficient_attention,
            can_use_flash_attention,
            cudnn_sdp_enabled,
            flash_sdp_enabled,
            math_sdp_enabled,
            mem_efficient_sdp_enabled,
        )
        from torch.nn.attention import SDPBackend
    except Exception as exc:
        log_on_main_worker(f"[SDPA Probe] {tag}: failed to import SDPA helpers: {exc!r}")
        return

    def report(params: SDPAParams) -> tuple[dict[str, bool], dict[str, bool], str | None]:
        enabled = {
            "flash": flash_sdp_enabled(),
            "mem_efficient": mem_efficient_sdp_enabled(),
            "math": math_sdp_enabled(),
            "cudnn": cudnn_sdp_enabled(),
        }
        can_use = {
            "flash": can_use_flash_attention(params),
            "mem_efficient": can_use_efficient_attention(params),
            "cudnn": can_use_cudnn_attention(params),
            "math": True,
        }
        backend_names = {
            SDPBackend.FLASH_ATTENTION.value: "flash",
            SDPBackend.EFFICIENT_ATTENTION.value: "mem_efficient",
            SDPBackend.MATH.value: "math",
            SDPBackend.CUDNN_ATTENTION.value: "cudnn",
        }
        selected = None
        for backend_id in torch._C._get_sdp_priority_order():
            name = backend_names.get(backend_id)
            if name and enabled.get(name) and can_use.get(name):
                selected = name
                break
        return enabled, can_use, selected

    q = torch.zeros((batch_size, num_heads, seq_len, head_dim), device=device, dtype=dtype)
    v = torch.zeros((batch_size, num_kv_heads, seq_len, head_dim), device=device, dtype=dtype)
    k = v
    params = SDPAParams(q, k, v, None, 0.0, True, False)
    enabled, can_use, selected = report(params)
    log_on_main_worker(
        f"[SDPA Probe] {tag}: heads={num_heads}, kv_heads={num_kv_heads}, "
        + f"head_dim={head_dim}, seq={seq_len}, bs={batch_size}, "
        + f"enabled={enabled}, can_use={can_use}, selected={selected}"
    )
