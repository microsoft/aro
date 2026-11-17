"""Rotation-free SinkGD optimizer."""

import math
from collections.abc import Callable
from typing import Any, Literal

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT


def _local(tensor: Tensor) -> Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _norm(tensor: Tensor, dim: int | None, group: dist.ProcessGroup | None) -> Tensor:
    norm = torch.linalg.vector_norm(tensor, dim=dim, keepdim=dim is not None)
    if group is not None and dim != 1:
        norm.square_()
        dist.all_reduce(norm, group=group)
        norm.sqrt_()
    return norm


class SinkGD(Optimizer):
    """Nesterov momentum with row/column balancing, without rotations.

    Defaults use eleven alternating passes starting with rows, near-zero row
    filtering, and global RMS alignment followed by ``alpha=0.2`` scaling.

    ``normalization="alternating"`` counts individual row or column passes;
    ``"simultaneous"`` counts joint passes whose row and column norms are both
    computed before either division. ``final_row_pass`` optionally appends one
    row pass to simultaneous normalization. Without ``constant_norm``, scaling
    uses the square root of the last normalized dimension (columns for
    simultaneous mode), as in the source implementation.

    Real, dense 2D parameters are supported on CPU and CUDA. FSDP2 parameters
    must use a one-dimensional mesh with ``Shard(0)`` or ``Replicate()``.
    Momentum and updates stay local: only column-norm vectors and scalar
    statistics are all-reduced within the parameter's shard group. Uneven and
    empty local shards are supported; full matrices are never gathered.
    Ranks in each shard group must step the same ordered active parameters.

    Momentum uses float32, or float64 for double parameters, and is the only
    persistent optimizer tensor. Incoming gradients are not modified. DDP
    callers supply already-synchronized ordinary gradients.
    """

    def __init__(
        self,
        params: ParamsT,
        *,
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        sinkhorn_steps: int = 11,
        eps: float = 1e-20,
        row_filter_threshold: float = 1e-3,
        alpha: float = 0.2,
        constant_norm: bool = True,
        weight_decay: float = 0.0,
        start_axis: Literal["row", "col"] = "row",
        normalization: Literal["alternating", "simultaneous"] = "alternating",
        final_row_pass: bool = False,
    ) -> None:
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                sinkhorn_steps=sinkhorn_steps,
                eps=eps,
                row_filter_threshold=row_filter_threshold,
                alpha=alpha,
                constant_norm=constant_norm,
                weight_decay=weight_decay,
                start_axis=start_axis,
                normalization=normalization,
                final_row_pass=final_row_pass,
            ),
        )
        for group in self.param_groups:
            self._validate_group(group)

    @staticmethod
    def _shard_group(parameter: Tensor) -> dist.ProcessGroup | None:
        if not isinstance(parameter, DTensor):
            return None
        if parameter.device_mesh.ndim != 1 or parameter.placements not in (
            (Shard(0),),
            (Replicate(),),
        ):
            raise ValueError("SinkGD requires a 1D DTensor mesh with Shard(0) or Replicate()")
        if parameter.placements == (Shard(0),) and parameter.device_mesh.size() > 1:
            return parameter.device_mesh.get_group()
        return None

    @classmethod
    def _validate_group(cls, group: dict[str, Any]) -> None:
        for name in ("lr", "eps", "momentum", "row_filter_threshold", "alpha", "weight_decay"):
            if not math.isfinite(group[name]) or group[name] < 0:
                raise ValueError(f"SinkGD {name} must be finite and nonnegative")
        if group["eps"] == 0 or group["momentum"] >= 1:
            raise ValueError("SinkGD requires eps > 0 and momentum < 1")
        mode = group.get("normalization", "alternating")
        if mode not in ("alternating", "simultaneous"):
            raise ValueError("SinkGD normalization must be 'alternating' or 'simultaneous'")
        steps = group["sinkhorn_steps"]
        if (
            not isinstance(steps, int)
            or isinstance(steps, bool)
            or steps < 1
            or (mode == "alternating" and steps % 2 == 0)
        ):
            required = "positive odd" if mode == "alternating" else "positive"
            raise ValueError(f"SinkGD sinkhorn_steps must be a {required} number of passes")
        if group["start_axis"] not in ("row", "col"):
            raise ValueError("SinkGD start_axis must be 'row' or 'col'")
        if group.get("final_row_pass") and mode != "simultaneous":
            raise ValueError("SinkGD final_row_pass only applies to simultaneous normalization")
        for parameter in group["params"]:
            if parameter.ndim != 2 or parameter.numel() == 0 or not parameter.is_floating_point():
                raise ValueError("SinkGD requires nonempty real 2D parameters")
            cls._shard_group(parameter)

    def ensure_state_initialized(self) -> None:
        """Allocate momentum without applying updates, including before checkpoint loading."""
        for group in self.param_groups:
            self._validate_group(group)
            for parameter in group["params"]:
                if "momentum_buffer" not in self.state[parameter]:
                    dtype = torch.float64 if parameter.dtype == torch.float64 else torch.float32
                    self.state[parameter]["momentum_buffer"] = torch.zeros_like(
                        parameter, dtype=dtype
                    )

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            self._validate_group(group)
            beta, eps = group["momentum"], group["eps"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.layout != torch.strided:
                    raise RuntimeError("SinkGD requires dense gradients")
                if isinstance(parameter, DTensor) and (
                    not isinstance(gradient, DTensor)
                    or gradient.device_mesh != parameter.device_mesh
                    or gradient.placements != parameter.placements
                ):
                    raise ValueError(
                        "SinkGD gradient and parameter must share mesh and placements"
                    )
                shard_group = self._shard_group(parameter)
                dtype = torch.float64 if parameter.dtype == torch.float64 else torch.float32
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(parameter, dtype=dtype)
                buffer = _local(state["momentum_buffer"])
                local_gradient = _local(gradient).to(dtype=dtype)
                buffer.mul_(beta).add_(local_gradient, alpha=1 - beta)
                update = (
                    buffer.mul(beta).add_(local_gradient, alpha=1 - beta)
                    if group["nesterov"]
                    else buffer.clone()
                )
                row_norms = _norm(update, 1, shard_group)
                row_sum = row_norms.sum()
                if shard_group is not None:
                    dist.all_reduce(row_sum, group=shard_group)
                threshold = group["row_filter_threshold"] * row_sum / parameter.shape[0]
                update.masked_fill_(row_norms <= threshold, 0)
                first_dim, second_dim = (1, 0) if group["start_axis"] == "row" else (0, 1)
                if group.get("normalization", "alternating") == "simultaneous":
                    for _ in range(group["sinkhorn_steps"]):
                        rows = _norm(update, 1, shard_group).add_(eps)
                        columns = _norm(update, 0, shard_group).add_(eps)
                        update.div_(rows).div_(columns)
                    if group.get("final_row_pass"):
                        update.div_(_norm(update, 1, shard_group).add_(eps))
                    scale_dim = 1
                else:
                    for index in range(group["sinkhorn_steps"]):
                        dim = first_dim if index % 2 == 0 else second_dim
                        update.div_(_norm(update, dim, shard_group).add_(eps))
                    scale_dim = first_dim
                update.mul_(math.sqrt(parameter.shape[scale_dim]))
                if group["constant_norm"]:
                    update.mul_(
                        math.sqrt(parameter.numel()) / (_norm(update, None, shard_group) + eps)
                    )
                update = update.to(dtype=parameter.dtype)
                if isinstance(parameter, DTensor):
                    update = DTensor.from_local(
                        update,
                        device_mesh=parameter.device_mesh,
                        placements=parameter.placements,
                        shape=parameter.shape,
                        stride=parameter.stride(),
                        run_check=False,
                    )
                if group["weight_decay"]:
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"] * group["alpha"])
        return loss

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        # Restore from the saved values, not the potentially downcast Optimizer copy.
        for group, saved_group in zip(self.param_groups, state_dict["param_groups"], strict=True):
            self._validate_group(group)
            for parameter, saved_id in zip(group["params"], saved_group["params"], strict=True):
                saved = state_dict["state"].get(saved_id, {})
                if "momentum_buffer" in saved:
                    dtype = torch.float64 if parameter.dtype == torch.float64 else torch.float32
                    self.state[parameter]["momentum_buffer"] = (
                        saved["momentum_buffer"].to(device=parameter.device, dtype=dtype).clone()
                    )
