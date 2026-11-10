"""FSDP2 shards frozen base weights while FixedRankRGD owns replicated factors."""

import copy
import inspect
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint

from aro.lora import FixedRankRGD, LoRAADBLinear


class _Model(nn.Module):
    def __init__(self, checkpointing: str) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.Sequential(LoRAADBLinear(8, 8, rank=2, bias=True), nn.Tanh()) for _ in range(2)]
        )
        self.head = nn.Linear(8, 4)
        self.checkpointing = checkpointing
        self.requires_grad_(False)
        for layer in self.adapters():
            with torch.no_grad():
                layer.weight.normal_(std=0.2)
                layer.d_raw.copy_(torch.tensor([0.8, 0.3]))

    def adapters(self) -> list[LoRAADBLinear]:
        return [m for m in self.modules() if isinstance(m, LoRAADBLinear)]

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            if self.checkpointing == "none":
                x = block(x)
            else:
                x = checkpoint(block, x, use_reentrant=self.checkpointing == "reentrant")
        return self.head(x)


def _worker(
    rank: int,
    world_size: int,
    init_method: str,
    compute_dtype: torch.dtype,
    checkpointing: str,
    factor_dtype: torch.dtype,
) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        init_method=init_method,
        timeout=timedelta(seconds=120),
    )
    optimizers: list[FixedRankRGD] = []
    try:
        torch.manual_seed(123)
        model = _Model(checkpointing).to(device)
        for layer in model.adapters():
            for factor in (layer.A, layer.d_raw, layer.B):
                factor.data = factor.data.to(factor_dtype)
        reference = copy.deepcopy(model)
        reference.checkpointing = "none"
        factors = {
            factor
            for layer in model.adapters()
            for factor in (layer.A, layer.d_raw, layer.B)
            if isinstance(factor, nn.Parameter)
        }
        policy = MixedPrecisionPolicy(param_dtype=compute_dtype)
        for block in model.blocks:
            fully_shard(
                block, ignored_params=factors, mp_policy=policy, reshard_after_forward=True
            )
        fully_shard(model, ignored_params=factors, mp_policy=policy)

        opt = FixedRankRGD(model.adapters(), lr=0.03, momentum=0.9)
        ref_opt = FixedRankRGD(reference.adapters(), lr=0.03, momentum=0.9)
        optimizers.extend((opt, ref_opt))
        base_weights: list[Tensor] = []
        for layer in model.adapters():
            assert isinstance(layer.weight, DTensor)
            base_weights.append(layer.weight.full_tensor().detach().clone())
            assert layer.weight.to_local().numel() == layer.weight.numel() // world_size
            assert not isinstance(layer.A, DTensor)
            assert not isinstance(layer.B, DTensor)
            assert not isinstance(layer.d_raw, DTensor)
        assert isinstance(model.head.weight, DTensor)

        for step in range(3):
            generator = torch.Generator().manual_seed(100 + step)
            x = torch.randn(12, 8, generator=generator).to(device)
            target = torch.randn(12, 4, generator=generator).to(device)
            opt.zero_grad()
            ref_opt.zero_grad()
            for microbatch in range(2):
                start = (microbatch * world_size + rank) * 3
                local_x = x[start : start + 3].detach().requires_grad_(True)
                with torch.autocast(
                    "cuda", dtype=compute_dtype, enabled=compute_dtype != torch.float32
                ):
                    output = model(local_x)
                for layer in model.adapters():
                    assert isinstance(layer.weight, DTensor)
                    assert layer.weight.to_local().numel() == layer.weight.numel() // world_size
                loss = (output.float() - target[start : start + 3]).square().mean() / 2
                loss.backward()
                for layer in model.adapters():
                    assert isinstance(layer.weight, DTensor)
                    assert layer.weight.to_local().numel() == layer.weight.numel() // world_size
            with torch.autocast(
                "cuda", dtype=compute_dtype, enabled=compute_dtype != torch.float32
            ):
                ref_output = reference(x.detach().requires_grad_(True))
            (ref_output.float() - target).square().mean().backward()
            opt.step()
            ref_opt.step()

            tolerance = 2e-5 if compute_dtype == torch.float32 else 2e-3
            for layer, ref_layer, base in zip(
                model.adapters(), reference.adapters(), base_weights, strict=True
            ):
                torch.testing.assert_close(
                    layer._adapter(), ref_layer._adapter(), atol=tolerance, rtol=tolerance
                )
                assert isinstance(layer.weight, DTensor)
                torch.testing.assert_close(layer.weight.full_tensor(), base, atol=0, rtol=0)
                for factor in (layer.A, layer.d_raw, layer.B):
                    replicas = [torch.empty_like(factor) for _ in range(world_size)]
                    dist.all_gather(replicas, factor.detach())
                    for replica in replicas:
                        torch.testing.assert_close(replica, factor, atol=0, rtol=0)
                assert all(
                    value.dtype == torch.float32
                    for key in ("master", "momentum")
                    for value in opt.state[layer.A][key]
                )
                velocity_norm = (
                    torch.stack([value.square().sum() for value in opt.state[layer.A]["momentum"]])
                    .sum()
                    .sqrt()
                )
                ref_velocity_norm = (
                    torch.stack(
                        [value.square().sum() for value in ref_opt.state[ref_layer.A]["momentum"]]
                    )
                    .sum()
                    .sqrt()
                )
                torch.testing.assert_close(
                    velocity_norm,
                    ref_velocity_norm,
                    atol=1e-7,
                    rtol=2e-4 if compute_dtype == torch.float32 else 2e-2,
                )
            if step == 0:
                saved = copy.deepcopy(opt.state_dict())
                opt.remove_hooks()
                opt = FixedRankRGD(model.adapters(), lr=0.03, momentum=0.9)
                optimizers.append(opt)
                opt.load_state_dict(saved)
    finally:
        for optimizer in optimizers:
            optimizer.remove_hooks()
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_nccl_available()
    or torch.cuda.device_count() < 2
    or "ignored_params" not in inspect.signature(fully_shard).parameters,
    reason="requires two CUDA devices and FSDP2 ignored_params",
)
@pytest.mark.parametrize(
    ("compute_dtype", "checkpointing", "factor_dtype"),
    [
        (torch.float32, "none", torch.float32),
        (torch.bfloat16, "none", torch.float32),
        (torch.bfloat16, "none", torch.bfloat16),
        (torch.bfloat16, "reentrant", torch.float32),
        (torch.bfloat16, "nonreentrant", torch.float32),
    ],
)
def test_fsdp_matches_global_batch(
    tmp_path: Path,
    compute_dtype: torch.dtype,
    checkpointing: str,
    factor_dtype: torch.dtype,
) -> None:
    mp.spawn(
        _worker,
        args=(2, f"file://{tmp_path / 'init'}", compute_dtype, checkpointing, factor_dtype),
        nprocs=2,
        join=True,
    )
