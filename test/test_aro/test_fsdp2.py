from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from time import monotonic
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from aro.optim import AroAdam, AroSign, AroSinkhorn, Muon
from aro.optim._rot import AroBase, _build_u_identity, _rowsplit_sizes
from aro.processors import ConstantNormProcessor
from aro.processors.u import IdentityUpdate

type OptimizerType = type[AroAdam] | type[AroSign] | type[AroSinkhorn] | type[Muon]


def _spawn_workers(worker: Callable[..., None], world_size: int, args: tuple[object, ...]) -> None:
    context = mp.spawn(worker, args=args, nprocs=world_size, join=False)
    assert context is not None
    deadline = monotonic() + 90
    try:
        while not context.join(timeout=1):
            if monotonic() >= deadline:
                pytest.fail("Distributed optimizer regression exceeded its 90-second deadline")
    finally:
        # A failing rank may leave its peers inside mismatched collectives.
        processes = [process for process in context.processes if process is not None]
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)


@pytest.mark.parametrize("rows", [0, 1, 2, 5, 6, 8, 10, 13])
@pytest.mark.parametrize("world_size", [1, 2, 3, 4, 8])
def test_row_splits_match_dtensor_chunking(rows: int, world_size: int) -> None:
    chunks = torch.empty(rows).chunk(world_size)
    expected = [chunk.shape[0] for chunk in chunks]
    expected.extend([0] * (world_size - len(expected)))
    assert _rowsplit_sizes(rows, world_size) == expected


def _check_identity(mesh: DeviceMesh, shape: tuple[int, int]) -> None:
    parameter = distribute_tensor(torch.zeros(shape), mesh, [Shard(0)])
    rotation = _build_u_identity(parameter, owner=False)
    assert isinstance(rotation, DTensor)
    size = min(shape)
    expected_local_rows = _rowsplit_sizes(size, mesh.size())[mesh.get_local_rank()]
    assert rotation.to_local().shape == (expected_local_rows, size)
    torch.testing.assert_close(rotation.full_tensor(), torch.eye(size), rtol=0, atol=0)


def _identity_worker(rank: int, world_size: int, init_method: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=init_method,
        timeout=timedelta(seconds=30),
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,))
        for shape in [(10, 8), (8, 6), (2, 5), (5, 2)]:
            _check_identity(mesh, shape)

        meshes = init_device_mesh("cpu", (2, 2), mesh_dim_names=("replicate", "shard"))
        if rank >= 2:
            _check_identity(meshes["shard"], (8, 5))
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="requires the gloo distributed backend",
)
def test_rotation_identity_has_canonical_shards_and_isolates_submeshes(tmp_path: Path) -> None:
    _spawn_workers(
        _identity_worker,
        4,
        (4, f"file://{tmp_path / 'identity-init'}"),
    )


@torch.no_grad()
def _reference_step(
    optimizer: AroBase,
    parameter: Tensor,
    state: dict[str, Any],
    gradient: Tensor | None,
) -> None:
    """Apply the processors to a full tensor without the distributed optimizer pipeline."""
    if gradient is None:
        return
    group = optimizer.param_groups[0]
    if not state:
        state.update(
            step=0,
            ema=optimizer._ema_processor.init_state_dict(gradient),
            base=optimizer._base_processor.init_state_dict(gradient),
            u=torch.eye(min(parameter.shape), device=parameter.device),
        )
    momentum = optimizer._ema_processor.process_grad(gradient.clone(), state_dict=state["ema"])
    update, state["u"] = optimizer._rot_processor.process_grad(
        momentum,
        state["u"],
        state["step"] % group["update_freq"] == 0,
        state_dict=state["base"],
    )
    if optimizer._norm_processor is not None:
        update = optimizer._norm_processor.process_grad(update)
    parameter.add_(update, alpha=-(group["lr"] * group["alpha"]))
    parameter.add_(parameter, alpha=-(group["lr"] * group["weight_decay"]))
    state["step"] += 1


def _full_tensor(value: Tensor) -> Tensor:
    return value.full_tensor() if isinstance(value, DTensor) else value


def _check_updates(
    optimizer_type: OptimizerType,
    shape: tuple[int, int],
    mesh: DeviceMesh | None,
    *,
    changing_owner: bool = False,
    constant_norm: bool = False,
    prefetch_factor: int = 8,
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator().manual_seed(531)
    world_size = (
        mesh.size()
        if mesh is not None
        else (dist.get_world_size() if dist.is_initialized() else 1)
    )
    count = 2 if changing_owner else (world_size + 1 if dist.is_initialized() else 1)
    expected = [torch.randn(shape, generator=generator).to(device) for _ in range(count)]
    parameters = [
        torch.nn.Parameter(
            distribute_tensor(value.clone(), mesh, [Shard(0)])
            if mesh is not None
            else value.clone()
        )
        for value in expected
    ]
    optimizer: AroBase
    if changing_owner:
        optimizer = AroAdam(
            parameters,
            lr=0.01,
            betas=(0.8, 0.99),
            momentum=0.0,
            constant_norm=False,
            u_processor=IdentityUpdate(),
        )
        masks = [(True, True), (False, True), (True, True), (False, False), (True, True)]
    else:
        optimizer = optimizer_type(
            parameters,
            lr=0.01,
            momentum=0.8,
            constant_norm=constant_norm,
            weight_decay=0.1,
            update_freq=2,
        )
        masks = [(True,) * count] * 3
    states: list[dict[str, Any]] = [{} for _ in range(count)]
    for mask in masks:
        gradients = [
            torch.randn(shape, generator=generator).to(device) if active else None
            for active in mask
        ]
        for reference, state, gradient in zip(expected, states, gradients, strict=True):
            _reference_step(optimizer, reference, state, gradient)
        for parameter, gradient in zip(parameters, gradients, strict=True):
            parameter.grad = (
                distribute_tensor(gradient.clone(), mesh, [Shard(0)])
                if mesh is not None and gradient is not None
                else gradient
            )
        optimizer.step(prefetch_factor=prefetch_factor)
        for index, (parameter, reference, state) in enumerate(
            zip(parameters, expected, states, strict=True)
        ):
            torch.testing.assert_close(_full_tensor(parameter), reference, rtol=1e-5, atol=1e-6)
            actual_state = optimizer.state[parameter]
            assert actual_state["step"] == state["step"]
            if mesh is None and dist.is_initialized() and index % world_size != dist.get_rank():
                continue
            torch.testing.assert_close(
                _full_tensor(actual_state["u"]), state["u"], rtol=1e-5, atol=1e-6
            )
            for key, value in state["base"].items():
                if isinstance(value, Tensor):
                    torch.testing.assert_close(
                        _full_tensor(actual_state["base"][key]), value, rtol=1e-5, atol=1e-6
                    )
                else:
                    assert actual_state["base"][key] == value


def _optimizer_worker(rank: int, world_size: int, init_method: str, case: str) -> None:
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        init_method=init_method,
        timeout=timedelta(seconds=45),
        device_id=torch.device("cuda", rank),
    )
    try:
        mesh = init_device_mesh("cuda", (world_size,))
        if case == "uneven":
            for shape in [(10, 8), (12, 12), (8, 7), (2, 5)]:
                for optimizer_type in [AroAdam, AroSign, AroSinkhorn, Muon]:
                    for constant_norm in [False, True]:
                        _check_updates(optimizer_type, shape, mesh, constant_norm=constant_norm)
        elif case == "owners":
            for prefetch_factor in [0, 8]:
                _check_updates(
                    AroAdam, (6, 6), mesh, changing_owner=True, prefetch_factor=prefetch_factor
                )
        elif case == "ddp":
            for optimizer_type in [AroAdam, AroSign, AroSinkhorn, Muon]:
                for constant_norm in [False, True]:
                    _check_updates(
                        optimizer_type,
                        (12, 12),
                        None,
                        constant_norm=constant_norm,
                        prefetch_factor=0,
                    )
        else:
            assert case == "subgroup"
            group = dist.new_group([1, 2])
            if rank in [1, 2]:
                assert isinstance(group, dist.ProcessGroup)
                submesh = DeviceMesh.from_group(group, "cuda")
                _check_updates(AroAdam, (7, 5), submesh)
                _check_updates(AroAdam, (8, 8), submesh, changing_owner=True, prefetch_factor=0)
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_nccl_available() or torch.cuda.device_count() < 3,
    reason="requires three CUDA devices and the NCCL distributed backend",
)
@pytest.mark.parametrize("case", ["uneven", "owners", "subgroup", "ddp"])
def test_distributed_optimizer_matches_full_tensor_reference(tmp_path: Path, case: str) -> None:
    _spawn_workers(
        _optimizer_worker,
        3,
        (3, f"file://{tmp_path / case}", case),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ARO requires CUDA streams")
@pytest.mark.parametrize("optimizer_type", [AroAdam, AroSign, AroSinkhorn, Muon])
def test_single_device_optimizer_matches_full_tensor_reference(
    optimizer_type: OptimizerType,
) -> None:
    _check_updates(optimizer_type, (5, 3), None)


def _normalization_lifetime_worker(
    rank: int, world_size: int, init_method: str, shape: tuple[int, int]
) -> None:
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        init_method=init_method,
        timeout=timedelta(seconds=45),
        device_id=device,
    )
    try:
        mesh = init_device_mesh("cuda", (world_size,))
        parameters = [
            torch.nn.Parameter(
                distribute_tensor(torch.zeros(shape, device=device), mesh, [Shard(0)])
            )
            for _ in range(world_size)
        ]
        optimizer = AroSign(
            parameters,
            lr=1.0,
            alpha=1.0,
            momentum=0.0,
            constant_norm=True,
            u_processor=IdentityUpdate(),
        )
        for parameter in parameters:
            parameter.grad = torch.ones_like(parameter)
        # Warm kernels first: lazy CUDA module loading can synchronize away the race.
        optimizer.step()
        torch.cuda._sleep(1)
        torch.cuda.synchronize()
        with torch.no_grad():
            for parameter in parameters:
                parameter.zero_()
                parameter.grad = torch.ones_like(parameter)

        normalize = ConstantNormProcessor.process_grad
        allocations: list[Tensor] = []

        def delayed_normalize(self: ConstantNormProcessor, gradient: Tensor) -> Tensor:
            torch.cuda.current_stream().synchronize()
            if rank != 0:
                torch.cuda._sleep(100_000_000)
            return normalize(self, gradient)

        def scatter_with_allocator_pressure(
            tensor: Tensor,
            scatter_list: list[Tensor] | None = None,
            src: int | None = None,
            group: dist.ProcessGroup | None = None,
            async_op: bool = False,
        ) -> dist.Work | None:
            # Earlier parameters scatter before this rank's normalization finishes.
            if rank != 0 and not allocations:
                for _ in range(32):
                    allocation = torch.empty(shape, device=device)
                    allocation.zero_()
                    allocations.append(allocation)
            return dist.scatter(tensor, scatter_list, src=src, group=group, async_op=async_op)

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(ConstantNormProcessor, "process_grad", delayed_normalize)
            monkeypatch.setattr("aro.optim._rot.scatter", scatter_with_allocator_pressure)
            optimizer.step()
        torch.cuda.synchronize()
        updated = [_full_tensor(parameter) for parameter in parameters]
        for parameter in updated:
            torch.testing.assert_close(parameter, -torch.ones_like(parameter))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_nccl_available() or torch.cuda.device_count() < 3,
    reason="requires three CUDA devices and the NCCL distributed backend",
)
@pytest.mark.parametrize("shape", [(96, 48), (128, 64)], ids=["even", "uneven"])
def test_normalization_preserves_cross_stream_input_lifetime(
    tmp_path: Path, shape: tuple[int, int]
) -> None:
    _spawn_workers(
        _normalization_lifetime_worker,
        3,
        (3, f"file://{tmp_path / 'normalization-lifetime'}", shape),
    )
