import copy
import math
import os
from collections.abc import Generator
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.nn.parallel import DistributedDataParallel

from aro.optim import SinkGD


def _reference(
    gradient: np.ndarray[Any, np.dtype[np.float64]],
    momentum: np.ndarray[Any, np.dtype[np.float64]],
    options: dict[str, Any],
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
    """Independent NumPy form of the source branch's complete update."""
    beta, eps = options["momentum"], options["eps"]
    momentum = beta * momentum + (1 - beta) * gradient
    update = beta * momentum + (1 - beta) * gradient if options["nesterov"] else momentum.copy()
    rows = np.linalg.norm(update, axis=1, keepdims=True)
    update[(rows <= options["row_filter_threshold"] * rows.mean()).ravel()] = 0
    if options["normalization"] == "simultaneous":
        for _ in range(options["sinkhorn_steps"]):
            rows = np.linalg.norm(update, axis=1, keepdims=True) + eps
            columns = np.linalg.norm(update, axis=0, keepdims=True) + eps
            update = update / rows / columns
        if options["final_row_pass"]:
            update /= np.linalg.norm(update, axis=1, keepdims=True) + eps
        scale = math.sqrt(update.shape[1])
    else:
        first = 1 if options["start_axis"] == "row" else 0
        for index in range(options["sinkhorn_steps"]):
            axis = first if index % 2 == 0 else 1 - first
            update /= np.linalg.norm(update, axis=axis, keepdims=True) + eps
        scale = math.sqrt(update.shape[first])
    update *= scale
    if options["constant_norm"]:
        update *= math.sqrt(update.size) / (np.linalg.norm(update) + eps)
    return update, momentum


def test_defaults_match_source_branch() -> None:
    optimizer = SinkGD([torch.nn.Parameter(torch.ones(5, 3))])
    assert optimizer.defaults == {
        "lr": 1e-3,
        "momentum": 0.95,
        "nesterov": True,
        "sinkhorn_steps": 11,
        "eps": 1e-20,
        "row_filter_threshold": 1e-3,
        "alpha": 0.2,
        "constant_norm": True,
        "weight_decay": 0.0,
        "start_axis": "row",
        "normalization": "alternating",
        "final_row_pass": False,
    }
    assert not optimizer.state


@pytest.mark.parametrize("constant_norm", [False, True])
@pytest.mark.parametrize("nesterov", [False, True])
@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"sinkhorn_steps": 1},
        {"sinkhorn_steps": 3, "start_axis": "col"},
        {"normalization": "simultaneous", "sinkhorn_steps": 1},
        {"normalization": "simultaneous", "sinkhorn_steps": 2},
        {"normalization": "simultaneous", "sinkhorn_steps": 5},
        {"normalization": "simultaneous", "sinkhorn_steps": 5, "final_row_pass": True},
    ],
)
def test_matches_original_algorithm(
    constant_norm: bool, nesterov: bool, settings: dict[str, Any]
) -> None:
    initial = np.arange(15, dtype=np.float64).reshape(5, 3) / 10
    parameter = torch.nn.Parameter(torch.tensor(initial))
    optimizer = SinkGD(
        [parameter],
        lr=0.03,
        weight_decay=0.2,
        constant_norm=constant_norm,
        nesterov=nesterov,
        **settings,
    )
    expected = initial.copy()
    momentum = np.zeros_like(initial)
    gradient = np.array([[1, 2, 3], [-2, 4, -1], [1e-9, 0, 0], [0, 0, 0], [8, 6, -9]])
    for multiplier in [1, -0.3, 0, 2]:
        actual_gradient = gradient * multiplier
        parameter.grad = torch.tensor(actual_gradient)
        update, momentum = _reference(actual_gradient, momentum, optimizer.defaults)
        expected *= 1 - 0.03 * 0.2
        expected -= 0.03 * optimizer.defaults["alpha"] * update
        optimizer.step()
        torch.testing.assert_close(parameter, torch.tensor(expected), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(parameter.grad, torch.tensor(actual_gradient), rtol=0, atol=0)
        assert set(optimizer.state[parameter]) == {"momentum_buffer"}
        torch.testing.assert_close(
            optimizer.state[parameter]["momentum_buffer"], torch.tensor(momentum)
        )


@pytest.mark.parametrize("constant_norm", [False, True])
def test_global_filter_and_rms_include_masked_rows(constant_norm: bool) -> None:
    parameter = torch.nn.Parameter(torch.zeros(4, 2, dtype=torch.float64))
    parameter.grad = torch.tensor([[3, 0], [1, 0], [0, 0], [0, 0]], dtype=torch.float64)
    optimizer = SinkGD(
        [parameter], lr=1, momentum=0, row_filter_threshold=1, constant_norm=constant_norm
    )
    optimizer.step()
    assert torch.count_nonzero(parameter[1:]) == 0
    expected_rms = 0.2 if constant_norm else 0.1
    assert parameter.square().mean().sqrt().item() == pytest.approx(expected_rms)
    assert optimizer.state[parameter]["momentum_buffer"][1, 0].item() == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16, torch.float64])
@pytest.mark.parametrize("mode", ["alternating", "simultaneous"])
def test_momentum_state_roundtrip_and_zero_gradients(
    dtype: torch.dtype, mode: Literal["alternating", "simultaneous"]
) -> None:
    torch.manual_seed(123)
    parameter = torch.nn.Parameter(torch.randn(5, 3, dtype=dtype))
    missing = torch.nn.Parameter(torch.ones(5, 3, dtype=dtype))
    optimizer = SinkGD([parameter, missing], normalization=mode, weight_decay=0.2)
    parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    assert torch.isfinite(parameter).all()
    assert missing not in optimizer.state
    for _ in range(2):
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()
    saved = copy.deepcopy(optimizer.state_dict())
    other = torch.nn.Parameter(parameter.detach().clone())
    other_missing = torch.nn.Parameter(missing.detach().clone())
    restored = SinkGD([other, other_missing])
    restored.load_state_dict(saved)
    buffer = restored.state[other]["momentum_buffer"]
    assert buffer.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    torch.testing.assert_close(
        buffer, optimizer.state[parameter]["momentum_buffer"], rtol=0, atol=0
    )
    for _ in range(2):
        parameter.grad = torch.randn_like(parameter)
        other.grad = parameter.grad.clone()
        optimizer.step()
        restored.step()
        torch.testing.assert_close(parameter, other, rtol=0, atol=0)


def test_closure_groups_and_initialization() -> None:
    first = torch.nn.Parameter(torch.ones(3, 2))
    second = torch.nn.Parameter(torch.ones(3, 2))
    optimizer = SinkGD([{"params": [first], "lr": 0}, {"params": [second], "lr": 0.1}])
    optimizer.ensure_state_initialized()
    optimizer.ensure_state_initialized()
    assert all(set(state) == {"momentum_buffer"} for state in optimizer.state.values())

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = first.square().sum() + second.square().sum()
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    assert loss is not None and loss.item() == 12
    torch.testing.assert_close(first, torch.ones_like(first))
    assert not torch.equal(first, second)


@pytest.mark.parametrize(
    "options",
    [
        {"lr": -1},
        {"lr": float("nan")},
        {"momentum": 1},
        {"eps": 0},
        {"alpha": float("inf")},
        {"weight_decay": -1},
        {"row_filter_threshold": -1},
        {"sinkhorn_steps": 0},
        {"sinkhorn_steps": 2},
        {"sinkhorn_steps": True},
        {"normalization": "unknown"},
        {"start_axis": "unknown"},
        {"final_row_pass": True},
    ],
)
def test_rejects_invalid_hyperparameters(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="SinkGD"):
        SinkGD([torch.nn.Parameter(torch.ones(3, 2))], **options)


@pytest.mark.parametrize("shape", [(), (3,), (2, 3, 4), (0, 3)])
def test_rejects_nonmatrices_or_empty_global_shapes(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="real 2D"):
        SinkGD([torch.nn.Parameter(torch.ones(shape))])


def test_rejects_sparse_gradients() -> None:
    parameter = torch.nn.Parameter(torch.ones(3, 2))
    parameter.grad = torch.ones_like(parameter).to_sparse()
    optimizer = SinkGD([parameter])
    with pytest.raises(RuntimeError, match="dense gradients"):
        optimizer.step()


def test_legacy_checkpoint_defaults_to_alternating() -> None:
    parameter = torch.nn.Parameter(torch.ones(3, 2))
    original = SinkGD([parameter])
    parameter.grad = torch.randn_like(parameter)
    original.step()
    saved = copy.deepcopy(original.state_dict())
    del saved["param_groups"][0]["normalization"]
    del saved["param_groups"][0]["final_row_pass"]
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = SinkGD([restored_parameter], normalization="simultaneous", sinkhorn_steps=5)
    restored.load_state_dict(saved)
    parameter.grad = torch.randn_like(parameter)
    restored_parameter.grad = parameter.grad.clone()
    original.step()
    restored.step()
    torch.testing.assert_close(parameter, restored_parameter, rtol=0, atol=0)


@contextmanager
def _only_normalization_collectives(mesh: DeviceMesh, columns: int) -> Generator[list[int]]:
    reductions: list[int] = []
    original_reduce = dist.all_reduce

    def reduce_stats(tensor: Tensor, **options: Any) -> Any:
        assert options["group"] == mesh.get_group()
        assert tensor.numel() in (1, columns)
        reductions.append(tensor.numel())
        return original_reduce(tensor, **options)

    with (
        patch("aro.optim._sinkgd.dist.all_reduce", reduce_stats),
        patch.object(
            DTensor, "full_tensor", side_effect=AssertionError("No full gathers allowed")
        ),
    ):
        yield reductions


def _distributed_parity(mesh: DeviceMesh, device: torch.device) -> None:
    for shape in [(10, 8), (8, 5), (2, 7)]:
        for mode, passes, extra in [
            ("alternating", 11, {}),
            ("alternating", 3, {"start_axis": "col", "nesterov": False}),
            ("simultaneous", 2, {}),
            ("simultaneous", 5, {"final_row_pass": True}),
        ]:
            for constant_norm in [False, True]:
                torch.manual_seed(13)
                value = torch.randn(shape, dtype=torch.float64, device=device)
                parameter = torch.nn.Parameter(distribute_tensor(value.clone(), mesh, [Shard(0)]))
                assert isinstance(parameter, DTensor)
                reference = torch.nn.Parameter(value.clone())
                kwargs = {
                    "lr": 0.03,
                    "weight_decay": 0.2,
                    "normalization": mode,
                    "sinkhorn_steps": passes,
                    "constant_norm": constant_norm,
                    **extra,
                }
                distributed = SinkGD([parameter], **kwargs)
                dense = SinkGD([reference], **kwargs)
                for step in range(3):
                    gradient = torch.randn_like(value)
                    if step == 0:
                        gradient[0].zero_()
                    if step == 1:
                        gradient.zero_()
                    parameter.grad = distribute_tensor(gradient.clone(), mesh, [Shard(0)])
                    reference.grad = gradient.clone()
                    with _only_normalization_collectives(mesh, shape[1]) as reductions:
                        distributed.step()
                    dense.step()
                    expected_reductions = (
                        1
                        + (
                            passes
                            if mode == "simultaneous"
                            else passes // 2 + int(extra.get("start_axis") == "col")
                        )
                        + int(constant_norm)
                    )
                    assert len(reductions) == expected_reductions
                    torch.testing.assert_close(
                        parameter.full_tensor(), reference, rtol=1e-10, atol=1e-10
                    )
                    momentum = distributed.state[parameter]["momentum_buffer"]
                    assert isinstance(momentum, DTensor)
                    assert momentum.to_local().shape == parameter.to_local().shape
                    torch.testing.assert_close(
                        momentum.full_tensor(), dense.state[reference]["momentum_buffer"]
                    )
                    torch.testing.assert_close(
                        parameter.grad.full_tensor(), gradient, rtol=0, atol=0
                    )
                saved = copy.deepcopy(distributed.state_dict())
                restored = SinkGD([parameter])
                restored.load_state_dict(saved)
                torch.testing.assert_close(
                    restored.state[parameter]["momentum_buffer"].to_local(),
                    distributed.state[parameter]["momentum_buffer"].to_local(),
                    rtol=0,
                    atol=0,
                )
                parameter.grad = None
                with patch("aro.optim._sinkgd.dist.all_reduce", side_effect=AssertionError):
                    distributed.step()


def _low_precision_parity(mesh: DeviceMesh, device: torch.device) -> None:
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        for mode in ("alternating", "simultaneous"):
            torch.manual_seed(29)
            value = torch.randn(7, 5, device=device, dtype=dtype)
            parameter = torch.nn.Parameter(distribute_tensor(value.clone(), mesh, [Shard(0)]))
            assert isinstance(parameter, DTensor)
            reference = torch.nn.Parameter(value.clone())
            optimizer = SinkGD([parameter], lr=0.2, normalization=mode, sinkhorn_steps=5)
            dense = SinkGD([reference], lr=0.2, normalization=mode, sinkhorn_steps=5)
            for _ in range(3):
                gradient = torch.randn_like(value)
                parameter.grad = distribute_tensor(gradient.clone(), mesh, [Shard(0)])
                reference.grad = gradient.clone()
                optimizer.step()
                dense.step()
                torch.testing.assert_close(
                    parameter.full_tensor(), reference, rtol=1e-2, atol=1e-3
                )
                state = optimizer.state[parameter]["momentum_buffer"]
                assert state.dtype == torch.float32
                torch.testing.assert_close(
                    state.full_tensor(), dense.state[reference]["momentum_buffer"], rtol=0, atol=0
                )
                saved = copy.deepcopy(optimizer.state_dict())
                optimizer = SinkGD([parameter])
                optimizer.load_state_dict(saved)
                assert optimizer.state[parameter]["momentum_buffer"].dtype == torch.float32


def _model_parity(device: torch.device, *, sharded: bool) -> None:
    torch.manual_seed(71)
    model: torch.nn.Module = torch.nn.Sequential(
        torch.nn.Linear(7, 5, bias=False), torch.nn.Tanh(), torch.nn.Linear(5, 3, bias=False)
    ).to(device)
    reference = copy.deepcopy(model)
    if sharded:
        fully_shard(model)
    else:
        model = DistributedDataParallel(model)
    distributed = SinkGD(
        model.parameters(), lr=0.01, normalization="simultaneous", sinkhorn_steps=5
    )
    dense = SinkGD(reference.parameters(), lr=0.01, normalization="simultaneous", sinkhorn_steps=5)
    for _ in range(3):
        x = torch.randn(12, 7, device=device)
        y = torch.randn(12, 3, device=device)
        rank = dist.get_rank()
        distributed.zero_grad()
        dense.zero_grad()
        (
            model(x[rank * 4 : (rank + 1) * 4]) - y[rank * 4 : (rank + 1) * 4]
        ).square().mean().backward()
        (reference(x) - y).square().mean().backward()
        distributed.step()
        dense.step()
        for parameter, expected in zip(model.parameters(), reference.parameters(), strict=True):
            actual = parameter.full_tensor() if isinstance(parameter, DTensor) else parameter
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def _distributed_worker(rank: int, world_size: int, init_method: str, cuda: bool) -> None:
    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group(
        "nccl" if cuda else "gloo",
        rank=rank,
        world_size=world_size,
        init_method=init_method,
        timeout=timedelta(seconds=60),
        device_id=device if cuda else None,
    )
    try:
        mesh = init_device_mesh(device.type, (world_size,))
        _distributed_parity(mesh, device)
        _low_precision_parity(mesh, device)
        group = dist.new_group([1, 2])
        if rank in [1, 2]:
            assert isinstance(group, dist.ProcessGroup)
            _distributed_parity(DeviceMesh.from_group(group, device.type), device)
        dist.barrier()
        replicated = torch.nn.Parameter(
            distribute_tensor(torch.ones(5, 3, device=device), mesh, [Replicate()])
        )
        replicated.grad = torch.ones_like(replicated)
        with patch(
            "aro.optim._sinkgd.dist.all_reduce", side_effect=AssertionError("No replica reduction")
        ):
            SinkGD([replicated]).step()
        invalid = torch.nn.Parameter(
            distribute_tensor(torch.ones(5, 3, device=device), mesh, [Shard(1)])
        )
        with pytest.raises(ValueError, match="Shard"):
            SinkGD([invalid])
        if cuda:
            _model_parity(device, sharded=True)
            _model_parity(device, sharded=False)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("cuda", [False, True], ids=["gloo-cpu", "nccl-fsdp2"])
def test_distributed_updates_match_global_reference(tmp_path: Path, cuda: bool) -> None:
    if cuda and (not dist.is_nccl_available() or torch.cuda.device_count() < 3):
        pytest.skip("requires three CUDA devices and NCCL")
    if not cuda and not dist.is_gloo_available():
        pytest.skip("requires Gloo")
    mp.spawn(
        _distributed_worker, args=(3, f"file://{tmp_path / 'init'}", cuda), nprocs=3, join=True
    )
