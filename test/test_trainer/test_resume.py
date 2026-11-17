import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel

from aro.optim import AroAdam, AroSinkhorn
from trainer.config import ConfigContainer, ConfigServices
from trainer.models import ModelContainer
from trainer.optim import OptimContainer
from trainer.state import (
    FullState,
    ShardedState,
    StateContainer,
    StateServices,
    TrainingState,
    latest_checkpoint,
    mark_latest,
)


def _train_step(model: torch.nn.Linear, optimizer: AroSinkhorn) -> None:
    model(torch.arange(8, dtype=torch.float32).reshape(2, 4)).square().sum().backward()
    optimizer.step()
    optimizer.zero_grad()


def test_full_state_resumes_from_latest_with_optimizer_and_rng_state(tmp_path: Path) -> None:
    assert latest_checkpoint(tmp_path) is None
    torch.manual_seed(3)
    model = torch.nn.Linear(4, 3, bias=False)
    optimizer = AroSinkhorn([model.weight], lr=0.01)
    _train_step(model, optimizer)
    checkpoint = tmp_path / "ckpt-000000001"
    FullState(1, model, [optimizer], []).save_checkpoint(checkpoint)
    mark_latest(tmp_path, checkpoint)
    expected_random = torch.rand(3)

    restored_model = torch.nn.Linear(4, 3, bias=False)
    restored_optimizer = AroSinkhorn([restored_model.weight], lr=0.01)
    restored = FullState(0, restored_model, [restored_optimizer], [])
    latest = latest_checkpoint(tmp_path)
    assert latest is not None and latest == checkpoint
    restored.resume_from_checkpoint_if_exists(latest)

    assert restored.step == 1
    torch.testing.assert_close(torch.rand(3), expected_random, rtol=0, atol=0)
    _train_step(model, optimizer)
    _train_step(restored_model, restored_optimizer)
    torch.testing.assert_close(restored_model.weight, model.weight, rtol=0, atol=0)

    (tmp_path / "latest").write_text("ckpt-missing")
    with pytest.raises(FileNotFoundError):
        latest_checkpoint(tmp_path)


def test_default_checkpoint_dir_identifies_the_run() -> None:
    def checkpoint_dir(*overrides: str) -> Path:
        config = ConfigContainer("config", "nanogpt", list(overrides)).get(
            ConfigServices.BASE_CONFIG
        )
        models = ModelContainer(config)
        state = StateContainer(config, models, OptimContainer(config, models))
        return state.get(StateServices.CHECKPOINT_DIR)

    default = checkpoint_dir()
    assert default.parent == Path("outputs/checkpoints")
    assert default.name.startswith("mixed-arosinkhorn-")
    assert checkpoint_dir("logger=null", "training.log_interval=1") == default
    assert checkpoint_dir("optimizer.lr=1e-3") != default
    assert checkpoint_dir("optimizer=muon") != default
    assert checkpoint_dir("training.checkpoint_path=/tmp/run") == Path("/tmp/run")


# (in_features, out_features) of bias-free layers: five matrices with uneven row counts, so the
# three ranks own different numbers of matrices and FSDP2 shards are uneven.
_LAYERS = [(8, 12), (12, 12), (12, 6), (6, 10), (10, 4)]


def _full(value: Tensor) -> Tensor:
    return value.full_tensor() if isinstance(value, DTensor) else value


def _build(strategy: str, seed: int) -> tuple[nn.Module, AroAdam, TrainingState]:
    torch.manual_seed(seed)
    model: nn.Module = nn.Sequential(*[nn.Linear(i, o, bias=False) for i, o in _LAYERS]).cuda()
    if strategy == "fsdp2":
        fully_shard(model, mesh=init_device_mesh("cuda", (dist.get_world_size(),)))
    else:
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    optimizer = AroAdam(list(model.parameters()), lr=0.05, alpha=0.2, betas=(0.0, 0.95))
    state_type = ShardedState if strategy == "fsdp2" else FullState
    return model, optimizer, state_type(0, model, [optimizer], [])


def _backward(model: nn.Module, index: int) -> None:
    # Each rank trains on different data; DDP/FSDP2 synchronize the gradients in backward.
    generator = torch.Generator().manual_seed(1000 * index + dist.get_rank())
    inputs = torch.randn(4, _LAYERS[0][0], generator=generator).cuda()
    model(inputs).square().mean().backward()


def _step(model: nn.Module, optimizer: AroAdam, index: int) -> None:
    _backward(model, index)
    optimizer.step()
    optimizer.zero_grad()


def _continuation_worker(
    rank: int, world_size: int, init_method: str, strategy: str, run: str
) -> None:
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size))
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        init_method=init_method,
        timeout=timedelta(seconds=45),
        device_id=torch.device("cuda", rank),
    )
    try:
        model, optimizer, state = _build(strategy, seed=0)
        for index in range(2):
            _step(model, optimizer, index)
        state.step = 2
        run_dir = Path(run)
        checkpoint = run_dir / "ckpt-000000002"
        state.save_checkpoint(checkpoint)
        mark_latest(run_dir, checkpoint)

        restored_model, restored_optimizer, restored = _build(strategy, seed=1)
        latest = latest_checkpoint(run_dir)
        assert latest is not None and latest == checkpoint
        restored.resume_from_checkpoint_if_exists(latest)
        assert restored.step == 2
        if strategy == "ddp":
            # A newly constructed DDP wrapper's first gradient all-reduce can round differently
            # (~1e-10) from later ones even for identical local gradients, and ARO's normalized
            # updates amplify that on this tiny rank-deficient problem. One discarded synchronized
            # backward makes both models reduce gradients identically.
            _backward(restored_model, 0)
            restored_optimizer.zero_grad()
        for index in range(2, 4):
            _step(model, optimizer, index)
            _step(restored_model, restored_optimizer, index)
        for expected, actual in zip(model.parameters(), restored_model.parameters(), strict=True):
            torch.testing.assert_close(_full(actual), _full(expected), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_nccl_available() or torch.cuda.device_count() < 3,
    reason="requires three CUDA devices and the NCCL distributed backend",
)
@pytest.mark.parametrize("strategy", ["ddp", "fsdp2"])
def test_distributed_resume_continues_like_uninterrupted_training(
    tmp_path: Path, strategy: str
) -> None:
    # DDP/FSDP2-wrapped training with per-rank data. Under DDP, ARO keeps rotation and Adam
    # base state only on each matrix's owner rank, so the resumed updates match only if every
    # rank's optimizer state is restored.
    world_size = 3
    init_method = f"file://{tmp_path / 'rendezvous'}"
    args = (world_size, init_method, strategy, str(tmp_path / "run"))
    mp.spawn(_continuation_worker, args=args, nprocs=world_size, join=True)
