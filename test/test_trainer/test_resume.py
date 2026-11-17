from pathlib import Path

import pytest
import torch

from aro.optim import AroSinkhorn
from trainer.config import ConfigContainer, ConfigServices
from trainer.models import ModelContainer
from trainer.optim import OptimContainer
from trainer.state import FullState, StateContainer, StateServices, latest_checkpoint, mark_latest


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
