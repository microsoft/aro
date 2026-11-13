from pathlib import Path

import pytest
import torch

from aro.optim import SinkGD
from trainer.state._sharded import ShardedState


@pytest.mark.parametrize("kind", ["adamw", "sinkgd"])
def test_generator_optimizer_state_matches_reusable_collection(kind: str) -> None:
    model = torch.nn.Linear(4, 3, bias=False)
    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=0.01)
        if kind == "adamw"
        else SinkGD(model.parameters(), lr=0.01)
    )
    if kind == "adamw":
        model(torch.ones(2, 4)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()

    state = ShardedState(3, model, iter([optimizer]), [])
    reference = ShardedState(3, model, [optimizer], [])
    for _ in range(2):
        actual = state.state_dict()
        expected = reference.state_dict()
        assert actual["optimizers"]["param_groups"] == expected["optimizers"]["param_groups"]
        torch.testing.assert_close(
            actual["optimizers"]["state"], expected["optimizers"]["state"], rtol=0, atol=0
        )
        torch.testing.assert_close(actual["model"], expected["model"], rtol=0, atol=0)


def test_generator_optimizer_checkpoint_restores_next_update(tmp_path: Path) -> None:
    torch.manual_seed(17)
    model = torch.nn.Linear(4, 3, bias=False)
    optimizer = SinkGD(model.parameters(), lr=0.01)
    model(torch.ones(2, 4)).square().sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    state = ShardedState(7, model, iter([optimizer]), [])
    checkpoint = tmp_path / "checkpoint"
    state.save_checkpoint(checkpoint)

    restored_model = torch.nn.Linear(4, 3, bias=False)
    restored_optimizer = SinkGD(restored_model.parameters(), lr=0.5)
    restored_state = ShardedState(0, restored_model, iter([restored_optimizer]), [])
    restored_state.resume_from_checkpoint_if_exists(checkpoint)
    assert restored_state.step == state.step
    data = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    for network, opt in ((model, optimizer), (restored_model, restored_optimizer)):
        network(data).square().sum().backward()
        opt.step()
    torch.testing.assert_close(restored_model.weight, model.weight, rtol=0, atol=0)
    torch.testing.assert_close(
        restored_optimizer.state[restored_model.weight]["momentum_buffer"],
        optimizer.state[model.weight]["momentum_buffer"],
        rtol=0,
        atol=0,
    )
