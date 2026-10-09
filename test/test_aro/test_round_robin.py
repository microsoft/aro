import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from aro.optim import AroSinkhorn


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_optimizer_execution_path(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    parameter = torch.nn.Parameter(torch.randn(16, 8, device=device))
    optimizer = AroSinkhorn([parameter])
    assert (optimizer._comm_stream is not None) == (device == "cuda")
    assert (optimizer._comp_stream is not None) == (device == "cuda")
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    assert optimizer.state[parameter]["step"] == 1
    assert torch.isfinite(parameter).all()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires four CUDA GPUs")
def test_four_rank_round_robin() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=4",
            str(Path(__file__).resolve()),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("round-robin PASS") == 4


def _run_distributed_check() -> None:
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=rank, timeout=timedelta(seconds=90))
    try:
        generator = torch.Generator().manual_seed(1337)
        initial = [torch.randn(16 + i, 8 + i, generator=generator) for i in range(9)]
        parameters = [torch.nn.Parameter(value.to(rank)) for value in initial]
        reference = [torch.nn.Parameter(value.clone()) for value in initial]
        optimizer = AroSinkhorn(
            [{"params": parameters[:5]}, {"params": parameters[5:]}],
            weight_decay=0.1,
            update_freq=2,
        )
        reference_optimizer = AroSinkhorn(reference, weight_decay=0.1, update_freq=2)
        expected_shapes = [
            parameter.shape
            for group in (parameters[:5], parameters[5:])
            for index, parameter in enumerate(group)
            if index % 4 == rank
        ]
        assert optimizer._comm_stream is not None
        assert optimizer._comp_stream is not None
        with patch.object(
            optimizer._rot_processor,
            "process_grad",
            wraps=optimizer._rot_processor.process_grad,
        ) as rotate:
            for _ in range(3):
                for parameter, reference_parameter in zip(parameters, reference, strict=True):
                    gradient = torch.randn(parameter.shape, generator=generator)
                    parameter.grad = gradient.to(rank)
                    reference_parameter.grad = gradient
                rotate.reset_mock()
                optimizer.step(prefetch_factor=1)
                reference_optimizer.step()
                torch.cuda.synchronize()
                assert [call.args[0].shape for call in rotate.call_args_list] == expected_shapes
                for parameter, reference_parameter in zip(parameters, reference, strict=True):
                    torch.testing.assert_close(
                        parameter.cpu(), reference_parameter, rtol=1e-4, atol=1e-5
                    )
                    rank_zero = parameter.detach().clone()
                    dist.broadcast(rank_zero, src=0)
                    torch.testing.assert_close(parameter, rank_zero, rtol=0, atol=0)
                for group in optimizer.param_groups:
                    for index, parameter in enumerate(group["params"]):
                        assert (optimizer.state[parameter]["u"] is not None) == (index % 4 == rank)
        print(f"rank {rank}: round-robin PASS", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    _run_distributed_check()
