"""This file contains tools for reproducible randomness in PyTorch.

A common problem in machine learning is restoring random number generator state when a job resumes.
This file provides tools that you can use to avoid relying on global random number generator state,
and instead create 'single use' random number generators that are seeded from the arguments you provide.

Seeding from your own arguments gives you a lot of control:
- `rng_seed_based_on(seed, epoch, "data shuffling", worker_rank)` can be used to shuffle data differently across workers.
- `rng_state_from("initialize model", seed)` gives the same random numbers across workers, so they get the same model.
"""

import contextlib
import hashlib
import os
import random
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from torch import Generator

DeviceLikeType = str | torch.device | int


def generator_from(*args: str | bytes | int, device: DeviceLikeType | None = None) -> Generator:
    """Creates a random number generator that's deterministically seeded from the given arguments.

    If you want to use this generator on a GPU, it's recommended to pass that GPU as the `device` argument.
    """
    seed = rng_seed_based_on(*args)
    return Generator(device=device).manual_seed(seed)


def rng_seed_based_on(*args: str | bytes | int) -> int:
    """Creates a deterministic seed for a random number generator based on the given arguments."""
    hashed = hashlib.sha256(usedforsecurity=False)
    for element in args:
        hashed.update(_as_bytes(element))
    checksum = hashed.digest()

    return int.from_bytes(checksum, "big") % torch.iinfo(torch.int32).max


@contextlib.contextmanager
def rng_state_from(*args: str | bytes | int, devices: Iterable[int] | None = None):
    """Temporarily sets the random number generator to a state based on the arguments of this contextmanager.

    If you care about random number generation on a GPU, please set `devices` to the list of GPU
    device IDs you want to control.

    devices (iterable of Device IDs): GPU devices for which to fork
        the RNG. CPU RNG state is always forked. By default, this forks the state
        on all devices, but will emit a warning if your machine has a lot
        of devices, since this function will run very slowly in that case.
        If you explicitly specify devices, this warning will be suppressed.
    """
    seed = rng_seed_based_on(*args)

    with torch.random.fork_rng(devices=devices):
        torch.random.manual_seed(seed)
        yield


def _as_bytes(data: str | bytes | int) -> bytes:
    if isinstance(data, str):
        return data.encode("utf-8")

    if isinstance(data, int):
        return data.to_bytes(8, byteorder="big", signed=True)

    if isinstance(data, bytes):
        return data

    raise TypeError(f"Cannot project data of type {type(data)} to bytes.")


def rng_state_capture() -> dict[str, Any]:
    """Capture Python/NumPy/Torch RNG state for full reproducibility (per-process)."""
    rank = int(os.getenv("RANK", "0"))
    state = {
        "python": random.getstate(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if np is not None:
        try:
            name, keys, pos, has_gauss, cached_gaussian = np.random.get_state()  # noqa: NPY002
            # Builtin types keep checkpoints loadable with torch.load(weights_only=True).
            state["numpy"] = (name, keys.tolist(), pos, has_gauss, cached_gaussian)
        except Exception:
            print(f"[RANK {rank} Warning] Failed to get numpy RNG state")
            state["numpy"] = None
    if torch.cuda.is_available():
        try:
            state["cuda"] = torch.cuda.get_rng_state()  # current device
        except Exception:
            print(f"[RANK {rank} Warning] Failed to get cuda RNG state")
            state["cuda"] = None
    return state


def rng_state_restore(state: dict[str, Any]) -> None:
    """Restore Python/NumPy/Torch RNG state (per-process)."""
    rank = int(os.getenv("RANK", "0"))
    if "python" in state and state["python"] is not None:
        print(f"[RANK {rank}] Restoring python RNG state")
        random.setstate(state["python"])
    else:
        print(f"[RANK {rank} Warning] Failed to restore python RNG state")
    if "torch_cpu" in state and state["torch_cpu"] is not None:
        print(f"[RANK {rank}] Restoring torch_cpu RNG state")
        torch.random.set_rng_state(state["torch_cpu"].cpu())  # checkpoints may map it to GPU
    else:
        print(f"[RANK {rank} Warning] Failed to restore torch_cpu RNG state")
    if np is not None and "numpy" in state and state["numpy"] is not None:
        try:
            print(f"[RANK {rank}] Restoring numpy RNG state")
            np.random.set_state(state["numpy"])  # noqa: NPY002
        except Exception:
            print(f"[RANK {rank} Warning] Failed to restore numpy RNG state")
            pass
    if torch.cuda.is_available() and "cuda" in state and state["cuda"] is not None:
        print(f"[RANK {rank}] Restoring cuda RNG state")
        torch.cuda.set_rng_state(state["cuda"].cpu())  # current device
    else:
        print(f"[RANK {rank} Warning] Failed to set cuda RNG state")
        pass
