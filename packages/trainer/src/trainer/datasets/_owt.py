import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from trainer.utils.randomness import generator_from

DeviceLikeType = str | torch.device | int

logger = logging.getLogger(__name__)


def log_on_main_worker(message: str, *args: Any, level: int = logging.INFO):
    """Logs a message only on the worker with number (rank) 0."""
    if int(os.getenv("RANK", "0")) == 0:
        logger.log(level, message, *args)


def _random_offsets(
    num_tokens: int, batch_size: int, sequence_len: int, generator: torch.Generator | None
) -> Tensor:
    return torch.randint(num_tokens - sequence_len, (batch_size,), generator=generator)


def get_batch(
    split: str,
    data_dir: Path,
    batch_size: int,
    sequence_len: int,
    device: DeviceLikeType | torch.Generator | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Load a CPU batch, accepting a legacy device argument or positional generator."""
    if isinstance(device, torch.Generator):
        if generator is not None:
            raise TypeError("Pass the generator either positionally or by keyword, not both")
        generator = device
    data_file = "train.bin" if split == "train" else "val.bin"
    data = np.memmap(data_dir / data_file, dtype=np.uint16, mode="r")
    ix = _random_offsets(len(data), batch_size, sequence_len, generator)
    x = torch.stack([torch.from_numpy((data[i : i + sequence_len]).astype(np.int64)) for i in ix])
    y = torch.stack(
        [torch.from_numpy((data[i + 1 : i + 1 + sequence_len]).astype(np.int64)) for i in ix]
    )
    return x, y


class TrainOpenWebText(IterableDataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        path: Path,
        batch_size: int,
        max_length: int,
        seed: int,
        worker_id: int = 0,
        skip_batches: int = 0,
    ) -> None:
        """Infinite stream of random training windows.

        `skip_batches` fast-forwards the sampler past batches already consumed (when resuming)
        without reading their data.
        """
        log_on_main_worker(f"Loading OWT dataset from {path} in train mode")
        self._path = path
        self._max_length = max_length
        self._batch_size = batch_size
        self._seed = seed
        self._worker_id = worker_id
        self._skip_batches = skip_batches

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        path, max_length, batch_size = self._path, self._max_length, self._batch_size
        gen = generator_from("train", self._seed, self._worker_id, device="cpu")
        if self._skip_batches > 0:
            num_tokens = len(np.memmap(path / "train.bin", dtype=np.uint16, mode="r"))
            for _ in range(self._skip_batches):
                _random_offsets(num_tokens, batch_size, max_length, gen)
        return iter(lambda: get_batch("train", path, batch_size, max_length, generator=gen), None)


class EvalOpenWebText(torch.utils.data.Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, path: Path, batch_size: int, eval_size: int, max_length: int = 512) -> None:
        log_on_main_worker(f"Loading OWT dataset from {path} in eval mode")
        self._path = path
        self._max_length = max_length
        self._batch_size = batch_size
        self._eval_size = eval_size  # number of batches, not number of examples

    def __len__(self) -> int:
        return self._eval_size

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        generator = generator_from("validation", index, device="cpu")
        path, batch_size, max_length = self._path, self._batch_size, self._max_length
        return get_batch("val", path, batch_size, max_length, generator=generator)
