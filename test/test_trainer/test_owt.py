from pathlib import Path

import numpy as np
import pytest
import torch

from trainer.datasets._owt import EvalOpenWebText, TrainOpenWebText, get_batch
from trainer.utils.randomness import generator_from


@pytest.mark.parametrize("split", ["train", "val"])
def test_get_batch_preserves_legacy_and_positional_generator_calls(
    tmp_path: Path, split: str
) -> None:
    np.arange(128, dtype=np.uint16).tofile(tmp_path / f"{split}.bin")
    expected = get_batch(split, tmp_path, 3, 8, generator=torch.Generator().manual_seed(17))
    actual = [
        get_batch(split, tmp_path, 3, 8, "cuda", torch.Generator().manual_seed(17)),
        get_batch(
            split, tmp_path, 3, 8, device="cpu", generator=torch.Generator().manual_seed(17)
        ),
        get_batch(split, tmp_path, 3, 8, torch.Generator().manual_seed(17)),
    ]
    for batch in actual:
        for tensor, reference in zip(batch, expected, strict=True):
            assert tensor.device.type == "cpu"
            torch.testing.assert_close(tensor, reference, rtol=0, atol=0)
        torch.testing.assert_close(batch[1], batch[0] + 1)

    with pytest.raises(TypeError, match="either positionally or by keyword"):
        get_batch(split, tmp_path, 3, 8, torch.Generator(), torch.Generator())


def test_owt_iterators_keep_their_seeded_batch_sequences(tmp_path: Path) -> None:
    for split in ("train", "val"):
        np.arange(128, dtype=np.uint16).tofile(tmp_path / f"{split}.bin")
    train = iter(TrainOpenWebText(tmp_path, batch_size=3, max_length=8, seed=17, worker_id=1))
    generator = generator_from("train", 17, 1, device="cpu")
    for _ in range(3):
        expected = get_batch("train", tmp_path, 3, 8, "cpu", generator)
        torch.testing.assert_close(next(train), expected, rtol=0, atol=0)
    validation = EvalOpenWebText(tmp_path, batch_size=3, eval_size=2, max_length=8)
    for index in range(len(validation)):
        expected = get_batch(
            "val", tmp_path, 3, 8, "cpu", generator_from("validation", index, device="cpu")
        )
        torch.testing.assert_close(validation[index], expected, rtol=0, atol=0)


def test_train_iterator_skip_resumes_the_batch_sequence(tmp_path: Path) -> None:
    np.arange(128, dtype=np.uint16).tofile(tmp_path / "train.bin")
    full = iter(TrainOpenWebText(tmp_path, batch_size=3, max_length=8, seed=17))
    expected = [next(full) for _ in range(5)][3:]
    resumed = iter(TrainOpenWebText(tmp_path, batch_size=3, max_length=8, seed=17, skip_batches=3))
    for reference in expected:
        torch.testing.assert_close(next(resumed), reference, rtol=0, atol=0)
