import os
from pathlib import Path

import torch.distributed as dist

LATEST = "latest"


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Returns the checkpoint referenced by `<checkpoint_dir>/latest`, or None if there is none."""
    pointer = checkpoint_dir / LATEST
    if not pointer.is_file():
        return None
    checkpoint = checkpoint_dir / pointer.read_text().strip()
    if not checkpoint.exists():
        raise FileNotFoundError(f"{pointer} refers to missing checkpoint {checkpoint}")
    return checkpoint


def mark_latest(checkpoint_dir: Path, checkpoint: Path) -> None:
    """Points `<checkpoint_dir>/latest` at `checkpoint` once every rank has finished writing it.

    Returns on every rank only after the pointer is updated, so all ranks can read it afterwards.
    """
    if dist.is_initialized():
        dist.barrier()
    if int(os.getenv("RANK", "0")) == 0:
        tmp = checkpoint_dir / f"{LATEST}.tmp"
        tmp.write_text(checkpoint.name)
        tmp.replace(checkpoint_dir / LATEST)  # atomic: never leaves a partial pointer
    if dist.is_initialized():
        dist.barrier()
