from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class WarmupLR(LRScheduler):
    """Base class for learning rate schedulers."""

    def __init__(self, optimizer: Optimizer, warmup_iters: int, last_epoch: int = -1) -> None:
        self._warmup_iters = warmup_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float | Tensor]:
        """Compute the learning rate."""
        return [base_lr * self.last_epoch / (self._warmup_iters + 1) for base_lr in self.base_lrs]
