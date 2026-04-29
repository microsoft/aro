from typing import Any

import torch
from torch import Tensor
from torch.optim.optimizer import ParamsT

from aro.processors import (
    AdamProcessor,
    ConstantNormProcessor,
    GradientProcessor,
    RotationProcessor,
)
from aro.processors.u import IdentityUpdate, UProcessor

from ._rot import AroBase


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5) -> Tensor:
    """Newton-Schulz iteration adapted from the modded-nanogpt 2025-05-25 record to compute the zeroth power / orthogonalization of G. We opt to use a quintic iteration whose coefficients are selected to maximize the slope at zero.

    For the purpose of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert (
        G.ndim >= 2
    )  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class NewtonSchulzProcessor(GradientProcessor):
    """This implements the Newton-Schulz processor to approximately orthonormalize the input tensor."""

    def __init__(self, ns_steps: int, eps: float) -> None:
        """Initialize the Newton-Schulz processor.

        Args:
            ns_steps (int): the number of NS iteration steps. 5 is a reasonable choice.
            eps (float): a small value to prevent division by zero. 1e-8 is a reasonable choice.
        """
        self._ns_steps = ns_steps
        self._eps = eps

    def process_grad(self, grad: Tensor, *, state_dict: dict[str, Any] | None = None) -> Tensor:
        """The processing function that will orthonormalize the grad.

        Args:
            grad (Tensor): The input gradient or momentum.

        Returns:
            Tensor: The approximately orthonormalized gradient or momentum
        """
        original_dtype = grad.dtype
        x = zeropower_via_newtonschulz5(grad, steps=self._ns_steps)
        return x.to(dtype=original_dtype)


class Muon(AroBase):
    """Muon optimizer with Newton-Schulz iteration. Adapted from Kimi-Moonlight version.

    A spectral descent method with Newton-Schulz algorithm for efficient orthogonalization.

    Args:
        params (ParamsT): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate. Default: 1e-3
        ns_steps (int, optional): Number of Newton-Schulz iterations for matrix square root approximation.
            Default: 5
        eps (float, optional): Small constant for numerical stability. Default: 1e-8
        momentum (float, optional): Momentum coefficient for the EMA processor. Default: 0.95
        weight_decay (float, optional): Weight decay (L2 penalty) coefficient. Default: 0.0
        alpha (float, optional): Scaling factor for update magnitude. Default: 0.2
        update_freq (int | None, optional): Frequency of rotation updates in steps.
            If None, updates on every step. Default: None
        constant_norm (bool, optional): Whether to maintain constant norm of parameters.
            If True, applies ConstantNormProcessor. Default: True
    """

    def __init__(
        self,
        params: ParamsT,
        *,
        lr: float = 1e-3,
        ns_steps: int = 5,
        eps: float = 1e-8,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        alpha: float = 0.2,
        update_freq: int | None = None,
        constant_norm: bool = True,
    ):
        base_processor = NewtonSchulzProcessor(ns_steps, eps)
        u_processor: UProcessor = IdentityUpdate()
        if update_freq is None:
            update_freq = torch.iinfo(torch.int64).max
        ema_processor = AdamProcessor(betas=(momentum, 0.0), eps=eps, correct_bias=True)
        rot_processor = RotationProcessor(base_processor, u_processor)
        norm_processor = ConstantNormProcessor(eps=eps) if constant_norm else None

        super().__init__(
            params,
            base_processor=base_processor,
            ema_processor=ema_processor,
            rot_processor=rot_processor,
            norm_processor=norm_processor,
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            update_freq=update_freq,
            momentum=momentum,
        )
