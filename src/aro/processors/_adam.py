import math
from typing import NotRequired, TypedDict

import torch
from torch import Tensor

from ._defs import GradientProcessor


class AdamStateDict(TypedDict):
    step: int
    exp_avg: NotRequired[Tensor]
    exp_avg_sq: NotRequired[Tensor]


class AdamProcessor(GradientProcessor):
    """This gradient processor processes the gradient using the Adam's update rule."""

    def __init__(self, betas: tuple[float, float], eps: float, correct_bias: bool):
        """Init method.

        Args:
            betas (tuple[float, float], optional): The betas of Adam optimizer.
            eps (float, optional): The eps added to second moment to avoid divide 0 error.
            correct_bias (bool, optional): Whether to enable correct bias in Adam.
        """
        self._beta1 = betas[0]
        self._beta2 = betas[1]
        self._eps = eps
        self._correct_bias = correct_bias

    def process_grad(self, grad: Tensor, *, state_dict: AdamStateDict) -> Tensor:
        """Process the gradient based on the Adam update rule.

        state_dict is modified in place.

        Args:
            grad: The input gradient tensor.
            state_dict: The state dict of the processor for grad parameter.

        """
        # note that when beta2=0 we perform momentum update
        reg = 1.0
        num = grad
        denom = 1.0
        if self._beta1 > 0.0:
            assert "exp_avg" in state_dict
            state_dict["exp_avg"].mul_(self._beta1).add_(grad, alpha=(1.0 - self._beta1))
            num = state_dict["exp_avg"]
        if self._beta2 > 0.0:
            assert "exp_avg_sq" in state_dict
            exp_avg_sq = state_dict["exp_avg_sq"]
            exp_avg_sq.mul_(self._beta2).addcmul_(grad, grad, value=(1.0 - self._beta2))
            denom = exp_avg_sq.sqrt().add_(self._eps)
        state_dict["step"] += 1
        if self._correct_bias:
            bias_correction1 = 1.0 - self._beta1 ** (state_dict["step"] + 1)
            bias_correction2 = 1.0 - self._beta2 ** (state_dict["step"] + 1)
            reg = math.sqrt(bias_correction2) / bias_correction1

        return reg * num / denom

    def init_state_dict(self, grad: Tensor) -> AdamStateDict:
        """Initialize the state dict of the optimizer.

        Returns:
            dict[str, Any]: The state dict.
        """
        d: AdamStateDict = {"step": 0}
        if self._beta1 > 0.0:
            d["exp_avg"] = torch.zeros_like(grad)
        if self._beta2 > 0.0:
            d["exp_avg_sq"] = torch.zeros_like(grad)

        return d
