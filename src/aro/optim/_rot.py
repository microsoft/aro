# type: ignore
import warnings

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT

from aro.processors import (
    AdamProcessor,
    ConstantNormProcessor,
    GradientProcessor,
    RotationProcessor,
)


def params_to_groups(params: ParamsT) -> list[dict[str, Tensor]]:
    """Helper function to create a group of parameters.

    Args:
        params (ParamsT): The input params.

    Returns:
        list[dict[str, Tensor]]: A list of dict that contains the parameter tensors.
    """
    if all(isinstance(p, dict) for p in params):
        return params
    if all(isinstance(p, torch.nn.Parameter) for p in params):
        return [dict(params=params)]
    if all(isinstance(p, list) for p in params):
        return [dict(params=p) for p in params]
    raise ValueError(f"Invalid paramst_to_groups input: {params}")


# identity builder now supports uneven splits
def _build_u_identity(p: Tensor) -> Tensor:
    """Build U = I(rs).

    Global shape: (rs, rs), placement: Shard(0) on a 1D mesh.
    """
    assert p.ndim == 2
    m, n = p.shape
    rs = min(m, n)

    return torch.eye(rs, dtype=p.dtype, device=p.device)


#  main optimizer
class AroBase(Optimizer):
    """Aro Optimizer with overlapped gather/rotate/scatter.

    This efficient implementation can be 10x faster than Aro DDP with large number of GPUs (e.g. 64) and large models (e.g. 1b).
    """

    def __init__(
        self,
        params: ParamsT,
        base_processor: GradientProcessor,
        ema_processor: AdamProcessor,
        rot_processor: RotationProcessor,
        norm_processor: ConstantNormProcessor | None = None,
        *,
        lr: float = 1e-3,
        alpha: float = 0.2,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        update_freq: int = 1,
        momentum: float = 0.95,
    ):
        defaults = dict(
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            update_freq=update_freq,
            momentum=momentum,
        )
        super().__init__(params_to_groups(list(params)), defaults)

        self._base_processor = base_processor
        self._ema_processor = ema_processor
        self._norm_processor = norm_processor
        self._rot_processor = rot_processor

        # Per-parameter persistent state (same keys you already use)
        for pg in self.param_groups:
            for p in pg["params"]:
                assert p.ndim == 2, f"ARO supports 2D only; got {p.shape}"
                self.state[p] = dict(step=0, u=None, ema=None, base=None)

    # state helpers
    def _ensure_ema(self, p: Tensor) -> AdamProcessor:
        st = self.state[p]
        if st["ema"] is None:
            st["ema"] = self._ema_processor.init_state_dict(p.grad)
        return st["ema"]

    def _ensure_base(self, p: Tensor) -> GradientProcessor:
        st = self.state[p]
        if st["base"] is None:
            st["base"] = self._base_processor.init_state_dict(p.grad)
        return st["base"]

    def _ensure_u(self, p: Tensor) -> Tensor:
        st = self.state[p]
        if st["u"] is None:
            U = _build_u_identity(p)
            self.state[p]["u"] = U
        return self.state[p]["u"]

    @torch.no_grad()
    def step(self, closure: None = None, *, prefetch_factor: int = 8):
        for pg in self.param_groups:
            lr: float = pg["lr"]
            alpha: float = pg["alpha"]
            wd: float = pg["weight_decay"]
            update_freq: int = pg["update_freq"]

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                plist: list[Tensor] = [p for p in pg["params"] if p.grad is not None]

            for p in plist:
                st = self.state[p]
                update_sub = (st["step"] % update_freq) == 0
                ema_st = self._ensure_ema(p)  # ema's state dict
                b_st = self._ensure_base(p)  # initialize base processor's state dict
                u_dt = self._ensure_u(p)  # initialize U if needed

                g_dt = self._ema_processor.process_grad(p.grad, state_dict=ema_st)
                gt, ut = self._rot_processor.process_grad(g_dt, u_dt, update_sub, state_dict=b_st)

                gt = gt.view_as(p).type_as(p)  # same casting logic
                if self._norm_processor is not None:
                    gt = self._norm_processor.process_grad(gt)

                st["step"] += 1
                st["u"] = ut.view_as(st["u"]).type_as(st["u"])
                p.add_(gt, alpha=-(lr * alpha))
                if wd != 0.0:
                    p.add_(p, alpha=-(lr * wd))
