from torch.optim.optimizer import ParamsT

from aro.processors import (
    AdamProcessor,
    ConstantNormProcessor,
    RotationProcessor,
    SinkhornProcessor,
)
from aro.processors.u import SCQRUpdate, UProcessor

from ._rot import AroBase

SCQRDefault = SCQRUpdate(rotation_jitter=1e-9)


class AroSinkhorn(AroBase):
    """Initialize an ARO Sinkhorn optimizer.

    Args:
        params (ParamsT): Model parameters to optimize.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        sinkhorn_steps (int, optional): Number of Sinkhorn iterations. Defaults to 5.
        eps (float, optional): Small value for numerical stability. Defaults to 1e-8.
        momentum (float, optional): Momentum factor for EMA processor. Defaults to 0.95.
        weight_decay (float, optional): Weight decay coefficient. Defaults to 0.0.
        alpha (float, optional): Weighting factor for base and EMA processors. Defaults to 0.2.
        update_freq (int, optional): Frequency of parameter updates. Defaults to 1.
        constant_norm (bool, optional): Whether to apply constant norm constraint. Defaults to True.
        u_processor (UProcessor, optional): Processor for U matrix decomposition. Defaults to SCQRDefault.
    """

    def __init__(
        self,
        params: ParamsT,
        *,
        lr: float = 1e-3,
        sinkhorn_steps: int = 5,
        eps: float = 1e-8,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        alpha: float = 0.2,
        update_freq: int = 1,
        constant_norm: bool = True,
        u_processor: UProcessor = SCQRDefault,
    ):
        base_processor = SinkhornProcessor(sinkhorn_steps=sinkhorn_steps, eps=eps)
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
            eps=eps,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=alpha,
            update_freq=update_freq,
        )
