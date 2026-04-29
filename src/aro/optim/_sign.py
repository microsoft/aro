from torch.optim.optimizer import ParamsT

from aro.processors import AdamProcessor, ConstantNormProcessor, RotationProcessor, SignProcessor
from aro.processors.u import SCQRUpdate, UProcessor

from ._rot import AroBase

SCQRDefault = SCQRUpdate(rotation_jitter=1e-9)


class AroSign(AroBase):
    """Initialize the AroSign optimizer.

    Args:
        params (ParamsT): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): Learning rate. Default: 1e-3
        eps (float, optional): Term added to the denominator to improve numerical
            stability. Default: 1e-8
        momentum (float, optional): Momentum factor for EMA processor. Default: 0.95
        weight_decay (float, optional): Weight decay (L2 penalty). Default: 0.0
        alpha (float, optional): Scaling factor for the update. Default: 0.2
        update_freq (int, optional): Frequency of updates. Default: 1
        constant_norm (bool, optional): If True, applies constant norm processor
            to maintain consistent norm across updates. Default: True
        u_processor (UProcessor, optional): Processor for handling U matrix
            decomposition in rotation. Default: SCQRDefault

    Example:
        >>> optimizer = AroSign(model.parameters(), lr=1e-3, momentum=0.95)
    """

    def __init__(
        self,
        params: ParamsT,
        *,
        lr: float = 1e-3,
        eps: float = 1e-8,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        alpha: float = 0.2,
        update_freq: int = 1,
        constant_norm: bool = True,
        u_processor: UProcessor = SCQRDefault,
    ):
        base_processor = SignProcessor(eps=eps)
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
