from torch.optim.optimizer import ParamsT

from aro.processors import AdamProcessor, ConstantNormProcessor, RotationProcessor
from aro.processors.u import SCQRUpdate, UProcessor

from ._rot import AroBase

SCQRDefault = SCQRUpdate(rotation_jitter=1e-9)


class AroAdam(AroBase):
    """Rotated Adam optimizer with Sinkhorn normalization.

    Applies the Adam optimizer update within an optimized rotated space,
    using Sinkhorn iterations for robust rotation matrix computation.

    Args:
        params: Model parameters
        lr: Learning rate. Defaults to 1e-3.
        betas: Adam coefficients applied in the rotated basis. Defaults to (0.0, 0.99).
        eps: Small constant for numerical stability. Defaults to 1e-8.
        correct_bias: Apply bias correction in Adam. Defaults to True.
        alpha: Hyperparameter scaling factor. Defaults to 1.0.
        weight_decay: L2 regularization coefficient. Defaults to 0.0.
        update_freq: Frequency of rotation updates. Defaults to 1.
        momentum: EMA decay for rotation matrix. Defaults to 0.95.
        constant_norm: Apply constant norm constraint. Defaults to True.
        u_processor: Processor for rotation matrix updates. Defaults to SCQRUpdate.
    """

    def __init__(
        self,
        params: ParamsT,
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.0, 0.99),
        eps: float = 1e-8,
        correct_bias: bool = True,
        alpha: float = 1.0,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        update_freq: int = 1,
        constant_norm: bool = True,
        u_processor: UProcessor = SCQRDefault,
    ):
        base_processor = AdamProcessor(betas=betas, eps=eps, correct_bias=correct_bias)
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
