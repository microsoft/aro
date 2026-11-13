import torch

from aro.processors import SignProcessor
from aro.processors.u import (
    CholQrUpdate,
    SCQRUpdate,
    chol_qr_safe_take_stats,
    scqr_safe_take_stats,
)
from aro.processors.u._qr import chol_qr_safe, scqr_safe


def test_qr_defaults_preserve_internal_regularization() -> None:
    diagonal = torch.tensor([1.0, 1e-4, 2.0, 3.0], dtype=torch.float64)
    matrix = torch.diag(diagonal)
    regularizer = 1e-7 * torch.linalg.vector_norm(diagonal.square())
    expected = torch.diag(diagonal / (diagonal.square() + regularizer).sqrt())
    for normalize in (scqr_safe, chol_qr_safe):
        torch.testing.assert_close(normalize(matrix), expected, rtol=1e-12, atol=1e-12)

    processor = SignProcessor(eps=1e-8)
    transformed = matrix @ processor.process_grad(matrix).T
    expected_update = chol_qr_safe(transformed, jitter=1e-7)
    for update in (SCQRUpdate(), CholQrUpdate()):
        actual = update(torch.eye(4, dtype=torch.float64), matrix, False, processor, state_dict={})
        torch.testing.assert_close(actual, expected_update, rtol=0, atol=0)


def test_legacy_qr_statistics_share_the_scqr_counter() -> None:
    scqr_safe_take_stats()
    try:
        scqr_safe(torch.eye(4))
        chol_qr_safe(torch.zeros(4, 4))
        stats = chol_qr_safe_take_stats(reset=False)
        assert stats["calls"] == 2
        assert stats["items_fallback"] == 1
        assert scqr_safe_take_stats() == stats
        assert chol_qr_safe_take_stats()["calls"] == 0
    finally:
        scqr_safe_take_stats()
