"""Tests for :class:`aro.lora.LoRAADBLinear` (ADB rank/orthogonality/frozen contract)."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.testing

from aro.lora import LoRAADBLinear


class TestRankGuard:
    def test_raises_when_rank_is_not_positive(self) -> None:
        with pytest.raises(ValueError, match="0 < 2 \\* rank <= min"):
            LoRAADBLinear(in_features=8, out_features=10, rank=0)

    def test_raises_when_rank_too_large(self) -> None:
        # min(in, out) = 8, 2 * rank must be <= 8 -> rank=5 violates it.
        with pytest.raises(ValueError, match="2 \\* rank <= min"):
            LoRAADBLinear(in_features=8, out_features=10, rank=5)

    def test_allows_boundary_rank(self) -> None:
        # 2 * rank == min(in, out) is allowed (not a strict inequality).
        lora = LoRAADBLinear(in_features=8, out_features=10, rank=4)
        assert lora.rank == 4

    def test_raises_from_linear_too(self) -> None:
        linear = nn.Linear(8, 10)
        with pytest.raises(ValueError, match="2 \\* rank <= min"):
            LoRAADBLinear.from_linear(linear, rank=5)


class TestOrthogonality:
    def test_a_has_orthonormal_columns(self) -> None:
        torch.manual_seed(0)
        lora = LoRAADBLinear(in_features=16, out_features=12, rank=4)
        torch.testing.assert_close(lora.A.T @ lora.A, torch.eye(4), atol=1e-5, rtol=0)

    def test_b_has_orthonormal_columns(self) -> None:
        torch.manual_seed(1)
        lora = LoRAADBLinear(in_features=16, out_features=12, rank=4)
        torch.testing.assert_close(lora.B.T @ lora.B, torch.eye(4), atol=1e-5, rtol=0)

    def test_factor_shapes_follow_adb_transpose_convention(self) -> None:
        lora = LoRAADBLinear(in_features=16, out_features=12, rank=4)
        assert lora.A.shape == (12, 4)
        assert lora.B.shape == (16, 4)


class TestFrozenContract:
    """ADB contract: A, D, B are always frozen (manual-product optimization)."""

    def test_parameters_are_frozen(self) -> None:
        lora = LoRAADBLinear(in_features=16, out_features=12, rank=4)
        assert not lora.A.requires_grad
        assert not lora.d_raw.requires_grad
        assert not lora.B.requires_grad
        assert not lora.weight.requires_grad
        assert dict(lora.named_parameters())["weight"] is lora.weight

    def test_manual_factors_are_plain_buffers(self) -> None:
        lora = LoRAADBLinear(in_features=16, out_features=12, rank=4, manual_factors=True)
        params = dict(lora.named_parameters())
        assert "A" not in params
        assert "d_raw" not in params
        assert "B" not in params
        buffers = dict(lora.named_buffers())
        assert "A" in buffers
        assert "d_raw" in buffers
        assert "B" in buffers

    def test_non_manual_factors_are_frozen_parameters(self) -> None:
        lora = LoRAADBLinear(in_features=16, out_features=12, rank=4, manual_factors=False)
        params = dict(lora.named_parameters())
        assert "A" in params
        assert "d_raw" in params
        assert "B" in params
        assert not params["A"].requires_grad
        assert not params["d_raw"].requires_grad
        assert not params["B"].requires_grad


class TestDInitAndSign:
    def test_default_d_init_is_1e_3(self) -> None:
        lora = LoRAADBLinear(in_features=16, out_features=12, rank=4)
        torch.testing.assert_close(lora.D, torch.full((4,), 1e-3))

    def test_d_is_identity_alias_for_d_raw(self) -> None:
        lora = LoRAADBLinear(in_features=16, out_features=12, rank=4)
        assert lora.D is lora.d_raw

    def test_custom_d_init(self) -> None:
        lora = LoRAADBLinear(in_features=16, out_features=12, rank=4, d_init=0.5)
        torch.testing.assert_close(lora.D, torch.full((4,), 0.5))

    def test_signed_d_is_used_directly_no_softplus(self) -> None:
        """A negative D must flip the adapter's sign directly (no softplus floor)."""
        torch.manual_seed(2)
        lora = LoRAADBLinear(in_features=16, out_features=12, rank=4, d_init=1e-3)
        with torch.no_grad():
            lora.d_raw.copy_(torch.tensor([-0.2, 0.3, -0.4, 0.1]))
        adapter = lora._adapter()
        expected = (lora.A * lora.d_raw.unsqueeze(0)) @ lora.B.T
        torch.testing.assert_close(adapter, expected)
        # A negative entry must be reachable / preserved (not clamped to positive).
        assert (lora.d_raw < 0).any()


class TestLowRankVsDenseForward:
    def test_forward_matches_dense_weight(self) -> None:
        torch.manual_seed(3)
        linear = nn.Linear(20, 16, bias=True)
        lora = LoRAADBLinear.from_linear(linear, rank=6, d_init=0.1)
        x = torch.randn(5, 20)

        low_rank_out = lora(x)
        dense_out = F.linear(x, lora._weight_eff(), lora.bias)
        torch.testing.assert_close(low_rank_out, dense_out, atol=1e-5, rtol=1e-5)

    def test_adapter_starts_close_to_base_with_small_d_init(self) -> None:
        torch.manual_seed(4)
        linear = nn.Linear(20, 16, bias=False)
        lora = LoRAADBLinear.from_linear(linear, rank=6, d_init=1e-3)
        x = torch.randn(5, 20)
        torch.testing.assert_close(lora(x), linear(x), atol=2e-2, rtol=2e-2)


class TestFromLinear:
    def test_base_weight_matches_source_exactly(self) -> None:
        linear = nn.Linear(10, 8, bias=True)
        lora = LoRAADBLinear.from_linear(linear, rank=3)
        torch.testing.assert_close(lora.weight, linear.weight.data)

    def test_bias_copied_and_frozen(self) -> None:
        linear = nn.Linear(10, 8, bias=True)
        lora = LoRAADBLinear.from_linear(linear, rank=3)
        assert lora.bias is not None
        torch.testing.assert_close(lora.bias.data, linear.bias.data)
        assert not lora.bias.requires_grad

    def test_preserves_evaluation_state(self) -> None:
        linear = nn.Linear(10, 8).eval()
        lora = LoRAADBLinear.from_linear(linear, rank=3)
        assert not lora.training


class TestDisableAndMerge:
    def test_disabled_adapter_returns_base_only(self) -> None:
        torch.manual_seed(5)
        linear = nn.Linear(16, 12, bias=True)
        lora = LoRAADBLinear.from_linear(linear, rank=4)
        x = torch.randn(3, 16)

        lora.set_adapter_enabled(False)
        torch.testing.assert_close(lora(x), F.linear(x, lora.weight, lora.bias))

    def test_merge_unmerge_equivalence(self) -> None:
        torch.manual_seed(6)
        linear = nn.Linear(16, 12, bias=True)
        lora = LoRAADBLinear.from_linear(linear, rank=4, d_init=0.2)
        lora.eval()
        x = torch.randn(3, 16)

        unmerged_out = lora(x)
        lora.merge_weight()
        torch.testing.assert_close(lora(x), unmerged_out, atol=1e-5, rtol=1e-5)
        lora.unmerge_weight()
        torch.testing.assert_close(lora(x), unmerged_out, atol=1e-5, rtol=1e-5)

    def test_loading_state_invalidates_merged_weight(self) -> None:
        source = LoRAADBLinear.from_linear(nn.Linear(16, 12), rank=4, d_init=0.2).eval()
        target = LoRAADBLinear.from_linear(nn.Linear(16, 12), rank=4).eval()
        target.merge_weight()

        target.load_state_dict(source.state_dict())

        assert target._merged_weight is None
        x = torch.randn(3, 16)
        torch.testing.assert_close(target(x), source(x))
