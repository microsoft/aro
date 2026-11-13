from unittest.mock import Mock

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from gpt import NanoGPT
from gpt.config import NanoGptConfig


def test_pretrained_loader_matches_huggingface_logits_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = NanoGptConfig(n_layer=1, n_head=2, n_embd=8, vocab_size=32, sequence_len=8)
    reference = GPT2LMHeadModel(
        GPT2Config(
            n_layer=1,
            n_head=2,
            n_embd=8,
            vocab_size=32,
            n_positions=8,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
    ).eval()
    # Keep the fixture small and offline while using real Hugging Face weights and inference.
    monkeypatch.setattr("gpt._nanogpt.NanoGptConfig", Mock(return_value=config))
    monkeypatch.setattr(GPT2LMHeadModel, "from_pretrained", Mock(return_value=reference))
    model = NanoGPT.from_pretrained("gpt2").eval()
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    with torch.no_grad():
        actual, _ = model(tokens)
        expected = reference(tokens, return_dict=True).logits[:, -1:, :]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
