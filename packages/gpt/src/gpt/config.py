from typing import Literal

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class NanoGptConfig:
    name: Literal["nanogpt"] = "nanogpt"
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    vocab_size: int = 50304  # vocabulary 50257 padded up to nearest multiple of 64 for efficiency
    sequence_len: int = 1024
    bias: bool = True  # bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    dropout: float = 0.0


@dataclass(frozen=True)
class NanoChatConfig:
    name: Literal["nanochat"] = "nanochat"
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6  # number of query heads
    n_kv_head: int = 6  # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "L"


GPT_CONF = NanoGptConfig | NanoChatConfig
