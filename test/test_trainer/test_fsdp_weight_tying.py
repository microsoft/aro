from unittest.mock import patch

import pytest
import torch

from gpt import NanoGPT
from trainer.config import ConfigContainer, ConfigServices
from trainer.models import ModelContainer


@pytest.mark.parametrize("tied", [True, False])
def test_fsdp_embedding_and_head_group(tied: bool) -> None:
    config = ConfigContainer(
        "config",
        "nanogpt",
        ["model.n_layer=1", "model.n_embd=16", "model.n_head=2", "model.vocab_size=32"],
    ).get(ConfigServices.BASE_CONFIG)
    with patch("trainer.models._services.get_device", return_value=torch.device("cpu")):
        container = ModelContainer(config)
        model = container._raw_model()
    assert isinstance(model, NanoGPT)
    token_embedding = model.transformer["wte"]
    assert isinstance(token_embedding, torch.nn.Embedding)
    if not tied:
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())

    with patch("trainer.models._services.fully_shard") as shard:
        container._fsdp2_nano_models()

    groups = [call.args[0] for call in shard.call_args_list]
    if tied:
        assert groups[-2] == [token_embedding, model.lm_head]
        assert token_embedding.weight is model.lm_head.weight
    else:
        assert groups[-2] is model.lm_head
    assert groups[-1] is model
