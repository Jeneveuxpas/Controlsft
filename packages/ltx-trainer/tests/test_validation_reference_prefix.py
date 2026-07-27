import torch

from ltx_core.types import LatentState
from ltx_trainer.validation_runner import ValidationRunner


def test_validation_reference_tokens_match_training_prefix_layout() -> None:
    token_values = torch.tensor([10.0, 11.0, 20.0, 21.0, 30.0])
    attention_mask = torch.arange(25.0).reshape(1, 5, 5)
    state = LatentState(
        latent=token_values.reshape(1, 5, 1),
        denoise_mask=torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]).reshape(1, 5, 1),
        positions=token_values.reshape(1, 1, 5, 1),
        clean_latent=token_values.reshape(1, 5, 1),
        attention_mask=attention_mask,
    )

    output = ValidationRunner._move_appended_video_references_to_prefix(state, target_len=2)

    order = torch.tensor([2, 3, 4, 0, 1])
    assert torch.equal(output.latent.flatten(), token_values.index_select(0, order))
    assert torch.equal(output.clean_latent.flatten(), token_values.index_select(0, order))
    assert torch.equal(output.positions.flatten(), token_values.index_select(0, order))
    assert torch.equal(output.denoise_mask.flatten(), torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]))
    expected_attention = attention_mask.index_select(1, order).index_select(2, order)
    assert torch.equal(output.attention_mask, expected_attention)
