import pytest
import torch
from pydantic import ValidationError

from ltx_trainer.sra import CleanRGBSRAHead
from ltx_trainer.training_strategies.base_strategy import ModelInputs
from ltx_trainer.training_strategies.flexible import (
    FlexibleStrategy,
    FlexibleStrategyConfig,
    ModalityConfig,
)


def _strategy() -> FlexibleStrategy:
    return FlexibleStrategy(
        FlexibleStrategyConfig(
            video=ModalityConfig(is_generated=True, latents_dir="latents"),
            clean_rgb_sra_loss_weight=0.005,
            clean_rgb_sra_warmup_steps=100,
            clean_rgb_sra_beta=0.05,
        )
    )


def test_clean_rgb_sra_head_projects_per_token() -> None:
    head = CleanRGBSRAHead(input_dim=32, hidden_dim=16, output_dim=8)
    output = head(torch.randn(2, 7, 32))
    assert output.shape == (2, 7, 8)


def test_clean_rgb_sra_head_depth_and_learnable_residual_scaling() -> None:
    for num_layers in (2, 3, 5, 8):
        head = CleanRGBSRAHead(input_dim=32, hidden_dim=16, output_dim=8, num_layers=num_layers)
        assert len(head.blocks) == num_layers - 2
        assert sum(isinstance(module, torch.nn.Linear) for module in head.modules()) == num_layers
        scale_params = [param for name, param in head.named_parameters() if name.endswith("residual_scale")]
        assert len(scale_params) == len(head.blocks)
        if head.blocks:
            expected_scale = 1.0 / (len(head.blocks) ** 0.5)
            assert all(
                torch.isclose(block.residual_scale.detach(), torch.tensor(expected_scale)) for block in head.blocks
            )


def test_clean_rgb_sra_head_rejects_fewer_than_two_layers() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        CleanRGBSRAHead(input_dim=32, hidden_dim=16, output_dim=8, num_layers=1)


def test_clean_rgb_sra_hidden_layer_is_one_based() -> None:
    config = FlexibleStrategyConfig(
        video=ModalityConfig(is_generated=True, latents_dir="latents"),
        clean_rgb_sra_hidden_layer=1,
    )
    assert config.clean_rgb_sra_hidden_layer == 1

    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        FlexibleStrategyConfig(
            video=ModalityConfig(is_generated=True, latents_dir="latents"),
            clean_rgb_sra_hidden_layer=0,
        )


def test_clean_rgb_sra_uses_detached_clean_x0_and_warmup() -> None:
    clean_x0 = torch.ones(2, 3, 8, requires_grad=True)
    prediction = torch.zeros(2, 3, 8, requires_grad=True)
    inputs = ModelInputs(
        video=None,
        audio=None,
        video_targets=None,
        audio_targets=None,
        video_loss_mask=torch.tensor([[False] * 6 + [True] * 3, [False] * 6 + [True] * 3]),
        audio_loss_mask=None,
        video_clean_latents=clean_x0,
        video_target_start_index=6,
    )

    assert torch.equal(
        _strategy().compute_clean_rgb_sra_loss(prediction, inputs, global_step=0)[0],
        torch.zeros(2),
    )

    loss, metrics = _strategy().compute_clean_rgb_sra_loss(prediction, inputs, global_step=100)
    loss.mean().backward()

    assert prediction.grad is not None
    assert clean_x0.grad is None
    assert torch.allclose(loss, torch.full((2,), 0.004875))
    assert torch.allclose(metrics["train/clean_rgb_sra_raw"], torch.tensor(0.975))
    assert torch.allclose(metrics["train/clean_rgb_sra_weight"], torch.tensor(0.005))
