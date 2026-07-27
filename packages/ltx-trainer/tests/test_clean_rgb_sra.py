import torch

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
