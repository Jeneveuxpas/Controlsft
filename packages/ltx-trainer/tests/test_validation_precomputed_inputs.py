from typing import Never

import pytest
import torch

from ltx_trainer.config import ReferenceConditionConfig, ValidationConfig, ValidationSample
from ltx_trainer.validation_runner import (
    CachedConditionMedia,
    CachedPromptEmbeddings,
    CachedSampleMedia,
    ValidationRunner,
)


def _config() -> ValidationConfig:
    return ValidationConfig(
        samples=[
            ValidationSample(
                prompt="precomputed",
                conditions=[ReferenceConditionConfig(video="precomputed://signal_latent")],
            )
        ],
        generate_audio=False,
        generate_video=True,
        interval=None,
    )


def test_validation_runner_accepts_precomputed_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(*_args, **_kwargs) -> Never:
        raise AssertionError("online encoder path should not be called")

    monkeypatch.setattr(ValidationRunner, "_cache_prompt_embeddings", fail_if_called)
    monkeypatch.setattr(ValidationRunner, "_encode_conditioning_media", fail_if_called)
    monkeypatch.setattr(ValidationRunner, "_load_decoder_components", lambda _self: None)

    context = torch.zeros(1, 4, 8)
    embeddings = [
        CachedPromptEmbeddings(
            video_context_positive=context,
            audio_context_positive=context,
        )
    ]
    media = [CachedSampleMedia(conditions={0: CachedConditionMedia(latent=torch.zeros(1, 128, 2, 2, 2))})]

    runner = ValidationRunner(
        config=_config(),
        model_path="unused.safetensors",
        text_encoder_path=None,
        precomputed_embeddings=embeddings,
        precomputed_media=media,
        tiled_video_decode=False,
    )
    assert runner._tiled_video_decode is False


def test_validation_runner_rejects_mismatched_precomputed_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ValidationRunner, "_load_decoder_components", lambda _self: None)
    with pytest.raises(ValueError, match="Expected 1 precomputed embedding entries, got 0"):
        ValidationRunner(
            config=_config(),
            model_path="unused.safetensors",
            text_encoder_path=None,
            precomputed_embeddings=[],
            precomputed_media=[],
        )
