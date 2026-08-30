"""ControlValidationRunner with the *official* LTX guidance stack.

Keeps every part of the trainer harness (Part16 reference conditioning,
PrefixAware latent tools, tuned VAE decode, precomputed TE, output writing,
multi-GPU sharding in the calling script) and replaces ONLY the denoising
loop with the official pipeline components:

* ``FactoryGuidedDenoiser`` + ``_guided_denoise`` — cond/uncond/ptb passes
  batched into one forward
* ``MultiModalGuider.calculate`` — CFG + STG + modality-CFG + internal + **rescale**
  (rescale and skip_step do not exist in the trainer loop at all)
* ``euler_denoising_loop`` semantics via ``_step_state``

Internal Guidance supplies the official guider with an intermediate prediction,
so it is combined with CFG/STG before the shared rescale.

Not supported here (falls back loudly): frozen modalities. This runner is for
the video-only evaluation path (``generate_audio=False``).
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ltx_core.components.guiders import (
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.model.transformer.model import X0Model
from ltx_core.tools import VideoLatentTools
from ltx_core.types import LatentState
from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser
from ltx_pipelines.utils.samplers import _step_state
from ltx_trainer import logger
from ltx_trainer.official_ig import IGDenoiser, IGSettings
from ltx_trainer.progress import SamplingContext

if TYPE_CHECKING:
    from ltx_core.model.transformer import LTXModel
    from ltx_trainer.sra import CleanRGBSRAHead


class _NullContext:
    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


class OfficialStackMixin:
    """Mixin that swaps the trainer denoising loop for the official guidance stack.

    Put it FIRST in the bases so its ``_run_denoising`` and
    ``_create_video_tools`` win, while ``super()`` still reaches the
    project-specific runner (e.g. the ControlValidationRunner defined in
    ``infer_fixed56.py``)::

        class OfficialControlRunner(OfficialStackMixin, ControlValidationRunner):
            pass
    """

    def __init__(
        self,
        *args,
        rescale_scale: float = 0.7,
        modality_scale: float = 1.0,
        skip_step: int = 0,
        sra_head: "CleanRGBSRAHead | None" = None,
        ig_settings: IGSettings | None = None,
        **kwargs,
    ) -> None:
        self._rescale_scale = float(rescale_scale)
        self._modality_scale = float(modality_scale)
        self._skip_step = int(skip_step)
        self._sra_head = sra_head
        self._ig_settings = ig_settings
        self._ig_target_token_count: int | None = None
        super().__init__(*args, **kwargs)

    # Remember the target token count of the sample about to be denoised
    # (called once per sample, right before state building).
    def _create_video_tools(self, width: int, height: int, num_frames: int) -> VideoLatentTools:
        tools = super()._create_video_tools(width, height, num_frames)
        self._ig_target_token_count = tools.target_shape.token_count()
        return tools

    def _run_denoising(  # noqa: PLR0913
        self,
        transformer: "LTXModel",
        video_state: LatentState | None,
        audio_state: LatentState | None,
        video_clean: LatentState | None,
        audio_clean: LatentState | None,
        *,
        video_frozen: bool,
        audio_frozen: bool,
        v_ctx_pos: Tensor,
        a_ctx_pos: Tensor,
        v_ctx_neg: Tensor | None,
        a_ctx_neg: Tensor | None,
        device: torch.device,
        sampling_ctx: SamplingContext,
    ) -> tuple[LatentState | None, LatentState | None]:
        if video_frozen or audio_frozen:
            raise NotImplementedError(
                "OfficialStackValidationRunner supports the video-only generation path; "
                "frozen-modality flows should use the trainer stack."
            )

        cfg = self._config
        sigmas = LTX2Scheduler().execute(steps=cfg.inference_steps).to(device).float()
        stepper = EulerDiffusionStep()
        x0_model = X0Model(transformer)

        cfg_scale = float(cfg.guidance_scale)
        if cfg_scale != 1.0 and v_ctx_neg is None:
            logger.warning("guidance_scale=%.2f but no negative context; forcing CFG off", cfg_scale)
            cfg_scale = 1.0

        ig_settings = self._ig_settings
        sra_head = self._sra_head
        if ig_settings is None or sra_head is None or not ig_settings.guider.enabled():
            ig_settings = None
        video_params = MultiModalGuiderParams(
            cfg_scale=cfg_scale,
            stg_scale=float(cfg.stg_scale),
            stg_blocks=list(cfg.stg_blocks) if cfg.stg_scale != 0 else [],
            rescale_scale=self._rescale_scale,
            modality_scale=self._modality_scale,
            skip_step=self._skip_step,
        )
        video_guider_factory = create_multimodal_guider_factory(video_params, negative_context=v_ctx_neg)
        if ig_settings is not None:
            internal_on = replace(video_params, internal_scale=ig_settings.guider.scale)
            internal_off = replace(video_params, internal_scale=1.0)
            # Factory keys are inclusive upper bounds; step below sigma_low to keep the lower bound inclusive.
            lower_boundary = math.nextafter(ig_settings.guider.sigma_low, -math.inf)
            video_guider_factory = MultiModalGuiderFactory.from_dict(
                {
                    math.inf: internal_off,
                    ig_settings.guider.sigma_high: internal_on,
                    lower_boundary: internal_off,
                },
                negative_context=v_ctx_neg,
            )
        inner = FactoryGuidedDenoiser(
            v_context=v_ctx_pos,
            a_context=a_ctx_pos if audio_state is not None else None,
            video_guider_factory=video_guider_factory,
            audio_guider_factory=None,  # video-only path; absent modality gets the positive-only guider
        )

        ig_context: IGDenoiser | _NullContext = _NullContext()
        denoiser = inner
        if ig_settings is not None:
            assert sra_head is not None
            if self._ig_target_token_count is None:
                raise ValueError("IG needs the target token count; _create_video_tools was never called")
            if video_state is None:
                raise ValueError("Internal Guidance requires a video latent state")
            total_tokens = video_state.latent.shape[1]
            target_start = total_tokens - self._ig_target_token_count
            if target_start < 0:
                raise ValueError(
                    f"target token count {self._ig_target_token_count} exceeds sequence length {total_tokens}"
                )
            ig_context = denoiser = IGDenoiser(
                inner, transformer, sra_head, ig_settings, target_start
            )
            logger.info(
                "Official stack + internal: cfg=%.2f stg=%.2f rescale=%.2f | layer=%d gamma=%.2f "
                "calibration=%s interval=[%.2f, %.2f] target_start=%d/%d",
                cfg_scale,
                cfg.stg_scale,
                self._rescale_scale,
                ig_settings.hidden_layer,
                ig_settings.guider.scale,
                ig_settings.calibration,
                ig_settings.guider.sigma_low,
                ig_settings.guider.sigma_high,
                target_start,
                total_tokens,
            )
        else:
            logger.info(
                "Official stack: cfg=%.2f stg=%.2f rescale=%.2f modality=%.2f skip_step=%d steps=%d",
                cfg_scale,
                cfg.stg_scale,
                self._rescale_scale,
                self._modality_scale,
                self._skip_step,
                cfg.inference_steps,
            )

        # Official euler_denoising_loop body, with sampling_ctx progress kept.
        with ig_context, torch.inference_mode():
            for step_idx in range(len(sigmas) - 1):
                video_result, audio_result = denoiser(x0_model, video_state, audio_state, sigmas, step_idx)
                denoised_video = video_result.denoised if video_result is not None else None
                denoised_audio = audio_result.denoised if audio_result is not None else None
                video_state = _step_state(video_state, denoised_video, stepper, sigmas, step_idx)
                audio_state = _step_state(audio_state, denoised_audio, stepper, sigmas, step_idx)
                sampling_ctx.advance_step()

        return video_state, audio_state
