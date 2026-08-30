"""ValidationRunner variant with Internal Guidance (IG) in the sampling loop.

Only two methods are overridden:

* ``_create_video_tools`` records the target token count for the sample that is
  about to be denoised (it is called once per sample, immediately before the
  state is built), so the hook knows where the target tokens start in the
  ``[references..., target]`` layout used during denoising.
* ``_run_denoising`` is a copy of the base implementation with the IG block
  added. Everything else (state building, conditioning, decoding, saving) is
  inherited unchanged.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import CFGGuider, STGGuider
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.model.transformer.model import X0Model
from ltx_core.tools import VideoLatentTools
from ltx_core.types import LatentState
from ltx_trainer import logger
from ltx_trainer.internal_guidance import (
    CalibrationMode,
    IGGuider,
    IGMode,
    InternalGuidanceCapture,
    calibrate_weak_prediction,
)
from ltx_trainer.progress import SamplingContext
from ltx_trainer.validation_runner import ValidationRunner

if TYPE_CHECKING:
    from ltx_core.model.transformer import LTXModel
    from ltx_trainer.sra import CleanRGBSRAHead


class InternalGuidanceMixin:
    """Adds Internal Guidance to any ValidationRunner subclass.

    Cooperative: put it *first* in the bases so its ``_create_video_tools`` and
    ``_run_denoising`` win, while ``super()`` still reaches the subclass that
    knows about the project-specific token layout / VAE / output handling::

        class IGControlValidationRunner(InternalGuidanceMixin, ControlValidationRunner):
            pass
    """

    def __init__(
        self,
        *args,
        sra_head: "CleanRGBSRAHead",
        ig_guider: IGGuider,
        ig_hidden_layer: int,
        ig_calibration: CalibrationMode = "none",
        ig_mode: IGMode = "parallel",
        **kwargs,
    ) -> None:
        # Set before super().__init__ so nothing downstream can observe a
        # half-initialised runner.
        self._sra_head = sra_head
        self._ig_guider = ig_guider
        self._ig_hidden_layer = int(ig_hidden_layer)
        self._ig_calibration: CalibrationMode = ig_calibration
        self._ig_mode: IGMode = ig_mode
        self._ig_target_token_count: int | None = None
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Override 1: remember how many tokens belong to the target video
    # ------------------------------------------------------------------

    def _create_video_tools(self, width: int, height: int, num_frames: int) -> VideoLatentTools:
        tools = super()._create_video_tools(width, height, num_frames)
        self._ig_target_token_count = tools.target_shape.token_count()
        return tools

    # ------------------------------------------------------------------
    # Override 2: denoising loop with the IG delta
    # ------------------------------------------------------------------

    def _run_denoising(  # noqa: PLR0913, PLR0912
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
        cfg = self._config
        scheduler = LTX2Scheduler()
        sigmas = scheduler.execute(steps=cfg.inference_steps).to(device).float()
        stepper = EulerDiffusionStep()
        cfg_guider = CFGGuider(cfg.guidance_scale)
        stg_guider = STGGuider(cfg.stg_scale)
        ig_guider = self._ig_guider

        stg_perturbation_config = (
            self._build_stg_perturbation_config(
                cfg.stg_blocks, cfg.stg_mode, transformer.num_blocks, device, next(transformer.parameters()).dtype
            )
            if stg_guider.enabled()
            else None
        )

        x0_model = X0Model(transformer)

        # --- IG setup -------------------------------------------------
        ig_enabled = ig_guider.enabled() and video_state is not None and not video_frozen
        capture: InternalGuidanceCapture | None = None
        if ig_enabled:
            if self._ig_target_token_count is None:
                raise ValueError("IG needs the target token count; _create_video_tools was never called")
            total_tokens = video_state.latent.shape[1]
            target_start = total_tokens - self._ig_target_token_count
            if target_start < 0:
                raise ValueError(
                    f"target token count {self._ig_target_token_count} exceeds sequence length {total_tokens}"
                )
            capture = InternalGuidanceCapture(transformer, self._sra_head, self._ig_hidden_layer)
            capture.set_target_start(target_start)
            logger.info(
                f"IG enabled: layer={self._ig_hidden_layer} scale={ig_guider.scale} "
                f"mode={self._ig_mode} calibration={self._ig_calibration} "
                f"interval=[{ig_guider.sigma_low}, {ig_guider.sigma_high}] "
                f"target_start={target_start}/{total_tokens}"
            )

        context = capture if capture is not None else _NullContext()
        with context:
            for step_idx, sigma in enumerate(sigmas[:-1]):
                v_sigma = torch.zeros_like(sigma) if video_frozen else sigma
                a_sigma = torch.zeros_like(sigma) if audio_frozen else sigma

                video = (
                    self._modality_from_latent_state(video_state, v_ctx_pos, v_sigma.unsqueeze(0))
                    if video_state is not None
                    else None
                )
                audio = (
                    self._modality_from_latent_state(audio_state, a_ctx_pos, a_sigma.unsqueeze(0))
                    if audio_state is not None
                    else None
                )

                # Arm the hook so it only captures the positive forward.
                step_needs_ig = capture is not None and ig_guider.active(sigma)
                if step_needs_ig:
                    capture.arm()
                pos_video, pos_audio = x0_model(video=video, audio=audio, perturbations=None)
                weak_video = capture.take() if step_needs_ig else None
                if capture is not None:
                    capture.disarm()

                denoised_video, denoised_audio = pos_video, pos_audio

                # CFG
                if cfg_guider.enabled() and v_ctx_neg is not None:
                    video_neg = replace(video, context=v_ctx_neg) if video is not None else None
                    audio_neg = replace(audio, context=a_ctx_neg) if audio is not None else None
                    neg_video, neg_audio = x0_model(video=video_neg, audio=audio_neg, perturbations=None)

                    if not video_frozen and denoised_video is not None:
                        denoised_video = denoised_video + cfg_guider.delta(pos_video, neg_video)
                    if not audio_frozen and denoised_audio is not None:
                        denoised_audio = denoised_audio + cfg_guider.delta(pos_audio, neg_audio)

                # STG
                if stg_perturbation_config is not None:
                    ptb_video, ptb_audio = x0_model(video=video, audio=audio, perturbations=stg_perturbation_config)
                    if not video_frozen and denoised_video is not None:
                        denoised_video = denoised_video + stg_guider.delta(pos_video, ptb_video)
                    if not audio_frozen and denoised_audio is not None and ptb_audio is not None:
                        denoised_audio = denoised_audio + stg_guider.delta(pos_audio, ptb_audio)

                # --- Internal Guidance (video, target tokens only) -----
                # `parallel` builds the delta from the *pre-guidance* prediction,
                # so it is an independent additive term exactly like STG and the
                # order relative to CFG does not matter.
                # `nested` builds it from the CFG/STG-corrected prediction, which
                # also multiplies those deltas by `scale`.
                if weak_video is not None and denoised_video is not None:
                    target_start = denoised_video.shape[1] - weak_video.shape[1]
                    strong_source = denoised_video if self._ig_mode == "nested" else pos_video
                    strong = strong_source[:, target_start:, :]
                    weak = calibrate_weak_prediction(
                        weak_video.to(strong.dtype), strong, self._ig_calibration
                    )
                    delta = ig_guider.delta(strong, weak)
                    denoised_video = denoised_video.clone()
                    denoised_video[:, target_start:, :] = denoised_video[:, target_start:, :] + delta

                # Re-apply conditioning mask
                if denoised_video is not None and video_clean is not None:
                    denoised_video = self._post_process_latent(
                        denoised_video,
                        video_state.denoise_mask,
                        video_clean.clean_latent,
                    )
                if denoised_audio is not None and audio_clean is not None:
                    denoised_audio = self._post_process_latent(
                        denoised_audio,
                        audio_state.denoise_mask,
                        audio_clean.clean_latent,
                    )

                # Euler step (skip for frozen modalities)
                if video_state is not None and not video_frozen:
                    video_state = replace(
                        video_state,
                        latent=stepper.step(video.latent, denoised_video, sigmas, step_idx),
                    )
                if audio_state is not None and not audio_frozen:
                    audio_state = replace(
                        audio_state,
                        latent=stepper.step(audio.latent, denoised_audio, sigmas, step_idx),
                    )

                sampling_ctx.advance_step()

        return video_state, audio_state


class IGValidationRunner(InternalGuidanceMixin, ValidationRunner):
    """Plain ValidationRunner + Internal Guidance."""


class _NullContext:
    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


# ---------------------------------------------------------------------------
# Offline diagnostic
# ---------------------------------------------------------------------------


@torch.inference_mode()
def probe_intermediate(
    runner: "InternalGuidanceMixin",
    transformer: "LTXModel",
    sample,
    cached_embeddings,
    cached_media,
    clean_target_latent: Tensor,
    sigmas: list[float],
    device: torch.device,
    calibration: CalibrationMode,
) -> list[dict[str, float]]:
    """Compare the weak (block-l) and strong (final) x0 predictions against ground truth.

    Runs no sampling loop: the ground-truth target latent is noised to each
    requested sigma and a single forward is taken. Cheap enough to run over a
    few hundred clips, and it answers the three questions that decide whether IG
    is worth wiring into generation at all:

    * ``mse_weak`` / ``mse_strong`` -- is the intermediate prediction a *bad*
      denoiser (usable) or a *broken* one (nothing to extrapolate from)?
    * ``cos_weak_strong`` -- if this is ~1.0 the delta is pure noise.
    * ``cos_delta_residual`` -- does ``x0_f - x0_i`` point along the final
      model's own residual error? Positive means extrapolating moves toward the
      data. This is a first-order proxy only: autoguidance is a statement about
      density, not per-sample MSE.

    ``norm_ratio`` is the global scale a cosine-trained head is missing, i.e.
    the ``alpha`` you would use for a fixed calibration.
    """
    from ltx_core.components.noisers import GaussianNoiser  # noqa: PLC0415

    dims = sample.video_dims or runner._config.video_dims
    width, height, num_frames = dims
    v_ctx_pos, a_ctx_pos, _, _ = runner._get_prompt_embeddings(cached_embeddings, device)

    video_tools = runner._create_video_tools(width, height, num_frames)
    target_token_count = video_tools.target_shape.token_count()

    base_state = video_tools.create_initial_state(
        device=device,
        dtype=torch.bfloat16,
        initial_latent=clean_target_latent.to(device=device, dtype=torch.bfloat16),
    )
    base_state = runner._apply_video_conditionings(base_state, video_tools, sample, cached_media, device)
    total_tokens = base_state.latent.shape[1]
    target_start = total_tokens - target_token_count
    truth = base_state.clean_latent[:, target_start:, :].float()

    x0_model = X0Model(transformer)
    capture = InternalGuidanceCapture(transformer, runner._sra_head, runner._ig_hidden_layer)
    capture.set_target_start(target_start)

    rows: list[dict[str, float]] = []
    with capture:
        for sigma_value in sigmas:
            generator = torch.Generator(device=device).manual_seed(sample.seed or runner._config.seed)
            noiser = GaussianNoiser(generator=generator)
            state = noiser(base_state, noise_scale=float(sigma_value))
            sigma = torch.tensor(float(sigma_value), device=device)
            video = runner._modality_from_latent_state(state, v_ctx_pos, sigma.unsqueeze(0))

            capture.arm()
            strong_full, _ = x0_model(video=video, audio=None, perturbations=None)
            weak_raw = capture.take()
            capture.disarm()

            strong = strong_full[:, target_start:, :].float()
            weak_raw = weak_raw.float()
            weak = calibrate_weak_prediction(weak_raw, strong, calibration).float()

            rows.append(
                {
                    "sigma": float(sigma_value),
                    "mse_weak": float(torch.mean((weak - truth) ** 2)),
                    "mse_strong": float(torch.mean((strong - truth) ** 2)),
                    "cos_weak_strong": float(_flat_cos(weak, strong)),
                    "cos_delta_residual": float(_flat_cos(strong - weak, truth - strong)),
                    "norm_ratio": float(strong.norm() / weak_raw.norm().clamp(min=1e-6)),
                    "rel_delta": float((strong - weak).norm() / strong.norm().clamp(min=1e-6)),
                }
            )
    return rows


def _flat_cos(a: Tensor, b: Tensor) -> Tensor:
    return torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1), dim=1).squeeze()
