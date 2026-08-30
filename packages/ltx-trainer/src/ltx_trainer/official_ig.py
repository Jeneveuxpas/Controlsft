"""Internal Guidance integrated into the *official* LTX denoising stack.

The official stack batches all guidance passes (cond / uncond / ptb / mod)
into one transformer forward. This module captures an intermediate prediction
from that same forward and supplies it to ``MultiModalGuider.calculate`` as
``internal``. CFG, STG, modality guidance and Internal Guidance are therefore
combined before a single shared rescale:

``pred = cond + cfg_delta + stg_delta + modality_delta + internal_delta``

The official ``MultiModalGuiderFactory`` is the source of truth for the sigma
interval: the hook is armed only when its current params enable ``internal``.

Because the official forward is batched (``[cond..., uncond..., ptb...]``
along batch), the block hook fires once per step; the capture slices the first
``orig_b`` rows to isolate the cond pass, then the target-token suffix of the
``[references..., target]`` layout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser
from ltx_trainer.internal_guidance import (
    CalibrationMode,
    IGGuider,
    calibrate_weak_prediction,
)
from ltx_trainer.sra import CleanRGBSRAHead

if TYPE_CHECKING:
    from ltx_core.model.transformer import X0Model
    from ltx_core.types import LatentState
    from ltx_pipelines.utils.types import DenoisedLatentResult


class BatchedIGCapture:
    """Post-block hook for the official batched-pass forward.

    Slices ``x[:orig_b, target_start:, :]`` — first ``orig_b`` batch rows are
    the cond pass; the token suffix is the denoised target (references are
    prefixed and stay clean). Must be armed per step with the current
    ``orig_b`` so the slice stays correct if the pass set changes.
    """

    def __init__(self, transformer: torch.nn.Module, head: CleanRGBSRAHead, hidden_layer: int) -> None:
        base = transformer.get_base_model() if hasattr(transformer, "get_base_model") else transformer
        num_blocks = len(base.transformer_blocks)
        if not 1 <= hidden_layer <= num_blocks:
            raise ValueError(f"hidden_layer {hidden_layer} is outside 1..{num_blocks}")
        self._base = base
        self._head = head
        self._hidden_layer = int(hidden_layer)
        self._target_start: int | None = None
        self._orig_b: int = 0
        self._armed = False
        self._prediction: torch.Tensor | None = None
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def set_target_start(self, target_start: int) -> None:
        self._target_start = int(target_start)

    def __enter__(self) -> "BatchedIGCapture":
        self._handle = self._base.register_post_block_hook(self._hidden_layer - 1, self._hook)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def arm(self, orig_b: int) -> None:
        self._armed = True
        self._orig_b = int(orig_b)
        self._prediction = None

    def disarm(self) -> None:
        self._armed = False

    def take(self) -> torch.Tensor | None:
        prediction, self._prediction = self._prediction, None
        return prediction

    def _hook(self, video_args: object, _audio_args: object) -> None:
        if not self._armed or video_args is None:
            return
        if self._target_start is None:
            raise ValueError("BatchedIGCapture.set_target_start was never called")
        hidden = video_args.x[: self._orig_b, self._target_start :, :]
        head_param = next(self._head.parameters())
        self._prediction = self._head(hidden.to(device=head_param.device, dtype=head_param.dtype))


@dataclass(frozen=True)
class IGSettings:
    guider: IGGuider
    hidden_layer: int
    calibration: CalibrationMode = "none"


class IGDenoiser:
    """Arm the block capture around an internal-aware official denoiser.

    Satisfies the same ``Denoiser`` protocol, so it drops into
    ``euler_denoising_loop`` / ``DiffusionStage`` unchanged. Use as a context
    manager so the block hook is registered/removed around the loop::

        with IGDenoiser(inner, transformer_raw, head, settings, target_start) as denoiser:
            euler_denoising_loop(..., denoiser=denoiser)
    """

    def __init__(
        self,
        inner: FactoryGuidedDenoiser,
        transformer_raw: torch.nn.Module,
        head: CleanRGBSRAHead,
        settings: IGSettings,
        target_start: int,
    ) -> None:
        self._inner = inner
        self._settings = settings
        self._capture = BatchedIGCapture(transformer_raw, head, settings.hidden_layer)
        self._capture.set_target_start(target_start)
        if not hasattr(inner, "video_internal_provider"):
            raise TypeError("IGDenoiser requires a guided denoiser with a video_internal_provider")
        if inner.video_internal_provider is not None:
            raise ValueError("The wrapped denoiser already has a video internal-prediction provider")

    def __enter__(self) -> "IGDenoiser":
        if self._inner.video_internal_provider is not None:
            raise ValueError("The wrapped denoiser already has a video internal-prediction provider")
        self._capture.__enter__()
        self._inner.video_internal_provider = self._provide_internal
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            self._capture.__exit__(*exc)
        finally:
            self._inner.video_internal_provider = None

    def __call__(
        self,
        transformer: X0Model,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[DenoisedLatentResult | None, DenoisedLatentResult | None]:
        factory = self._inner.video_guider_factory
        params = factory.params(sigmas[step_index]) if factory is not None else None
        step_needs_internal = (
            video_state is not None
            and params is not None
            and not math.isclose(params.internal_scale, 1.0)
        )
        if step_needs_internal:
            self._capture.arm(orig_b=video_state.latent.shape[0])

        try:
            return self._inner(transformer, video_state, audio_state, sigmas, step_index)
        finally:
            self._capture.disarm()

    def _provide_internal(self, cond: torch.Tensor) -> torch.Tensor | None:
        weak = self._capture.take()
        if weak is None:
            return None

        target_start = cond.shape[1] - weak.shape[1]
        if target_start < 0:
            raise ValueError(
                f"Internal prediction has {weak.shape[1]} tokens, more than cond's {cond.shape[1]} tokens"
            )
        strong = cond[:, target_start:, ...]
        weak = calibrate_weak_prediction(
            weak.to(device=strong.device, dtype=strong.dtype),
            strong,
            self._settings.calibration,
        )
        internal = cond.clone()
        internal[:, target_start:, ...] = weak
        return internal
