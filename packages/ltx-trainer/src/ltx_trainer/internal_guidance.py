"""Internal Guidance (IG) for LTX-2 sampling.

Reuses the Clean RGB SRA head trained by ``ltx_trainer.trainer`` as a *weak*
intermediate x0 predictor and extrapolates between it and the final-layer x0
prediction during sampling.

Reference: Zhou et al., "Guiding a Diffusion Transformer with the Internal
Dynamics of Itself", CVPR 2026 (arXiv:2512.24176).

Two facts about this codebase make the integration cheap:

1. ``X0Model`` already converts the transformer's velocity output to x0 via
   ``to_denoised``, and CFG/STG are applied as additive deltas *in x0 space*.
   The SRA head also predicts x0.  So no velocity reparameterisation is needed
   and IG has exactly the same algebraic shape as ``CFGGuider.delta``.
2. ``LTXModel.register_post_block_hook`` lets us read block ``l``'s video
   hidden state during the forward pass that already runs, so IG costs one
   small MLP per step and *zero* extra transformer forwards.

Calibration
-----------
A head trained with ``clean_rgb_sra_loss_type: cosine`` is only supervised on
direction; its output magnitude is arbitrary, so ``x0_f - x0_i`` would be
dominated by scale mismatch rather than by the weak/strong difference.
``calibrate_weak_prediction`` rescales the weak prediction onto the strong
prediction's magnitude before the delta is taken.  Use ``none`` for heads
trained with ``smooth_l1`` (they are already in x0 units).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from ltx_trainer.sra import CleanRGBSRAHead, extract_clean_rgb_sra_state_dict

CalibrationMode = Literal["none", "token_norm", "sample_norm"]
IGMode = Literal["parallel", "nested"]

_EPS = 1e-6


# ---------------------------------------------------------------------------
# Guider
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IGGuider:
    """Internal-guidance guider.

    ``x0_w = x0_i + scale * (x0_f - x0_i) = x0_f + (scale - 1) * (x0_f - x0_i)``

    The second form is what ``delta`` returns, so it composes additively with
    ``CFGGuider`` / ``STGGuider`` exactly like the other guiders in
    ``ltx_core.components.guiders``.

    Attributes:
        scale: Extrapolation strength ``gamma``. 1.0 disables IG. IG and
            ReWorld both converge to ~1.4 at XL / video scale.
        sigma_low / sigma_high: Guidance interval. IG is applied only when
            ``sigma_low <= sigma <= sigma_high``. The IG paper reports that,
            unlike CFG, internal guidance should be applied in the *high and
            mid* noise range; ``sigma_low=0.3`` reproduces their best setting.
        max_delta_norm: Optional safety clamp on the per-sample L2 norm of the
            delta. 0 disables it.
    """

    scale: float = 1.0
    sigma_low: float = 0.0
    sigma_high: float = 1.0
    max_delta_norm: float = 0.0

    def enabled(self) -> bool:
        return self.scale != 1.0

    def active(self, sigma: float | Tensor) -> bool:
        value = float(sigma.item() if isinstance(sigma, Tensor) else sigma)
        return self.enabled() and self.sigma_low <= value <= self.sigma_high

    def delta(self, strong: Tensor, weak: Tensor) -> Tensor:
        raw = (self.scale - 1.0) * (strong.float() - weak.float())
        if self.max_delta_norm > 0:
            batch = raw.shape[0]
            norm = raw.reshape(batch, -1).norm(p=2, dim=1).clamp(min=_EPS)
            factor = torch.minimum(torch.ones_like(norm), self.max_delta_norm / norm)
            raw = raw * factor.reshape(batch, *([1] * (raw.ndim - 1)))
        return raw.to(strong.dtype)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibrate_weak_prediction(weak: Tensor, strong: Tensor, mode: CalibrationMode) -> Tensor:
    """Put the weak x0 prediction on the same scale as the strong one.

    ``none``
        Identity. Correct for ``smooth_l1`` heads, which regress x0 values.
    ``token_norm``
        Per-token rescale so ``||weak_j|| == ||strong_j||`` for every token
        ``j``. The resulting delta is purely angular. This is the right default
        for ``cosine`` heads, which learn direction only and discard both the
        global and the per-token magnitude.
    ``sample_norm``
        One scalar per batch element. Cheaper and smoother than ``token_norm``,
        but it cannot recover the per-token magnitude structure that a cosine
        loss throws away.
    """
    if mode == "none":
        return weak
    weak_f = weak.float()
    strong_f = strong.float()
    if mode == "token_norm":
        weak_norm = weak_f.norm(dim=-1, keepdim=True).clamp(min=_EPS)
        strong_norm = strong_f.norm(dim=-1, keepdim=True)
        return (weak_f * (strong_norm / weak_norm)).to(weak.dtype)
    if mode == "sample_norm":
        batch = weak_f.shape[0]
        weak_norm = weak_f.reshape(batch, -1).norm(dim=1).clamp(min=_EPS)
        strong_norm = strong_f.reshape(batch, -1).norm(dim=1)
        factor = (strong_norm / weak_norm).reshape(batch, *([1] * (weak_f.ndim - 1)))
        return (weak_f * factor).to(weak.dtype)
    raise ValueError(f"Unknown calibration mode: {mode}")


# ---------------------------------------------------------------------------
# Head loading
# ---------------------------------------------------------------------------


def _base_model(transformer: torch.nn.Module) -> torch.nn.Module:
    return transformer.get_base_model() if hasattr(transformer, "get_base_model") else transformer


def load_sra_head(
    checkpoint: Path,
    transformer: torch.nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    hidden_layer: int | None = None,
    hidden_dim: int | None = None,
    num_layers: int | None = None,
) -> tuple[CleanRGBSRAHead, dict]:
    """Load a Clean RGB SRA head and return it with its training metadata.

    Accepts either the standalone export written by
    ``Trainer._save_clean_rgb_sra_head`` (``clean_rgb_sra_head_step_XXXXX.pt``,
    which carries the layer / depth / loss-type metadata) or a full training
    ``state_dict`` containing ``clean_rgb_sra_head.*`` entries.
    """
    if checkpoint.suffix == ".safetensors":
        from safetensors.torch import load_file  # noqa: PLC0415

        payload = load_file(checkpoint, device="cpu")
    else:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(payload, Mapping) and "clean_rgb_sra_head" in payload:
        state = dict(payload["clean_rgb_sra_head"])
        metadata = dict(payload.get("metadata", {}))
    else:
        state = extract_clean_rgb_sra_state_dict(payload)
        metadata = {}

    base = _base_model(transformer)
    input_dim = int(base.inner_dim)
    output_dim = int(base.proj_out.out_features)

    resolved_layers = num_layers or metadata.get("clean_rgb_sra_num_layers")
    if resolved_layers is None:
        # input_proj + output_proj + one Linear per residual block
        resolved_layers = 2 + sum(1 for key in state if key.startswith("blocks.") and key.endswith(".proj.weight"))
    resolved_hidden = hidden_dim or metadata.get("clean_rgb_sra_hidden_dim")
    if resolved_hidden is None:
        resolved_hidden = int(state["input_proj.weight"].shape[0])
    resolved_layer = hidden_layer or metadata.get("clean_rgb_sra_hidden_layer")
    if resolved_layer is None:
        raise ValueError(
            f"{checkpoint} has no clean_rgb_sra_hidden_layer metadata; pass --ig-layer explicitly."
        )

    head = CleanRGBSRAHead(
        input_dim=input_dim,
        hidden_dim=int(resolved_hidden),
        output_dim=output_dim,
        num_layers=int(resolved_layers),
    )
    head.load_state_dict(state, strict=True)
    head = head.to(device=device, dtype=dtype).eval()
    head.requires_grad_(False)

    metadata.setdefault("clean_rgb_sra_hidden_layer", int(resolved_layer))
    metadata.setdefault("clean_rgb_sra_num_layers", int(resolved_layers))
    metadata.setdefault("clean_rgb_sra_hidden_dim", int(resolved_hidden))
    metadata.setdefault("clean_rgb_sra_loss_type", "unknown")
    return head, metadata


def default_calibration_for(loss_type: str) -> CalibrationMode:
    """A cosine-trained head has no magnitude; an L1-trained head does."""
    return "token_norm" if loss_type == "cosine" else "none"


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


class InternalGuidanceCapture:
    """Runs the SRA head on block ``l``'s video hidden state during the forward.

    Registered as a post-block hook, so it sees the activations the model
    already computes. Must be *armed* around the positive (conditional) forward
    only: the hook would otherwise also fire for the CFG-negative and STG
    forwards and the last write would win.
    """

    def __init__(
        self,
        transformer: torch.nn.Module,
        head: CleanRGBSRAHead,
        hidden_layer: int,
    ) -> None:
        base = _base_model(transformer)
        num_blocks = len(base.transformer_blocks)
        if not 1 <= hidden_layer <= num_blocks:
            raise ValueError(f"hidden_layer {hidden_layer} is outside 1..{num_blocks}")
        self._base = base
        self._head = head
        self._hidden_layer = int(hidden_layer)
        self._target_start: int | None = None
        self._armed = False
        self._prediction: Tensor | None = None
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    @property
    def hidden_layer(self) -> int:
        return self._hidden_layer

    def set_target_start(self, target_start: int) -> None:
        """Index of the first *target* video token in the ``[refs..., target]`` layout."""
        self._target_start = int(target_start)

    def __enter__(self) -> "InternalGuidanceCapture":
        self._handle = self._base.register_post_block_hook(self._hidden_layer - 1, self._hook)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def arm(self) -> None:
        self._armed = True
        self._prediction = None

    def disarm(self) -> None:
        self._armed = False

    def take(self) -> Tensor | None:
        prediction, self._prediction = self._prediction, None
        return prediction

    def _hook(self, video_args: object, _audio_args: object) -> None:
        if not self._armed or video_args is None:
            return
        if self._target_start is None:
            raise ValueError("InternalGuidanceCapture.set_target_start was never called")
        hidden = video_args.x[:, self._target_start :, :]
        head_param = next(self._head.parameters())
        self._prediction = self._head(hidden.to(device=head_param.device, dtype=head_param.dtype))
