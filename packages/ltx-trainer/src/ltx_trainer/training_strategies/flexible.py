"""Flexible training strategy for a unified conditioning framework.
This strategy implements the Unified Conditioning Framework that supports:
- Simple fine-tuning with text conditioning (text-to-video/audio)
- Intrinsic conditioning (first_frame, prefix, suffix, spatial_crop, mask)
- Extrinsic conditioning (concatenation-based, IC-LoRA style)
The flexible strategy replaces TextToVideoStrategy and VideoToVideoStrategy by expressing
all conditioning scenarios through configuration rather than code.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from ltx_core.model.transformer.modality import Modality
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    VIDEO_SCALE_FACTORS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)

# =============================================================================
# Configuration Classes
# =============================================================================


class IntrinsicConditionBase(BaseModel):
    """Base for intrinsic conditioning — tokens get clean latents, timestep=0, no loss."""

    model_config = ConfigDict(extra="forbid")

    probability: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Probability of applying this condition",
    )


class FirstFrameConditionConfig(IntrinsicConditionBase):
    """First frame conditioning — frame 0 is clean, excluded from loss."""

    type: Literal["first_frame"] = "first_frame"


class PrefixConditionConfig(IntrinsicConditionBase):
    """Prefix conditioning — first N temporal units are clean, excluded from loss."""

    type: Literal["prefix"] = "prefix"
    temporal_boundary: int = Field(
        ...,
        ge=1,
        description="Number of temporal units for prefix region. "
        "For video: number of latent frames. For audio: number of audio latent timesteps.",
    )


class SuffixConditionConfig(IntrinsicConditionBase):
    """Suffix conditioning — last N temporal units are clean, excluded from loss."""

    type: Literal["suffix"] = "suffix"
    temporal_boundary: int = Field(
        ...,
        ge=1,
        description="Number of temporal units for suffix region. "
        "For video: number of latent frames. For audio: number of audio latent timesteps.",
    )


class SpatialCropConditionConfig(IntrinsicConditionBase):
    """Spatial crop conditioning — rectangular pixel region is clean, excluded from loss."""

    type: Literal["spatial_crop"] = "spatial_crop"
    spatial_region: tuple[int, int, int, int] = Field(
        ...,
        description="Spatial crop region as (y1, x1, y2, x2) in pixel coordinates",
    )


class MaskConditionConfig(IntrinsicConditionBase):
    """Mask conditioning — per-sample binary mask determines conditioning tokens."""

    type: Literal["mask"] = "mask"
    mask_dir: str = Field(
        ...,
        description="Directory containing per-sample masks",
    )


class ReferenceConditionConfig(BaseModel):
    """Reference conditioning (IC-LoRA style concatenation).
    External reference latents are concatenated to the target sequence.
    Reference tokens are clean (timestep=0), excluded from loss, and
    participate in bidirectional self-attention.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["reference"] = "reference"
    latents_dir: str = Field(..., description="Directory for reference latents")
    probability: float = Field(default=1.0, ge=0.0, le=1.0, description="Probability of applying this condition")


# Discriminated union for condition configs
ConditionConfig = Annotated[
    Union[
        FirstFrameConditionConfig,
        PrefixConditionConfig,
        SuffixConditionConfig,
        SpatialCropConditionConfig,
        MaskConditionConfig,
        ReferenceConditionConfig,
    ],
    Field(discriminator="type"),
]


def _reference_conditions_in_prepend_order(
    conditions: list[ConditionConfig],
) -> list[ReferenceConditionConfig]:
    """Return references in reverse application order so prepending preserves declaration order."""
    return [condition for condition in reversed(conditions) if isinstance(condition, ReferenceConditionConfig)]


class ModalityConfig(BaseModel):
    """Configuration for a single modality (video or audio)."""

    model_config = ConfigDict(extra="forbid")

    is_generated: bool = Field(
        ...,
        description="True = generated modality (denoised, contributes to loss), False = conditioning-only modality",
    )

    latents_dir: str = Field(
        ...,
        description="Directory for latents",
    )

    conditions: list[ConditionConfig] = Field(
        default_factory=list,
        description="List of conditions (e.g. first_frame, prefix, reference). Text conditioning is always applied.",
    )


class RepresentationDistillationConfig(BaseModel):
    """Frozen-teacher feature distillation with configurable timestep pairing."""

    model_config = ConfigDict(extra="forbid")

    teacher_checkpoint: str | None = Field(
        default=None,
        description="Full-model teacher checkpoint. Defaults to model.load_checkpoint when omitted.",
    )
    teacher_conditions: list[ReferenceConditionConfig] = Field(
        default_factory=list,
        description="Reference conditions visible to the teacher but not the student.",
    )
    student_hidden_layer: int = Field(default=16, ge=1, description="One-based student transformer block number")
    teacher_hidden_layer: int = Field(default=34, ge=1, description="One-based teacher transformer block number")
    projector: Literal["none", "mlp"] = Field(default="mlp")
    projector_hidden_dim: int = Field(
        default=1024,
        gt=0,
        description="Hidden width of the two-linear-layer student projector",
    )
    projector_learning_rate: float | None = Field(default=None, gt=0.0)
    loss_type: Literal["cosine", "l1", "l2"] = Field(
        default="cosine",
        description="Feature loss; cosine is preferred for cross-layer alignment",
    )
    loss_weight: float = Field(default=0.8, gt=0.0)
    warmup_steps: int = Field(default=0, ge=0)
    noise_mode: Literal["same", "independent_high_low", "sra", "dual_timestep"] = Field(
        default="same",
        description=(
            "same: shared t; independent_high_low: independently sample t/s and sort them; "
            "sra: teacher uses t-U(0,x); dual_timestep: student mixes independently sampled t/s"
        ),
    )
    sra_timestep_max_gap: float = Field(
        default=0.2,
        gt=0.0,
        le=1.0,
        description="Maximum x in the SRA teacher timestep t-U(0,x), clamped to [0,1]",
    )
    dual_timestep_second_probability: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description="Expected fraction of student target tokens assigned the independently sampled second timestep",
    )


class FlexibleStrategyConfig(TrainingStrategyConfigBase):
    """Configuration for the flexible training strategy.
    This strategy supports all conditioning scenarios through configuration:
    - Text-to-video/audio with simple fine-tuning
    - Intrinsic conditioning like first-frame, extension, outpainting
    - Reference conditioning like IC-LoRA (concatenation-based reference)
    """

    name: Literal["flexible"] = "flexible"

    video: ModalityConfig | None = Field(
        default=None,
        description="Video modality configuration",
    )

    audio: ModalityConfig | None = Field(
        default=None,
        description="Audio modality configuration",
    )

    clean_rgb_sra_checkpoint: str | None = Field(default=None)
    clean_rgb_sra_loss_weight: float = Field(default=0.0, ge=0.0)
    clean_rgb_sra_learning_rate: float | None = Field(default=None, gt=0.0)
    clean_rgb_sra_hidden_layer: int = Field(
        default=8,
        ge=1,
        description="One-based transformer block number whose video hidden states feed the clean RGB SRA head",
    )
    clean_rgb_sra_hidden_dim: int | None = Field(default=None, gt=0)
    clean_rgb_sra_num_layers: int = Field(default=5, ge=2)
    clean_rgb_sra_warmup_steps: int = Field(default=100, ge=0)
    clean_rgb_sra_loss_type: Literal["smooth_l1", "cosine"] = Field(
        default="smooth_l1",
        description="Loss used to align SRA predictions with detached clean video latents",
    )
    clean_rgb_sra_beta: float = Field(default=0.05, gt=0.0)
    representation_distillation: RepresentationDistillationConfig | None = Field(default=None)

    @model_validator(mode="after")
    def validate_at_least_one_generated(self) -> "FlexibleStrategyConfig":
        """Ensure at least one modality has is_generated=true."""
        has_video_target = self.video is not None and self.video.is_generated
        has_audio_target = self.audio is not None and self.audio.is_generated
        if not has_video_target and not has_audio_target:
            raise ValueError("At least one modality must have is_generated=true")
        return self

    @model_validator(mode="after")
    def validate_audio_intrinsic_regions(self) -> "FlexibleStrategyConfig":
        """Reject video-only intrinsic regions on the audio modality."""
        if self.audio is None:
            return self
        for cond in self.audio.conditions:
            if isinstance(cond, (FirstFrameConditionConfig, SpatialCropConditionConfig)):
                raise ValueError(
                    f"Intrinsic condition '{cond.type}' is not supported for audio. "
                    f"Audio supports: prefix, suffix, mask."
                )
        return self

    @model_validator(mode="after")
    def validate_representation_distillation(self) -> "FlexibleStrategyConfig":
        distillation = self.representation_distillation
        if distillation is None:
            return self
        if self.video is None or not self.video.is_generated:
            raise ValueError("Representation distillation requires a generated video modality")
        if self.audio is not None:
            raise ValueError("Representation distillation currently supports video-only training")
        if self.video.conditions:
            raise ValueError(
                "The distilled student must not have video conditions; put teacher-only references under "
                "representation_distillation.teacher_conditions"
            )
        if any(condition.probability != 1.0 for condition in distillation.teacher_conditions):
            raise ValueError("Teacher reference conditions must use probability=1.0 for deterministic pairing")
        return self

    def get_data_sources(self) -> dict[str, str]:
        """Dynamically determine required data sources from config.
        Returns a mapping of directory name (under ``preprocessed_data_root``) to
        the dataset output key.
        """
        sources: dict[str, str] = {"conditions": "conditions"}

        if self.video is not None:
            sources[self.video.latents_dir] = "video_latents"
        if self.audio is not None:
            sources[self.audio.latents_dir] = "audio_latents"

        for modality_config in (self.video, self.audio):
            if modality_config is None:
                continue
            for cond in modality_config.conditions:
                if isinstance(cond, ReferenceConditionConfig):
                    sources[cond.latents_dir] = cond.latents_dir
                elif isinstance(cond, MaskConditionConfig):
                    sources[cond.mask_dir] = cond.mask_dir

        if self.representation_distillation is not None:
            for condition in self.representation_distillation.teacher_conditions:
                sources[condition.latents_dir] = condition.latents_dir

        return sources


# =============================================================================
# Helper Data Structures
# =============================================================================


@dataclass
class ModalityProcessingResult:
    """Result of processing a single modality."""

    modality: Modality
    targets: Tensor | None
    loss_mask: Tensor | None
    clean_latents: Tensor | None = None
    target_start_index: int | None = None


@dataclass
class InitializedTarget:
    """A precomputed noisy view of one clean target sequence."""

    noisy_latents: Tensor
    targets: Tensor
    timesteps: Tensor
    loss_mask: Tensor
    sigmas: Tensor


@dataclass
class DistillationModelInputs:
    """Paired student and teacher inputs derived from the same clean sample and noise."""

    student: ModelInputs
    teacher: ModelInputs
    base_sigmas: Tensor
    low_sigmas: Tensor | None
    second_timestep_mask: Tensor | None


@dataclass
class LatentData:
    """Loaded and patchified latents with metadata."""

    latents: Tensor  # [B, seq_len, C]
    num_frames: int
    height: int
    width: int
    fps: float


# =============================================================================
# FlexibleStrategy Implementation
# =============================================================================


class FlexibleStrategy(TrainingStrategy):
    """Unified training strategy supporting all conditioning scenarios.
    This strategy implements the Unified Conditioning Framework, allowing
    any training scenario to be expressed through configuration.
    """

    config: FlexibleStrategyConfig

    def __init__(self, config: FlexibleStrategyConfig):
        """Initialize strategy with configuration.
        Args:
            config: Flexible strategy configuration
        """
        super().__init__(config)
        self.config = config
        self.reference_spatial_scale_factor, self.reference_temporal_scale_factor = (
            self._infer_reference_scale_factors_from_config()
        )

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare training inputs by processing video and audio modalities."""
        video_result = self._process_modality(self.config.video, batch, "video", timestep_sampler)
        audio_result = self._process_modality(self.config.audio, batch, "audio", timestep_sampler)

        return self._to_model_inputs(video_result, audio_result)

    def prepare_distillation_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> DistillationModelInputs:
        """Build paired video inputs with one shared clean target and Gaussian noise."""
        distillation = self.config.representation_distillation
        video_config = self.config.video
        if distillation is None or video_config is None:
            raise ValueError("Representation distillation is not configured")

        data = self._patchify_latent_data(batch["video_latents"], "video")
        student_target, teacher_target, low_sigmas, second_mask = self._initialize_distillation_targets(
            data.latents,
            timestep_sampler,
            noise_mode=distillation.noise_mode,
            sra_timestep_max_gap=distillation.sra_timestep_max_gap,
            dual_timestep_second_probability=distillation.dual_timestep_second_probability,
        )
        teacher_config = video_config.model_copy(update={"conditions": list(distillation.teacher_conditions)})
        student_result = self._process_modality_data(
            video_config,
            batch,
            "video",
            data,
            initialized_target=student_target,
        )
        teacher_result = self._process_modality_data(
            teacher_config,
            batch,
            "video",
            data,
            initialized_target=teacher_target,
        )
        return DistillationModelInputs(
            student=self._to_model_inputs(student_result, None),
            teacher=self._to_model_inputs(teacher_result, None),
            base_sigmas=student_target.sigmas,
            low_sigmas=low_sigmas,
            second_timestep_mask=second_mask,
        )

    @staticmethod
    def _to_model_inputs(
        video_result: ModalityProcessingResult | None,
        audio_result: ModalityProcessingResult | None,
    ) -> ModelInputs:
        return ModelInputs(
            video=video_result.modality if video_result else None,
            audio=audio_result.modality if audio_result else None,
            video_targets=video_result.targets if video_result else None,
            audio_targets=audio_result.targets if audio_result else None,
            video_loss_mask=video_result.loss_mask if video_result else None,
            audio_loss_mask=audio_result.loss_mask if audio_result else None,
            video_clean_latents=video_result.clean_latents if video_result else None,
            video_target_start_index=video_result.target_start_index if video_result else None,
        )

    def compute_loss(
        self,
        video_pred: Tensor | None,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute masked MSE loss for video and audio predictions. Returns [B,]."""
        total_loss = None

        if video_pred is not None and inputs.video_targets is not None:
            video_loss = self._compute_modality_loss(
                pred=video_pred,
                targets=inputs.video_targets,
                loss_mask=inputs.video_loss_mask,
            )
            total_loss = video_loss

        if audio_pred is not None and inputs.audio_targets is not None:
            audio_loss = self._compute_modality_loss(
                pred=audio_pred,
                targets=inputs.audio_targets,
                loss_mask=inputs.audio_loss_mask,
            )
            total_loss = audio_loss if total_loss is None else total_loss + audio_loss

        if total_loss is None:
            raise ValueError("No valid predictions and targets provided for loss computation")

        return total_loss

    def compute_clean_rgb_sra_loss(
        self, clean_rgb_pred: Tensor, inputs: ModelInputs, global_step: int
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Align target hidden states with detached clean RGB VAE latents."""
        if inputs.video_clean_latents is None:
            raise ValueError("Clean RGB SRA requires clean target video latents")
        target = inputs.video_clean_latents.to(device=clean_rgb_pred.device, dtype=torch.float32).detach()
        if clean_rgb_pred.shape != target.shape:
            raise ValueError(
                f"Clean RGB SRA prediction shape {tuple(clean_rgb_pred.shape)} "
                f"does not match target shape {tuple(target.shape)}"
            )
        prediction = clean_rgb_pred.float()
        if self.config.clean_rgb_sra_loss_type == "cosine":
            # Direction-only alignment is intentionally different from x0 regression:
            # latent magnitude is ignored while token masking and loss weighting stay unchanged.
            token_loss = 1.0 - torch.nn.functional.cosine_similarity(prediction, target, dim=-1)
        else:
            token_loss = torch.nn.functional.smooth_l1_loss(
                prediction, target, reduction="none", beta=float(self.config.clean_rgb_sra_beta)
            ).mean(dim=-1)
        if inputs.video_loss_mask is None or inputs.video_target_start_index is None:
            raise ValueError("Clean RGB SRA requires the target video loss mask")
        target_mask = inputs.video_loss_mask[:, inputs.video_target_start_index :].to(token_loss.device).float()
        raw_loss = (token_loss * target_mask).sum(dim=1) / target_mask.sum(dim=1).clamp(min=1.0)
        weight = float(self.config.clean_rgb_sra_loss_weight)
        warmup = int(self.config.clean_rgb_sra_warmup_steps)
        if warmup > 0:
            weight *= min(1.0, max(int(global_step), 0) / float(warmup))
        weight_tensor = torch.tensor(weight, device=clean_rgb_pred.device, dtype=torch.float32)
        weighted = raw_loss * weight_tensor
        return weighted, {
            "train/clean_rgb_sra_raw": raw_loss.detach().mean(),
            "train/clean_rgb_sra_loss": weighted.detach().mean(),
            "train/clean_rgb_sra_weight": weight_tensor.detach(),
        }

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        """Include reference scale factors in checkpoint metadata for inference pipelines."""
        metadata: dict[str, Any] = {}
        spatial = self.reference_spatial_scale_factor
        temporal = self.reference_temporal_scale_factor
        if spatial is not None and spatial != 1:
            metadata["reference_spatial_scale_factor"] = spatial
            metadata["reference_downscale_factor"] = spatial  # backward compat
        if temporal is not None and temporal != 1:
            metadata["reference_temporal_scale_factor"] = temporal
        return metadata

    def _infer_reference_scale_factors_from_config(self) -> tuple[int | None, int | None]:
        """Infer spatial and temporal scale factors by peeking at one sample pair."""
        if self.config.video is None:
            return None, None
        conditions: list[ConditionConfig] = list(self.config.video.conditions)
        if self.config.representation_distillation is not None:
            conditions.extend(self.config.representation_distillation.teacher_conditions)
        for cond in conditions:
            if not isinstance(cond, ReferenceConditionConfig):
                continue
            target_dir = Path(self.config.video.latents_dir)
            ref_dir = Path(cond.latents_dir)
            for sample_file in target_dir.rglob("*.pt"):
                ref_file = ref_dir / sample_file.relative_to(target_dir)
                if not ref_file.exists():
                    continue
                target_data = torch.load(sample_file, map_location="cpu", weights_only=True)
                ref_data = torch.load(ref_file, map_location="cpu", weights_only=True)
                if "height" not in ref_data or "height" not in target_data:
                    continue
                spatial = self._infer_scale_factor(
                    ref_data["height"],
                    ref_data["width"],
                    target_data["height"],
                    target_data["width"],
                )
                temporal = self._infer_temporal_scale_factor(
                    ref_data["num_frames"],
                    target_data["num_frames"],
                )
                return spatial, temporal
        return None, None

    def _process_modality(
        self,
        modality_config: ModalityConfig | None,
        batch: dict[str, Any],
        modality_key: str,
        timestep_sampler: TimestepSampler,
    ) -> ModalityProcessingResult | None:
        """Process a single modality: load latents, add noise, apply conditions, build Modality."""
        if modality_config is None:
            return None

        data = self._patchify_latent_data(batch[f"{modality_key}_latents"], modality_key)
        return self._process_modality_data(modality_config, batch, modality_key, data, timestep_sampler)

    def _process_modality_data(
        self,
        modality_config: ModalityConfig,
        batch: dict[str, Any],
        modality_key: str,
        data: LatentData,
        timestep_sampler: TimestepSampler | None = None,
        initialized_target: InitializedTarget | None = None,
    ) -> ModalityProcessingResult:
        """Build a modality from patchified data and an optional precomputed noisy target."""
        latents = data.latents
        clean_target_latents = latents if modality_config.is_generated and modality_key == "video" else None

        batch_size, seq_len, _ = latents.shape
        device = latents.device
        dtype = latents.dtype

        # Step 2: Get text embeddings
        conditions = batch["conditions"]
        prompt_embeds = conditions[f"{modality_key}_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        # Step 3: Initialize noise, timesteps, and loss mask based on is_generated flag
        if modality_config.is_generated:
            if initialized_target is None:
                if timestep_sampler is None:
                    raise ValueError("A timestep sampler is required when no initialized target is provided")
                noisy_latents, targets, timesteps, loss_mask, sigmas = self._initialize_noisy_target(
                    latents, timestep_sampler
                )
            else:
                noisy_latents = initialized_target.noisy_latents
                targets = initialized_target.targets
                timesteps = initialized_target.timesteps
                loss_mask = initialized_target.loss_mask
                sigmas = initialized_target.sigmas
        else:
            # Conditioning modality: keep clean (sigma=0), no loss
            noisy_latents = latents
            targets = None
            timesteps = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
            loss_mask = None
            sigmas = torch.zeros(batch_size, device=device, dtype=dtype)

        # Step 4: Generate positions
        if modality_key == "video":
            positions = self._get_video_positions(
                num_frames=data.num_frames,
                height=data.height,
                width=data.width,
                batch_size=batch_size,
                fps=data.fps,
                device=device,
            )
        else:
            positions = self._get_audio_positions(
                num_time_steps=seq_len,
                batch_size=batch_size,
                device=device,
            )

        # Step 5: Apply conditions (intrinsic first, then extrinsic)
        for cond in modality_config.conditions:
            if isinstance(cond, IntrinsicConditionBase) and modality_config.is_generated:
                noisy_latents, timesteps, loss_mask = self._apply_intrinsic_condition(
                    noisy_latents=noisy_latents,
                    clean_latents=latents,
                    timesteps=timesteps,
                    loss_mask=loss_mask,
                    config=cond,
                    height=data.height,
                    width=data.width,
                    batch=batch,
                )

        # Each reference is prepended to the current sequence. Apply them in
        # reverse declaration order so the final token order matches the config.
        for cond in _reference_conditions_in_prepend_order(modality_config.conditions):
            noisy_latents, positions, timesteps, loss_mask, targets = self._apply_reference_condition(
                noisy_latents=noisy_latents,
                positions=positions,
                timesteps=timesteps,
                loss_mask=loss_mask,
                targets=targets,
                batch=batch,
                config=cond,
                modality_key=modality_key,
            )

        # Step 6: Build Modality
        modality = Modality(
            enabled=True,
            latent=noisy_latents,
            sigma=sigmas,
            timesteps=timesteps,
            positions=positions,
            context=prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        return ModalityProcessingResult(
            modality=modality,
            targets=targets,
            loss_mask=loss_mask,
            clean_latents=clean_target_latents,
            target_start_index=noisy_latents.shape[1] - seq_len if modality_config.is_generated else None,
        )

    @staticmethod
    def _initialize_noisy_target(
        latents: Tensor,
        timestep_sampler: TimestepSampler,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Add noise to latents and create training targets. Returns (noisy, targets, timesteps, mask, sigmas)."""
        batch_size, seq_len, _ = latents.shape
        sigmas = timestep_sampler.sample_for(latents)
        noise = torch.randn_like(latents)
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_latents = (1 - sigmas_expanded) * latents + sigmas_expanded * noise
        targets = noise - latents  # velocity prediction
        timesteps = sigmas.view(-1, 1).expand(batch_size, seq_len).clone()
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=latents.device)
        return noisy_latents, targets, timesteps, loss_mask, sigmas

    @staticmethod
    def _initialize_distillation_targets(
        latents: Tensor,
        timestep_sampler: TimestepSampler,
        *,
        noise_mode: Literal["same", "independent_high_low", "sra", "dual_timestep"],
        sra_timestep_max_gap: float,
        dual_timestep_second_probability: float,
    ) -> tuple[InitializedTarget, InitializedTarget, Tensor | None, Tensor | None]:
        """Construct student/teacher views using the same x0 and Gaussian noise."""
        batch_size, seq_len, _ = latents.shape
        sampled_sigmas = timestep_sampler.sample_for(latents)
        noise = torch.randn_like(latents)
        targets = noise - latents
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=latents.device)

        student_sigmas = sampled_sigmas
        teacher_sigmas = sampled_sigmas
        low_sigmas = None
        second_sigmas = None
        second_mask = None

        if noise_mode == "sra":
            interval = float(sra_timestep_max_gap) * torch.rand_like(sampled_sigmas)
            low_sigmas = torch.clamp(sampled_sigmas - interval, min=0.0, max=1.0)
            teacher_sigmas = low_sigmas
        elif noise_mode in {"independent_high_low", "dual_timestep"}:
            second_sigmas = timestep_sampler.sample_for(latents)
            low_sigmas = torch.minimum(sampled_sigmas, second_sigmas)
            teacher_sigmas = low_sigmas
            if noise_mode == "independent_high_low":
                student_sigmas = torch.maximum(sampled_sigmas, second_sigmas)

        student_timesteps = student_sigmas[:, None].expand(batch_size, seq_len).clone()
        if noise_mode == "dual_timestep":
            assert second_sigmas is not None
            second_mask = torch.rand(batch_size, seq_len, device=latents.device) < dual_timestep_second_probability
            second_timesteps = second_sigmas[:, None].expand(batch_size, seq_len)
            student_timesteps = torch.where(second_mask, second_timesteps, student_timesteps)

        teacher_timesteps = teacher_sigmas[:, None].expand(batch_size, seq_len).clone()
        student_tau = student_timesteps.unsqueeze(-1)
        teacher_tau = teacher_timesteps.unsqueeze(-1)
        student_noisy = ((1 - student_tau) * latents + student_tau * noise).to(dtype=latents.dtype)
        teacher_noisy = ((1 - teacher_tau) * latents + teacher_tau * noise).to(dtype=latents.dtype)

        student = InitializedTarget(
            noisy_latents=student_noisy,
            targets=targets,
            timesteps=student_timesteps,
            loss_mask=loss_mask,
            # LTX-2.3 prompt AdaLN requires one scalar. Keep the base student t,
            # matching the unmasked branch and the original dual-timestep setup.
            sigmas=student_sigmas,
        )
        teacher = InitializedTarget(
            noisy_latents=teacher_noisy,
            targets=targets.clone(),
            timesteps=teacher_timesteps,
            loss_mask=loss_mask.clone(),
            sigmas=teacher_sigmas,
        )
        return student, teacher, low_sigmas, second_mask

    def _apply_intrinsic_condition(
        self,
        noisy_latents: Tensor,
        clean_latents: Tensor,
        timesteps: Tensor,
        loss_mask: Tensor,
        config: IntrinsicConditionBase,
        height: int,
        width: int,
        batch: dict[str, Any],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply intrinsic conditioning using a binary mask.
        For each token, the mask value determines conditioning strength:
        - mask=1: conditioned (clean latent, timestep=0, excluded from loss)
        - mask=0: generated (noisy latent, original timestep, contributes to loss)
        The conditioning decision is drawn independently per batch element so the training
        signal across samples in a batch is i.i.d. -- a single batch-wide draw would
        correlate gradient updates across the batch.
        """
        batch_size, seq_len, _ = noisy_latents.shape
        device = noisy_latents.device

        # Per-sample Bernoulli draw -- each element is independently conditioned.
        apply_per_sample = torch.rand(batch_size, device=device) < config.probability
        if not apply_per_sample.any():
            return noisy_latents, timesteps, loss_mask

        if isinstance(config, FirstFrameConditionConfig):
            mask = self._compute_temporal_mask(batch_size, seq_len, height, width, 1, False, device)
        elif isinstance(config, PrefixConditionConfig):
            mask = self._compute_temporal_mask(
                batch_size, seq_len, height, width, config.temporal_boundary, False, device
            )
        elif isinstance(config, SuffixConditionConfig):
            mask = self._compute_temporal_mask(
                batch_size, seq_len, height, width, config.temporal_boundary, True, device
            )
        elif isinstance(config, SpatialCropConditionConfig):
            mask = self._compute_spatial_crop_mask(batch_size, seq_len, height, width, config.spatial_region, device)
        elif isinstance(config, MaskConditionConfig):
            # Binarize to match inference, which thresholds masks at load time
            # (validation_runner._load_and_downsample_mask / _load_audio_mask).
            mask = (batch[config.mask_dir]["mask"].reshape(batch_size, seq_len) > 0.5).float()
        else:
            raise ValueError(f"Unknown intrinsic condition type: {type(config).__name__}")

        # Zero the mask for samples the per-sample draw did not select.
        mask = mask * apply_per_sample.view(-1, 1).to(mask.dtype)

        # Apply binary mask: clean conditioned tokens, noisy generated tokens.
        m = mask.unsqueeze(-1)
        noisy_latents = m * clean_latents + (1 - m) * noisy_latents
        timesteps = (1 - mask) * timesteps
        loss_mask = loss_mask & (mask == 0)

        return noisy_latents, timesteps, loss_mask

    @staticmethod
    def _compute_temporal_mask(
        batch_size: int,
        seq_len: int,
        height: int,
        width: int,
        num_frames: int,
        from_end: bool,
        device: torch.device,
    ) -> Tensor:
        """Compute float mask for temporal region (prefix or suffix). Returns [B, seq_len] in {0, 1}."""
        tokens_per_frame = height * width
        num_tokens = num_frames * tokens_per_frame
        mask = torch.zeros(batch_size, seq_len, device=device)
        if from_end:
            mask[:, -num_tokens:] = 1.0
        else:
            mask[:, :num_tokens] = 1.0
        return mask

    @staticmethod
    def _compute_spatial_crop_mask(
        batch_size: int,
        seq_len: int,
        height: int,
        width: int,
        region: tuple[int, int, int, int],
        device: torch.device,
    ) -> Tensor:
        """Compute float mask for spatial crop region (y1, x1, y2, x2) in pixel coords.
        Returns [B, seq_len] in {0, 1}.
        """
        y1, x1, y2, x2 = region
        num_frames = seq_len // (height * width)

        # Convert pixel to latent coordinates and clamp (per-axis VAE scale factor).
        def to_latent(v: int, scale: int, max_v: int) -> int:
            return max(0, min(v // scale, max_v))

        ly1 = to_latent(y1, VIDEO_SCALE_FACTORS.height, height)
        ly2 = to_latent(y2, VIDEO_SCALE_FACTORS.height, height)
        lx1 = to_latent(x1, VIDEO_SCALE_FACTORS.width, width)
        lx2 = to_latent(x2, VIDEO_SCALE_FACTORS.width, width)

        # Create spatial mask and tile across frames
        spatial_mask = torch.zeros(height, width, device=device)
        spatial_mask[ly1:ly2, lx1:lx2] = 1.0
        full_mask = spatial_mask.flatten().repeat(num_frames)

        return full_mask.unsqueeze(0).expand(batch_size, -1)

    def _patchify_latent_data(self, latent_data: dict[str, Any], modality_key: str) -> LatentData:
        """Patchify latent data and extract metadata."""
        latents = latent_data["latents"]

        if modality_key == "video":
            num_frames = latent_data["num_frames"][0].item()
            height = latent_data["height"][0].item()
            width = latent_data["width"][0].item()
            fps = latent_data.get("fps")
            fps = fps[0].item() if fps is not None else DEFAULT_FPS
            latents = self._video_patchifier.patchify(latents)
        else:
            num_frames = latent_data.get("num_frames", [latents.shape[2]])[0]
            if isinstance(num_frames, Tensor):
                num_frames = num_frames.item()
            height = 1
            width = 1
            fps = 1.0
            latents = self._audio_patchifier.patchify(latents)

        return LatentData(latents=latents, num_frames=num_frames, height=height, width=width, fps=fps)

    def _apply_reference_condition(
        self,
        noisy_latents: Tensor,
        positions: Tensor,
        timesteps: Tensor,
        loss_mask: Tensor | None,
        targets: Tensor | None,
        batch: dict[str, Any],
        config: ReferenceConditionConfig,
        modality_key: str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        """Concatenate reference latents to target sequence for reference conditioning (IC-LoRA style).
        The apply/skip decision is batch-wide (reference conditioning changes the sequence
        length, so it cannot be applied to only part of a batch) but is drawn from the torch
        RNG so runs are reproducible under ``torch.manual_seed`` — mirroring the intrinsic
        per-sample draw rather than Python's unseeded ``random``.
        """
        if torch.rand((), device=noisy_latents.device).item() >= config.probability:
            return noisy_latents, positions, timesteps, loss_mask, targets

        # Load and patchify condition latents
        cond = self._patchify_latent_data(batch[config.latents_dir], modality_key)
        cond_latents = cond.latents

        batch_size, cond_seq_len, _ = cond_latents.shape
        device = cond_latents.device
        dtype = cond_latents.dtype

        # Generate condition positions
        if modality_key == "video":
            cond_positions = self._get_video_positions(
                num_frames=cond.num_frames,
                height=cond.height,
                width=cond.width,
                batch_size=batch_size,
                fps=cond.fps,
                device=device,
            )
        else:
            cond_positions = self._get_audio_positions(
                num_time_steps=cond_seq_len,
                batch_size=batch_size,
                device=device,
            )

        # Translate / rescale ref positions into the target's frame (video only).
        if modality_key == "video":
            spatial_sf = self.reference_spatial_scale_factor or 1
            temporal_sf = self.reference_temporal_scale_factor or 1
            if spatial_sf != 1 or temporal_sf != 1:
                cond_positions = cond_positions.clone()
                if temporal_sf != 1:
                    # Ref positions are already at the ref's effective fps (source_fps / S,
                    # stored by process_videos.py). Shift by (S - 1) / target_fps so ref's
                    # last patch aligns with target's last; clamp the causal patch at 0.
                    t_target = positions[:, 0, 0:1, 1:2]  # = 1 / target_fps
                    cond_positions[:, 0, ...] = torch.clamp(
                        cond_positions[:, 0, ...] - (temporal_sf - 1) * t_target, min=0
                    )
                if spatial_sf != 1:
                    cond_positions[:, 1, ...] *= spatial_sf
                    cond_positions[:, 2, ...] *= spatial_sf

        # Condition tokens: clean, timestep=0, no loss
        cond_timesteps = torch.zeros(batch_size, cond_seq_len, device=device, dtype=dtype)
        cond_loss_mask = torch.zeros(batch_size, cond_seq_len, dtype=torch.bool, device=device)

        # Concatenate condition and target sequences (condition first, then target)
        combined_latents = torch.cat([cond_latents, noisy_latents], dim=1)
        combined_positions = torch.cat([cond_positions, positions], dim=2)
        combined_timesteps = torch.cat([cond_timesteps, timesteps], dim=1)

        combined_loss_mask = torch.cat([cond_loss_mask, loss_mask], dim=1) if loss_mask is not None else None

        # Targets remain unchanged (only for target portion, not condition portion)

        return combined_latents, combined_positions, combined_timesteps, combined_loss_mask, targets

    @staticmethod
    def _compute_modality_loss(pred: Tensor, targets: Tensor, loss_mask: Tensor) -> Tensor:
        """Compute per-element MSE loss for a single modality. Returns [B,]."""
        # Slice prediction to match targets length (removes any prepended condition tokens)
        target_len = targets.shape[1]
        pred = pred[:, -target_len:, :]
        mask = loss_mask[:, -target_len:]

        # Compute masked MSE loss, reduce per-element [B,] over (seq, channels)
        mask_expanded = mask.unsqueeze(-1).float()
        squared_error = (pred - targets).pow(2)
        masked_loss = squared_error * mask_expanded
        return masked_loss.mean(dim=[-2, -1]) / mask_expanded.mean(dim=[-2, -1]).clamp(min=1e-8)

    @staticmethod
    def _infer_scale_factor(cond_height: int, cond_width: int, target_height: int, target_width: int) -> int:
        """Infer spatial scale factor between condition and target resolutions."""
        if target_height == cond_height and target_width == cond_width:
            return 1
        scale_h = target_height // cond_height if cond_height > 0 else 1
        scale_w = target_width // cond_width if cond_width > 0 else 1
        if scale_h != scale_w:
            raise ValueError(
                f"Non-uniform scale factors between condition and target: height={scale_h}, width={scale_w}. "
                "Condition and target resolutions must scale uniformly."
            )
        return scale_h

    @staticmethod
    def _infer_temporal_scale_factor(cond_num_frames: int, target_num_frames: int) -> int:
        """Infer temporal scale factor between condition and target latent frame counts.
        The first latent frame encodes a single pixel frame (the VAE's causal structure),
        so the temporal groups count is (num_frames - 1).  The scale factor is the ratio
        of target groups to condition groups.
        """
        if target_num_frames == cond_num_frames:
            return 1
        target_groups = target_num_frames - 1
        cond_groups = cond_num_frames - 1
        if cond_groups <= 0 or target_groups <= 0:
            return 1
        if target_groups % cond_groups != 0:
            raise ValueError(
                f"Target temporal groups ({target_groups}) is not evenly divisible by "
                f"condition temporal groups ({cond_groups})."
            )
        return target_groups // cond_groups
