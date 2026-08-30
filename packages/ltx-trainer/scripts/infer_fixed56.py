#!/usr/bin/env python

"""Run fixed-56 control or text-only evaluation using official LTX components.

This file contains only project-specific orchestration: fixed-manifest loading,
precomputed TE/control handling, checkpoint replacement, optional tuned-VAE
replacement, Controlsft reference-token layout, multi-GPU worker dispatch, and
stable ``with_control/line-N`` and ``target_only/line-N`` output names. Model
construction, denoising, scheduling, and VAE architecture are delegated to LTX.
"""

# ruff: noqa: PLR0912, PLR0915, T201
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from rich.console import Console
from safetensors.torch import load_file
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF  # noqa: N812

from ltx_core.tools import VideoLatentTools
from ltx_core.types import LatentState
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
from ltx_core.model.video_vae import VAE_DECODER_COMFY_KEYS_FILTER, VideoDecoderConfigurator
from ltx_core.text_encoders.gemma import convert_to_additive_mask
from ltx_trainer.config import ReferenceConditionConfig, ValidationConfig, ValidationSample
from ltx_trainer.ig_validation_runner import InternalGuidanceMixin
from ltx_trainer.internal_guidance import IGGuider, default_calibration_for, load_sra_head
from ltx_trainer.official_ig import IGSettings
from ltx_trainer.official_validation_runner import OfficialStackMixin
from ltx_trainer.model_loader import load_embeddings_processor, load_transformer
from ltx_trainer.progress import TrainingProgress
from ltx_trainer.video_utils import save_video
from ltx_trainer.validation_runner import (
    CachedConditionMedia,
    CachedPromptEmbeddings,
    CachedSampleMedia,
    ValidationRunner,
)

console = Console()
VIDEO_SPATIAL_COMPRESSION = 32
VIDEO_TEMPORAL_COMPRESSION = 8


@dataclass(frozen=True)
class Item:
    manifest_index: int
    sample_id: str
    te_path: Path
    reference_latent: torch.Tensor
    video_dims: tuple[int, int, int]
    frame_rate: float
    with_control_path: Path
    target_only_path: Path


@dataclass(frozen=True)
class Worker:
    gpu: str
    start_index: int
    num_samples: int


class PrefixAwareVideoLatentTools(VideoLatentTools):
    """Keep generated target tokens whether references are prefixed or appended."""

    def clear_conditioning(self, latent_state: LatentState) -> LatentState:
        target_len = self.target_shape.token_count()
        total_len = latent_state.latent.shape[1]
        if total_len <= target_len:
            return super().clear_conditioning(latent_state)

        first_mask = latent_state.denoise_mask[:, :target_len].float().mean()
        last_mask = latent_state.denoise_mask[:, -target_len:].float().mean()
        target_slice = slice(0, target_len) if first_mask >= last_mask else slice(total_len - target_len, total_len)
        return LatentState(
            latent=latent_state.latent[:, target_slice],
            denoise_mask=torch.ones_like(latent_state.denoise_mask[:, target_slice]),
            positions=latent_state.positions[:, :, target_slice],
            clean_latent=latent_state.clean_latent[:, target_slice],
            attention_mask=None,
        )


class ControlValidationRunner(ValidationRunner):
    """Add Controlsft token layout, tuned VAE weights, and comparison output."""

    def __init__(
        self,
        *args: object,
        vae_checkpoint: Path | None = None,
        include_control_in_output: bool = True,
        target_output_paths: list[Path | None] | None = None,
        **kwargs: object,
    ) -> None:
        self._standalone_vae_checkpoint = vae_checkpoint
        self._standalone_include_control = include_control_in_output
        self._standalone_target_output_paths = target_output_paths
        self._standalone_target_output_index = 0
        super().__init__(*args, **kwargs)

    def _load_decoder_components(self) -> None:
        model_paths = (
            str(self._model_path)
            if self._standalone_vae_checkpoint is None
            else (str(self._model_path), str(self._standalone_vae_checkpoint))
        )
        self._vae_decoder = SingleGPUModelBuilder(
            model_path=model_paths,
            model_class_configurator=VideoDecoderConfigurator,
            model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
        ).build(device=torch.device("cpu"), dtype=torch.bfloat16)
        self._vae_decoder.requires_grad_(False)
        self._audio_decoder = None
        self._vocoder = None

    def _create_video_tools(self, width: int, height: int, num_frames: int) -> PrefixAwareVideoLatentTools:
        tools = super()._create_video_tools(width, height, num_frames)
        return PrefixAwareVideoLatentTools(
            patchifier=tools.patchifier,
            target_shape=tools.target_shape,
            fps=tools.fps,
            scale_factors=tools.scale_factors,
            causal_fix=tools.causal_fix,
        )

    @staticmethod
    def _apply_video_conditionings(
        state: LatentState,
        tools: VideoLatentTools,
        sample: ValidationSample,
        cached_media: CachedSampleMedia,
        device: torch.device,
    ) -> LatentState:
        state = ValidationRunner._apply_video_conditionings(state, tools, sample, cached_media, device)
        target_len = tools.target_shape.token_count()
        total_len = state.latent.shape[1]
        if total_len <= target_len:
            return state

        # Official inference appends references as [target, reference]. Controlsft
        # training uses [reference, target]. Avoid reordering if the installed
        # runner already applied the Controlsft layout.
        first_mask = state.denoise_mask[:, :target_len].float().mean()
        last_mask = state.denoise_mask[:, -target_len:].float().mean()
        if first_mask <= last_mask:
            return state
        order = torch.cat(
            [
                torch.arange(target_len, total_len, device=state.latent.device),
                torch.arange(target_len, device=state.latent.device),
            ]
        )
        attention_mask = state.attention_mask
        if attention_mask is not None:
            attention_mask = attention_mask.index_select(1, order).index_select(2, order)
        return LatentState(
            latent=state.latent.index_select(1, order),
            denoise_mask=state.denoise_mask.index_select(1, order),
            positions=state.positions.index_select(2, order),
            clean_latent=state.clean_latent.index_select(1, order),
            attention_mask=attention_mask,
        )

    def _generate_sample(self, *args: object, **kwargs: object) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        video, audio = super()._generate_sample(*args, **kwargs)
        target_output_path = None
        if self._standalone_target_output_paths is not None:
            target_output_path = self._standalone_target_output_paths[self._standalone_target_output_index]
            self._standalone_target_output_index += 1
        if video is not None and target_output_path is not None:
            save_video(
                video_tensor=video,
                output_path=target_output_path,
                fps=self._config.frame_rate,
                video_format="CFHW",
            )
            console.print(f"[green]Saved[/green] {target_output_path}")

        if video is None or not self._standalone_include_control:
            return video, audio

        sample = kwargs.get("sample")
        cached_media = kwargs.get("cached_media")
        device = kwargs.get("device")
        if not isinstance(sample, ValidationSample) or not isinstance(cached_media, CachedSampleMedia):
            return video, audio
        for index, condition in enumerate(sample.conditions):
            if condition.type != "reference" or index not in cached_media.conditions:
                continue
            media = cached_media.conditions[index]
            self._vae_decoder.to(device)
            latent = media.latent.to(device=device, dtype=torch.bfloat16)
            control = self._vae_decoder(latent)
            self._vae_decoder.to("cpu")
            control = ((control + 1.0) / 2.0).clamp(0.0, 1.0)[0].float().cpu()
            video = _concatenate_videos(control, video)
            break
        return video, audio


class OfficialControlRunner(OfficialStackMixin, ControlValidationRunner):
    """ControlValidationRunner + official guidance stack (CFG/STG/rescale/skip_step, batched passes)."""


class IGControlValidationRunner(InternalGuidanceMixin, ControlValidationRunner):
    """ControlValidationRunner + Internal Guidance.

    The mixin comes first so its ``_run_denoising`` wins; ``super()`` still
    reaches ControlValidationRunner for the tuned VAE, the PrefixAware latent
    tools and the with_control/target_only output handling.
    """


MANIFEST_SIZE = 56


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _parse_gpus(value: str) -> list[str]:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if not gpus:
        raise argparse.ArgumentTypeError("--gpus must contain at least one GPU id")
    if len(set(gpus)) != len(gpus):
        raise argparse.ArgumentTypeError("--gpus must not contain duplicate GPU ids")
    return gpus


def _build_parser() -> argparse.ArgumentParser:
    repo = _repo_root()
    workspace = repo.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=workspace / "checkpoints/ltx-2.3/ltx-2.3-22b-dev.safetensors",
    )
    parser.add_argument("--vae-checkpoint", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo / "datasets/controlsft_teacher_test1k/validation_fixed56.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory for with_control/ and target_only/ video outputs.",
    )
    parser.add_argument("--gpus", type=_parse_gpus, default=_parse_gpus("0,1,2,3,4,5,6,7"))
    parser.add_argument("--inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--stg-scale", type=float, default=1.0)
    parser.add_argument("--stg-blocks", type=int, nargs="+", default=[29])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-reference-control",
        action="store_true",
        help="Generate from text only; do not inject the precomputed control latent.",
    )
    parser.add_argument(
        "--target-only",
        action="store_true",
        help="Use control during generation but save only the generated target.",
    )
    parser.add_argument("--reference-downscale-factor", type=int, default=1)
    parser.add_argument("--reference-temporal-scale-factor", type=int, default=1)
    subset = parser.add_argument_group("sample subset")
    subset.add_argument(
        "--indices",
        type=int,
        nargs="+",
        help="validation_index values to render. Output names stay line-N, so a subset "
        "is directly comparable against a previous full run.",
    )
    subset.add_argument("--limit", type=int, help="Shorthand for --indices 0..N-1.")

    official = parser.add_argument_group("official guidance stack")
    official.add_argument(
        "--stack",
        choices=["trainer", "official"],
        default="trainer",
        help="trainer: current sequential loop (STG only, no CFG/rescale). "
        "official: batched cond/uncond/ptb passes with CFG + STG + rescale + skip_step.",
    )
    official.add_argument(
        "--negative-te",
        type=Path,
        help="Precomputed negative-prompt TE file (see scripts/encode_negative_te.py). "
        "Required for CFG (--guidance-scale > 1) on the official stack.",
    )
    official.add_argument("--rescale-scale", type=float, default=0.7, help="CFG rescale (official default 0.7).")
    official.add_argument("--modality-scale", type=float, default=1.0, help="Keep 1.0 for video-only generation.")
    official.add_argument("--skip-step", type=int, default=0, help="Reuse previous prediction every N+1 steps.")

    ig = parser.add_argument_group("internal guidance (IG)")
    ig.add_argument("--ig-scale", type=float, default=1.0, help="Extrapolation strength gamma. 1.0 disables IG.")
    ig.add_argument(
        "--sra-head",
        type=Path,
        help="Clean RGB SRA head. Either the standalone clean_rgb_sra_head_step_XXXXX.pt export "
        "(carries layer/depth metadata) or a full .safetensors checkpoint containing "
        "clean_rgb_sra_head.* (then --ig-layer is required). Defaults to --checkpoint.",
    )
    ig.add_argument("--ig-layer", type=int, help="One-based block number. Defaults to the head's metadata.")
    ig.add_argument(
        "--ig-calibration",
        choices=["auto", "none", "token_norm", "sample_norm"],
        default="auto",
        help="auto -> token_norm for cosine heads, none for smooth_l1 heads.",
    )
    ig.add_argument(
        "--ig-mode",
        choices=["parallel", "nested"],
        default="parallel",
        help="Trainer-stack composition mode. The official stack always combines internal with CFG/STG before rescale.",
    )
    ig.add_argument("--ig-sigma-low", type=float, default=0.0)
    ig.add_argument("--ig-sigma-high", type=float, default=1.0)

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-start", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-count", type=int, help=argparse.SUPPRESS)
    return parser


def _load_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as manifest:
        for line_number, line in enumerate(manifest, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on manifest line {line_number}: {exc}") from exc
    if len(rows) != MANIFEST_SIZE:
        raise ValueError(f"Fixed evaluation requires exactly {MANIFEST_SIZE} samples, got {len(rows)}")
    if [row.get("validation_index") for row in rows] != list(range(MANIFEST_SIZE)):
        raise ValueError(f"Manifest validation_index values must be exactly 0..{MANIFEST_SIZE - 1}")
    return rows


def _select_indices(args: argparse.Namespace) -> list[int]:
    """Which validation_index values this run renders.

    Subsetting never renumbers anything: items keep their manifest
    validation_index and their ``line-N.mp4`` name, so a 12-sample run drops
    straight into the same directory layout as a full 56-sample run.
    """
    if args.indices is not None and args.limit is not None:
        raise ValueError("Pass either --indices or --limit, not both")
    if args.indices is not None:
        chosen = sorted(set(args.indices))
    elif args.limit is not None:
        if not 1 <= args.limit <= MANIFEST_SIZE:
            raise ValueError(f"--limit must be in 1..{MANIFEST_SIZE}")
        chosen = list(range(args.limit))
    else:
        return list(range(MANIFEST_SIZE))
    out_of_range = [index for index in chosen if not 0 <= index < MANIFEST_SIZE]
    if out_of_range:
        raise ValueError(f"--indices out of range 0..{MANIFEST_SIZE - 1}: {out_of_range}")
    return chosen


def _resolve_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _latent_metadata(record: dict, manifest_path: Path) -> tuple[tuple[int, int, int, int], float]:
    shape = record.get("latent_shape")
    fps = float(record.get("fps", 24.0))
    if isinstance(shape, list) and len(shape) == 4:
        return tuple(int(value) for value in shape), fps

    latent_field = "latents" if "latents" in record else "reference_latents"
    if latent_field not in record:
        raise ValueError(f"Sample {record.get('validation_index')} has no latent shape or latent path")
    latent_path = _resolve_path(record[latent_field], manifest_path)
    data = torch.load(latent_path, map_location="cpu", weights_only=True)
    latent = data.get("latents")
    if not isinstance(latent, torch.Tensor) or latent.ndim != 4:
        raise ValueError(f"Expected [C,F,H,W] latent in {latent_path}")
    return tuple(latent.shape), float(data.get("fps", fps))


def _prepare_items(rows: list[dict], manifest_path: Path, output_dir: Path) -> list[Item]:
    items = []
    for index, record in enumerate(rows):
        if "conditions" not in record:
            raise ValueError(f"Manifest sample {index} is missing conditions")
        te_path = _resolve_path(record["conditions"], manifest_path)
        if "reference_latents" not in record:
            raise ValueError(f"Manifest sample {index} is missing reference_latents")
        reference_path = _resolve_path(record["reference_latents"], manifest_path)
        if not te_path.is_file():
            raise FileNotFoundError(te_path)
        if not reference_path.is_file():
            raise FileNotFoundError(reference_path)
        reference_data = torch.load(reference_path, map_location="cpu", weights_only=True)
        reference_latent = reference_data.get("latents")
        if not isinstance(reference_latent, torch.Tensor) or reference_latent.ndim != 4:
            raise ValueError(f"Expected [C,F,H,W] latent in {reference_path}")
        shape, frame_rate = _latent_metadata(record, manifest_path)
        channels, frames, height, width = shape
        if channels != 128:
            raise ValueError(f"Sample {index} has {channels} latent channels, expected 128")
        video_dims = (
            width * VIDEO_SPATIAL_COMPRESSION,
            height * VIDEO_SPATIAL_COMPRESSION,
            1 + (frames - 1) * VIDEO_TEMPORAL_COMPRESSION,
        )
        items.append(
            Item(
                manifest_index=index,
                sample_id=str(record.get("id", index)),
                te_path=te_path,
                reference_latent=reference_latent.unsqueeze(0).contiguous(),
                video_dims=video_dims,
                frame_rate=frame_rate,
                with_control_path=output_dir / "with_control" / f"line-{index}.mp4",
                target_only_path=output_dir / "target_only" / f"line-{index}.mp4",
            )
        )
    return items


@torch.inference_mode()
def _connect_te(
    te_path: Path, embeddings_processor: torch.nn.Module, device: torch.device
) -> CachedPromptEmbeddings:
    conditions = torch.load(te_path, map_location="cpu", weights_only=True)
    if "video_prompt_embeds" in conditions:
        video_features = conditions["video_prompt_embeds"]
        audio_features = conditions.get("audio_prompt_embeds")
    elif "prompt_embeds" in conditions:
        video_features = conditions["prompt_embeds"]
        audio_features = video_features
    else:
        raise ValueError(f"TE file has no video_prompt_embeds or prompt_embeds: {te_path}")
    if "prompt_attention_mask" not in conditions:
        raise ValueError(f"TE file has no prompt_attention_mask: {te_path}")

    video_features = video_features.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    audio_features = (
        audio_features.unsqueeze(0).to(device=device, dtype=torch.bfloat16) if audio_features is not None else None
    )
    mask = conditions["prompt_attention_mask"].unsqueeze(0).to(device)
    additive_mask = convert_to_additive_mask(mask, video_features.dtype)
    video_context, audio_context, _ = embeddings_processor.create_embeddings(
        video_features, audio_features, additive_mask
    )
    if audio_context is None:
        audio_context = video_context
    return CachedPromptEmbeddings(
        video_context_positive=video_context.cpu(),
        audio_context_positive=audio_context.cpu(),
        video_context_negative=None,
        audio_context_negative=None,
    )


def _load_transformer(base_checkpoint: Path, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    transformer = load_transformer(base_checkpoint, device=device, dtype=torch.bfloat16)
    state_dict = load_file(checkpoint, device="cpu")
    inference_state = {name: value for name, value in state_dict.items() if "clean_rgb_sra_head." not in name}
    incompatible = transformer.load_state_dict(inference_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint is incompatible with the official base model: "
            f"missing={sorted(incompatible.missing_keys)}, unexpected={sorted(incompatible.unexpected_keys)}"
        )
    transformer.requires_grad_(False)
    transformer.eval()
    return transformer


@torch.inference_mode()
def _run_worker(args: argparse.Namespace, rows: list[dict]) -> None:
    device = torch.device("cuda")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "target_only").mkdir(exist_ok=True)
    save_comparison = not args.no_reference_control and not args.target_only
    if save_comparison:
        (output_dir / "with_control").mkdir(exist_ok=True)
    chosen = set(_select_indices(args))
    all_items = _prepare_items(rows, args.manifest.resolve(), output_dir)
    selected = [item for item in all_items if item.manifest_index in chosen]
    items = selected[args.worker_start : args.worker_start + args.worker_count]
    if not args.overwrite:
        items = [
            item
            for item in items
            if not item.target_only_path.exists() or (save_comparison and not item.with_control_path.exists())
        ]
    if not items:
        return

    embeddings_processor = load_embeddings_processor(args.base_checkpoint.resolve(), device=device, dtype=torch.bfloat16)
    embeddings = [_connect_te(item.te_path, embeddings_processor, device) for item in items]
    if args.stack == "official" and args.negative_te is not None:
        negative = _connect_te(args.negative_te.resolve(), embeddings_processor, device)
        embeddings = [
            replace(
                emb,
                video_context_negative=negative.video_context_positive,
                audio_context_negative=negative.audio_context_positive,
            )
            for emb in embeddings
        ]
    del embeddings_processor
    torch.cuda.empty_cache()

    samples = [
        ValidationSample(
            prompt=f"precomputed:{item.sample_id}",
            video_dims=item.video_dims,
            seed=args.seed + item.manifest_index,
            conditions=(
                []
                if args.no_reference_control
                else [
                    ReferenceConditionConfig(
                        video="precomputed://control_latent",
                        downscale_factor=args.reference_downscale_factor,
                        temporal_scale_factor=args.reference_temporal_scale_factor,
                        include_in_output=False,
                    )
                ]
            ),
        )
        for item in items
    ]
    frame_rate = items[0].frame_rate
    config = ValidationConfig(
        samples=samples,
        video_dims=items[0].video_dims,
        frame_rate=frame_rate,
        seed=args.seed,
        inference_steps=args.inference_steps,
        interval=None,
        guidance_scale=args.guidance_scale,
        stg_scale=args.stg_scale,
        stg_blocks=args.stg_blocks,
        stg_mode="stg_v",
        generate_audio=False,
        generate_video=True,
        skip_initial_validation=True,
    )
    cached_media = [
        (
            CachedSampleMedia()
            if args.no_reference_control
            else CachedSampleMedia(conditions={0: CachedConditionMedia(latent=item.reference_latent)})
        )
        for item in items
    ]
    runner_kwargs = dict(
        config=config,
        model_path=args.base_checkpoint.resolve(),
        text_encoder_path=None,
        precomputed_embeddings=embeddings,
        precomputed_media=cached_media,
        tiled_video_decode=False,
        vae_checkpoint=args.vae_checkpoint.resolve() if args.vae_checkpoint is not None else None,
        include_control_in_output=save_comparison,
        target_output_paths=(
            [
                item.target_only_path if args.overwrite or not item.target_only_path.exists() else None
                for item in items
            ]
            if save_comparison
            else None
        ),
    )
    transformer = _load_transformer(args.base_checkpoint.resolve(), args.checkpoint.resolve(), device)

    if args.stack == "official":
        sra_head = None
        ig_settings = None
        if args.ig_scale != 1.0:
            head_path = (args.sra_head or args.checkpoint).resolve()
            sra_head, sra_metadata = load_sra_head(
                head_path, transformer, device=device, dtype=torch.bfloat16, hidden_layer=args.ig_layer
            )
            loss_type = str(sra_metadata.get("clean_rgb_sra_loss_type", "unknown"))
            calibration = (
                default_calibration_for(loss_type) if args.ig_calibration == "auto" else args.ig_calibration
            )
            ig_settings = IGSettings(
                guider=IGGuider(
                    scale=args.ig_scale,
                    sigma_low=args.ig_sigma_low,
                    sigma_high=args.ig_sigma_high,
                ),
                hidden_layer=int(args.ig_layer or sra_metadata["clean_rgb_sra_hidden_layer"]),
                calibration=calibration,
            )
        runner = OfficialControlRunner(
            **runner_kwargs,
            rescale_scale=args.rescale_scale,
            modality_scale=args.modality_scale,
            skip_step=args.skip_step,
            sra_head=sra_head,
            ig_settings=ig_settings,
        )
    elif args.ig_scale != 1.0:
        head_path = (args.sra_head or args.checkpoint).resolve()
        sra_head, sra_metadata = load_sra_head(
            head_path,
            transformer,
            device=device,
            dtype=torch.bfloat16,
            hidden_layer=args.ig_layer,
        )
        loss_type = str(sra_metadata.get("clean_rgb_sra_loss_type", "unknown"))
        calibration = default_calibration_for(loss_type) if args.ig_calibration == "auto" else args.ig_calibration
        if args.ig_calibration == "auto" and loss_type == "unknown":
            console.print(
                "[yellow]SRA head has no loss_type metadata; calibration defaults to 'none'. "
                "Pass --ig-calibration token_norm for a cosine-trained head.[/yellow]"
            )
        hidden_layer = int(args.ig_layer or sra_metadata["clean_rgb_sra_hidden_layer"])
        console.print(
            f"IG: layer={hidden_layer} gamma={args.ig_scale} mode={args.ig_mode} "
            f"calibration={calibration} (head loss={loss_type}) "
            f"interval=[{args.ig_sigma_low}, {args.ig_sigma_high}]"
        )
        runner = IGControlValidationRunner(
            **runner_kwargs,
            sra_head=sra_head,
            ig_guider=IGGuider(
                scale=args.ig_scale,
                sigma_low=args.ig_sigma_low,
                sigma_high=args.ig_sigma_high,
            ),
            ig_hidden_layer=hidden_layer,
            ig_calibration=calibration,
            ig_mode=args.ig_mode,
        )
    else:
        runner = ControlValidationRunner(**runner_kwargs)

    with tempfile.TemporaryDirectory(prefix=".fixed56_outputs_", dir=output_dir) as temp_dir:
        with TrainingProgress(enabled=False, total_steps=1) as progress:
            results = runner.run(
                transformer=transformer,
                step=0,
                output_dir=Path(temp_dir),
                device=device,
                progress=progress,
            )
        if len(results) != len(items):
            raise RuntimeError(f"Expected {len(items)} outputs, received {len(results)}")
        for local_index, generated_path in results:
            destination = (
                items[local_index].with_control_path if save_comparison else items[local_index].target_only_path
            )
            if destination.exists() and not args.overwrite:
                generated_path.unlink()
                continue
            if destination.exists():
                destination.unlink()
            shutil.move(str(generated_path), destination)
            console.print(f"[green]Saved[/green] {destination}")


def _partition(total: int, gpus: list[str]) -> list[Worker]:
    quotient, remainder = divmod(total, len(gpus))
    workers = []
    start = 0
    for index, gpu in enumerate(gpus):
        count = quotient + (1 if index < remainder else 0)
        if count:
            workers.append(Worker(gpu=gpu, start_index=start, num_samples=count))
            start += count
    return workers


def _worker_command(args: argparse.Namespace, worker: Worker) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--base-checkpoint",
        str(args.base_checkpoint.resolve()),
        "--manifest",
        str(args.manifest.resolve()),
        "--output-dir",
        str(args.output_dir.resolve()),
        "--inference-steps",
        str(args.inference_steps),
        "--guidance-scale",
        str(args.guidance_scale),
        "--stg-scale",
        str(args.stg_scale),
        "--stg-blocks",
        *(str(block) for block in args.stg_blocks),
        "--seed",
        str(args.seed),
        "--worker-start",
        str(worker.start_index),
        "--worker-count",
        str(worker.num_samples),
    ]
    if args.vae_checkpoint is not None:
        command.extend(["--vae-checkpoint", str(args.vae_checkpoint.resolve())])
    if args.overwrite:
        command.append("--overwrite")
    if args.no_reference_control:
        command.append("--no-reference-control")
    if args.target_only:
        command.append("--target-only")
    command.extend(["--reference-downscale-factor", str(args.reference_downscale_factor)])
    command.extend(["--reference-temporal-scale-factor", str(args.reference_temporal_scale_factor)])
    chosen = _select_indices(args)
    if len(chosen) != MANIFEST_SIZE:
        command.extend(["--indices", *(str(index) for index in chosen)])
    if args.stack != "trainer":
        command.extend(["--stack", args.stack])
        command.extend(["--rescale-scale", str(args.rescale_scale)])
        command.extend(["--modality-scale", str(args.modality_scale)])
        command.extend(["--skip-step", str(args.skip_step)])
        if args.negative_te is not None:
            command.extend(["--negative-te", str(args.negative_te.resolve())])
    if args.ig_scale != 1.0:
        command.extend(["--ig-scale", str(args.ig_scale)])
        if args.stack == "trainer":
            command.extend(["--ig-mode", args.ig_mode])
        command.extend(["--ig-calibration", args.ig_calibration])
        command.extend(["--ig-sigma-low", str(args.ig_sigma_low)])
        command.extend(["--ig-sigma-high", str(args.ig_sigma_high)])
        if args.sra_head is not None:
            command.extend(["--sra-head", str(args.sra_head.resolve())])
        if args.ig_layer is not None:
            command.extend(["--ig-layer", str(args.ig_layer)])
    return command


def _run_coordinator(args: argparse.Namespace, rows: list[dict]) -> None:
    chosen = _select_indices(args)
    workers = _partition(len(chosen), args.gpus)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    for worker in workers:
        span = chosen[worker.start_index : worker.start_index + worker.num_samples]
        print(f"GPU {worker.gpu}: validation_index {span[0]}..{span[-1]} ({len(span)} samples)")
        if args.dry_run:
            print("  " + " ".join(_worker_command(args, worker)))
    if args.dry_run:
        return

    processes = []
    started_at = time.monotonic()
    for worker in workers:
        end = worker.start_index + worker.num_samples - 1
        log_path = log_dir / f"gpu{worker.gpu}_{worker.start_index:02d}_{end:02d}.log"
        log_handle = log_path.open("w")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = worker.gpu
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            _worker_command(args, worker),
            cwd=_repo_root(),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((worker, process, log_handle, log_path))

    failures = []
    for worker, process, log_handle, log_path in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code:
            failures.append((worker, return_code, log_path))
    if failures:
        details = ", ".join(f"GPU {w.gpu} exit={code} log={path}" for w, code, path in failures)
        raise RuntimeError(f"Evaluation worker failure: {details}")

    expected_dirs = [args.output_dir / "target_only"]
    if not args.no_reference_control and not args.target_only:
        expected_dirs.append(args.output_dir / "with_control")
    missing = [
        output_dir / f"line-{index}.mp4"
        for output_dir in expected_dirs
        for index in chosen
        if not (output_dir / f"line-{index}.mp4").is_file()
    ]
    if missing:
        raise RuntimeError(f"Evaluation finished but {len(missing)} outputs are missing: {missing[:3]}")
    elapsed = (time.monotonic() - started_at) / 60
    mode = "text-only" if args.no_reference_control else "control-conditioned"
    output_count = len(chosen) * len(expected_dirs)
    print(f"Evaluation complete: {output_count}/{output_count} {mode} output files ({elapsed:.1f} min)")


def _concatenate_videos(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Concatenate [C,F,H,W] control and target videos horizontally."""
    if left.shape[2] != right.shape[2]:
        scale = right.shape[2] / left.shape[2]
        new_width = round(left.shape[3] * scale)
        channels, frames, height, width = left.shape
        resized = TF.resize(
            left.reshape(channels * frames, 1, height, width),
            [right.shape[2], new_width],
            interpolation=InterpolationMode.BICUBIC,
        )
        left = resized.clamp(0, 1).reshape(channels, frames, right.shape[2], new_width)
    if left.shape[1] < right.shape[1]:
        left = torch.cat([left, left[:, -1:].expand(-1, right.shape[1] - left.shape[1], -1, -1)], dim=1)
    elif right.shape[1] < left.shape[1]:
        right = torch.cat([right, right[:, -1:].expand(-1, left.shape[1] - right.shape[1], -1, -1)], dim=1)
    return torch.cat([left, right], dim=3)


def main() -> None:
    args = _build_parser().parse_args()
    for path in (args.checkpoint, args.base_checkpoint, args.manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.vae_checkpoint is not None and not args.vae_checkpoint.is_file():
        raise FileNotFoundError(args.vae_checkpoint)
    if args.inference_steps < 1 or args.guidance_scale < 1.0 or args.stg_scale < 0.0:
        raise ValueError("Invalid inference/guidance/STG values")
    if args.reference_downscale_factor < 1 or args.reference_temporal_scale_factor < 1:
        raise ValueError("Reference scale factors must be >= 1")
    if args.no_reference_control and args.target_only:
        raise ValueError("--target-only is redundant with --no-reference-control")
    if args.sra_head is not None and not args.sra_head.is_file():
        raise FileNotFoundError(args.sra_head)
    if not 0.0 <= args.ig_sigma_low <= args.ig_sigma_high <= 1.0:
        raise ValueError("--ig-sigma-low/--ig-sigma-high must satisfy 0 <= low <= high <= 1")
    if args.ig_scale != 1.0 and (args.sra_head or args.checkpoint).suffix == ".safetensors" and args.ig_layer is None:
        raise ValueError("A .safetensors SRA head carries no metadata; pass --ig-layer explicitly")
    if args.stack == "official":
        if args.ig_mode != "parallel":
            console.print(
                "[yellow]--ig-mode applies only to --stack trainer; "
                "the official stack always combines Internal Guidance before rescale.[/yellow]"
            )
        if args.guidance_scale != 1.0 and args.negative_te is None:
            raise ValueError("--guidance-scale > 1 on the official stack requires --negative-te")
        if args.negative_te is not None and not args.negative_te.is_file():
            raise FileNotFoundError(args.negative_te)
    elif args.negative_te is not None:
        raise ValueError("--negative-te only applies to --stack official")
    _select_indices(args)
    rows = _load_manifest(args.manifest)
    if args.worker_start is not None or args.worker_count is not None:
        if args.worker_start is None or args.worker_count is None:
            raise ValueError("Internal worker arguments must be provided together")
        _run_worker(args, rows)
    else:
        _run_coordinator(args, rows)


if __name__ == "__main__":
    main()
