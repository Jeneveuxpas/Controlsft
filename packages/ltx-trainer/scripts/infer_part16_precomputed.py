#!/usr/bin/env python

"""Batch Part16 inference directly from precomputed signal latents and TE features."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from rich.console import Console
from safetensors.torch import load_file

from ltx_core.text_encoders.gemma import convert_to_additive_mask
from ltx_trainer.config import ReferenceConditionConfig, ValidationConfig, ValidationSample
from ltx_trainer.model_loader import load_embeddings_processor, load_transformer
from ltx_trainer.progress import TrainingProgress
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
class PrecomputedItem:
    manifest_index: int
    sample_id: str
    reference_path: Path
    te_path: Path
    reference_latent: torch.Tensor
    target_dims: tuple[int, int, int]
    frame_rate: float
    output_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-generate videos from manifest signal_latent/reference_latents and te/conditions files."
    )
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--trained-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        help="Optional tuned VAE state dict used to override the base checkpoint's video decoder.",
    )
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--reference-field", default="reference_latents")
    parser.add_argument("--te-field", default="conditions")
    parser.add_argument("--negative-te", type=Path)
    parser.add_argument("--frame-rate", type=float, help="Override latent metadata FPS for every sample.")
    parser.add_argument("--inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--stg-scale", type=float, default=1.0)
    parser.add_argument("--stg-blocks", type=int, nargs="+", default=[29])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-downscale-factor", type=int, default=1)
    parser.add_argument("--reference-temporal-scale-factor", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-reference-control",
        action="store_true",
        help="Generate without injecting reference/Part16 control latents.",
    )
    parser.add_argument("--disable-progress-bars", action="store_true")
    parser.add_argument(
        "--tiled-video-decode",
        action="store_true",
        help="Use tiled VAE decoding. By default this script decodes each video as one complete tensor.",
    )
    return parser


def _resolve_manifest_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _safe_sample_id(value: object, fallback: str) -> str:
    text = str(value).strip() if value is not None else fallback
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return text or fallback


def _load_manifest_slice(manifest_path: Path, start_index: int, num_samples: int) -> list[tuple[int, dict]]:
    selected: list[tuple[int, dict]] = []
    with manifest_path.open(encoding="utf-8") as manifest:
        sample_index = 0
        for line_number, line in enumerate(manifest, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on manifest line {line_number}: {exc}") from exc
            if sample_index >= start_index and len(selected) < num_samples:
                selected.append((sample_index, record))
            sample_index += 1
            if len(selected) == num_samples:
                break
    if not selected:
        raise ValueError(f"No manifest samples found from start index {start_index}")
    return selected


def _load_reference_item(
    *,
    manifest_index: int,
    record: dict,
    manifest_path: Path,
    reference_field: str,
    te_field: str,
    output_dir: Path,
    spatial_factor: int,
    temporal_factor: int,
    frame_rate_override: float | None,
) -> PrecomputedItem:
    for field in (reference_field, te_field):
        if field not in record:
            raise ValueError(f"Manifest sample {manifest_index} is missing field '{field}'")

    reference_path = _resolve_manifest_path(record[reference_field], manifest_path)
    te_path = _resolve_manifest_path(record[te_field], manifest_path)
    if not reference_path.is_file() or not te_path.is_file():
        raise FileNotFoundError(
            f"Manifest sample {manifest_index} has missing files: reference={reference_path}, te={te_path}"
        )

    reference_data = torch.load(reference_path, map_location="cpu", weights_only=True)
    reference_latent = reference_data.get("latents")
    if not isinstance(reference_latent, torch.Tensor) or reference_latent.ndim != 4:
        raise ValueError(
            f"Expected {reference_path}['latents'] in [C,F,H,W] format, got {getattr(reference_latent, 'shape', None)}"
        )
    channels, ref_frames, ref_height, ref_width = reference_latent.shape
    if channels != 128:
        raise ValueError(f"Expected 128 latent channels in {reference_path}, got {channels}")

    target_latent_frames = 1 + (ref_frames - 1) * temporal_factor
    target_latent_height = ref_height * spatial_factor
    target_latent_width = ref_width * spatial_factor
    target_dims = (
        target_latent_width * VIDEO_SPATIAL_COMPRESSION,
        target_latent_height * VIDEO_SPATIAL_COMPRESSION,
        1 + (target_latent_frames - 1) * VIDEO_TEMPORAL_COMPRESSION,
    )
    metadata_fps = float(reference_data.get("fps", 24.0)) * temporal_factor
    frame_rate = frame_rate_override if frame_rate_override is not None else metadata_fps

    fallback_id = reference_path.stem
    sample_id = _safe_sample_id(record.get("id"), fallback_id)
    extension = ".png" if target_dims[2] == 1 else ".mp4"
    # Keep generated files independent of long dataset IDs.
    # The line number is stable for a fixed manifest.
    output_path = output_dir / f"line-{manifest_index}{extension}"
    return PrecomputedItem(
        manifest_index=manifest_index,
        sample_id=sample_id,
        reference_path=reference_path,
        te_path=te_path,
        reference_latent=reference_latent.unsqueeze(0).contiguous(),
        target_dims=target_dims,
        frame_rate=frame_rate,
        output_path=output_path,
    )


@torch.inference_mode()
def _connect_te(
    te_path: Path,
    embeddings_processor: torch.nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    return video_context.cpu(), audio_context.cpu()


def load_finetuned_transformer(
    *, base_checkpoint: Path, trained_checkpoint: Path, device: torch.device
) -> torch.nn.Module:
    transformer = load_transformer(base_checkpoint, device=device, dtype=torch.bfloat16)
    state_dict = load_file(trained_checkpoint, device="cpu")
    training_only_markers = ("clean_rgb_sra_head.", "representation_distillation_head.")
    inference_state = {
        name: value
        for name, value in state_dict.items()
        if not any(marker in name for marker in training_only_markers)
    }
    incompatible = transformer.load_state_dict(inference_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Fine-tuned checkpoint is incompatible with the official base model: "
            f"missing={sorted(incompatible.missing_keys)}, unexpected={sorted(incompatible.unexpected_keys)}"
        )
    del state_dict, inference_state
    transformer.requires_grad_(False)
    transformer.eval()
    return transformer


@torch.inference_mode()
def main() -> None:  # noqa: PLR0912, PLR0915
    parser = build_parser()
    args = parser.parse_args()
    if args.start_index < 0 or args.num_samples < 1:
        parser.error("--start-index must be >= 0 and --num-samples must be >= 1")
    if args.inference_steps < 1 or args.guidance_scale < 1.0 or args.stg_scale < 0.0:
        parser.error("invalid inference/guidance/STG values")
    if args.reference_downscale_factor < 1 or args.reference_temporal_scale_factor < 1:
        parser.error("reference scale factors must be >= 1")
    if args.guidance_scale > 1.0 and args.negative_te is None:
        parser.error("--negative-te is required when --guidance-scale is greater than 1")
    for path in (args.base_checkpoint, args.trained_checkpoint, args.manifest_path):
        if not path.is_file():
            parser.error(f"File does not exist: {path}")
    if args.vae_checkpoint is not None and not args.vae_checkpoint.is_file():
        parser.error(f"File does not exist: {args.vae_checkpoint}")
    if args.negative_te is not None and not args.negative_te.is_file():
        parser.error(f"Negative TE file does not exist: {args.negative_te}")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for LTX-2.3 inference")

    base_checkpoint = args.base_checkpoint.resolve()
    manifest_path = args.manifest_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _load_manifest_slice(manifest_path, args.start_index, args.num_samples)
    items = [
        _load_reference_item(
            manifest_index=index,
            record=record,
            manifest_path=manifest_path,
            reference_field=args.reference_field,
            te_field=args.te_field,
            output_dir=output_dir,
            spatial_factor=args.reference_downscale_factor,
            temporal_factor=args.reference_temporal_scale_factor,
            frame_rate_override=args.frame_rate,
        )
        for index, record in records
    ]
    if not args.overwrite:
        skipped = [item for item in items if item.output_path.exists()]
        items = [item for item in items if not item.output_path.exists()]
        if skipped:
            console.print(f"[yellow]Skipping {len(skipped)} existing outputs; pass --overwrite to regenerate.[/yellow]")
    if not items:
        console.print("[green]All selected outputs already exist.[/green]")
        return

    frame_rate = args.frame_rate if args.frame_rate is not None else items[0].frame_rate
    mismatched_fps = [item for item in items if abs(item.frame_rate - frame_rate) > 0.05]
    if mismatched_fps:
        console.print(
            f"[yellow]Warning: {len(mismatched_fps)} samples have FPS different from {frame_rate:.3f}; "
            "use separate runs with --frame-rate for exact positional alignment.[/yellow]"
        )

    device = torch.device("cuda")
    console.print(f"Connecting precomputed TE for {len(items)} samples...")
    embeddings_processor = load_embeddings_processor(base_checkpoint, device=device, dtype=torch.bfloat16)
    negative_context = (
        _connect_te(args.negative_te.resolve(), embeddings_processor, device) if args.negative_te is not None else None
    )
    cached_embeddings: list[CachedPromptEmbeddings] = []
    for item in items:
        video_context, audio_context = _connect_te(item.te_path, embeddings_processor, device)
        cached_embeddings.append(
            CachedPromptEmbeddings(
                video_context_positive=video_context,
                audio_context_positive=audio_context,
                video_context_negative=negative_context[0] if negative_context is not None else None,
                audio_context_negative=negative_context[1] if negative_context is not None else None,
            )
        )
    del embeddings_processor
    torch.cuda.empty_cache()

    samples = [
        ValidationSample(
            prompt=f"precomputed:{item.sample_id}",
            video_dims=item.target_dims,
            seed=args.seed + item.manifest_index,
            conditions=(
                []
                if args.no_reference_control
                else [
                    ReferenceConditionConfig(
                        video="precomputed://signal_latent",
                        downscale_factor=args.reference_downscale_factor,
                        temporal_scale_factor=args.reference_temporal_scale_factor,
                        include_in_output=True,
                    )
                ]
            ),
        )
        for item in items
    ]
    cached_media = [
        (
            CachedSampleMedia()
            if args.no_reference_control
            else CachedSampleMedia(conditions={0: CachedConditionMedia(latent=item.reference_latent)})
        )
        for item in items
    ]
    config = ValidationConfig(
        samples=samples,
        video_dims=items[0].target_dims,
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
    runner = ValidationRunner(
        config=config,
        model_path=base_checkpoint,
        text_encoder_path=None,
        precomputed_embeddings=cached_embeddings,
        precomputed_media=cached_media,
        tiled_video_decode=args.tiled_video_decode,
        video_vae_checkpoint=args.vae_checkpoint.resolve() if args.vae_checkpoint is not None else None,
    )
    transformer = load_finetuned_transformer(
        base_checkpoint=base_checkpoint,
        trained_checkpoint=args.trained_checkpoint.resolve(),
        device=device,
    )

    with tempfile.TemporaryDirectory(prefix=".part16_precomputed_", dir=output_dir) as temp_dir:
        with TrainingProgress(enabled=not args.disable_progress_bars, total_steps=1) as progress:
            results = runner.run(
                transformer=transformer,
                step=0,
                output_dir=Path(temp_dir),
                device=device,
                progress=progress,
            )
        if len(results) != len(items):
            raise RuntimeError(f"Expected {len(items)} outputs, received {len(results)}")
        for sample_index, generated_path in results:
            destination = items[sample_index].output_path
            if destination.exists() and args.overwrite:
                destination.unlink()
            shutil.move(str(generated_path), destination)
            console.print(f"[green]Saved[/green] {destination}")


if __name__ == "__main__":
    main()
