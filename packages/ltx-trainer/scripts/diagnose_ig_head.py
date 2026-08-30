#!/usr/bin/env python

"""Offline diagnostic for the Clean RGB SRA head as an Internal Guidance source.

Runs no sampling loop. For each clip it noises the ground-truth target latent to
a grid of sigmas, takes one forward, and compares the block-l prediction against
the final-layer prediction and the truth. Use it to compare a ``smooth_l1`` head
against a ``cosine`` head before spending GPU hours on video generation.

Reuses the manifest/TE loading of ``infer_part16_precomputed.py`` and the state
building of ``ValidationRunner``.

Example:

    uv run python scripts/diagnose_ig_head.py \\
      --base-checkpoint /path/to/ltx-2.3-22b-dev.safetensors \\
      --trained-checkpoint /path/to/model_weights_step_01000.safetensors \\
      --sra-head /path/to/clean_rgb_sra_head_step_01000.pt \\
      --manifest-path /path/to/val.jsonl --num-samples 32 \\
      --report-path outputs/ig_probe_l1.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from ltx_trainer.config import ReferenceConditionConfig, ValidationConfig, ValidationSample
from ltx_trainer.ig_validation_runner import IGValidationRunner, probe_intermediate
from ltx_trainer.internal_guidance import IGGuider, default_calibration_for, load_sra_head
from ltx_trainer.model_loader import load_embeddings_processor
from ltx_trainer.validation_runner import CachedConditionMedia, CachedPromptEmbeddings, CachedSampleMedia

from infer_part16_precomputed import (  # noqa: E402
    _connect_te,
    _load_manifest_slice,
    _load_reference_item,
    _resolve_manifest_path,
    load_finetuned_transformer,
)

console = Console()
DEFAULT_SIGMAS = [0.95, 0.8, 0.6, 0.45, 0.3, 0.2, 0.1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe the SRA head as an Internal Guidance source.")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--trained-checkpoint", type=Path, required=True)
    parser.add_argument("--sra-head", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--video-field", default="video", help="Manifest field holding the clean target latents.")
    parser.add_argument("--reference-field", default="reference_latents")
    parser.add_argument("--te-field", default="conditions")
    parser.add_argument("--frame-rate", type=float)
    parser.add_argument("--reference-downscale-factor", type=int, default=1)
    parser.add_argument("--reference-temporal-scale-factor", type=int, default=1)
    parser.add_argument("--ig-layer", type=int, help="Defaults to the head's metadata.")
    parser.add_argument(
        "--ig-calibration",
        choices=["auto", "none", "token_norm", "sample_norm"],
        default="auto",
    )
    parser.add_argument("--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")

    manifest_path = args.manifest_path.resolve()
    base_checkpoint = args.base_checkpoint.resolve()
    records = _load_manifest_slice(manifest_path, args.start_index, args.num_samples)

    scratch = Path("/tmp/ig_probe")
    scratch.mkdir(parents=True, exist_ok=True)
    items = [
        _load_reference_item(
            manifest_index=index,
            record=record,
            manifest_path=manifest_path,
            reference_field=args.reference_field,
            te_field=args.te_field,
            output_dir=scratch,
            spatial_factor=args.reference_downscale_factor,
            temporal_factor=args.reference_temporal_scale_factor,
            frame_rate_override=args.frame_rate,
        )
        for index, record in records
    ]
    target_latents = []
    for (_, record), item in zip(records, items, strict=True):
        if args.video_field not in record:
            raise ValueError(f"Manifest sample {item.manifest_index} has no '{args.video_field}' field")
        payload = torch.load(_resolve_manifest_path(record[args.video_field], manifest_path), map_location="cpu")
        target_latents.append(payload["latents"].unsqueeze(0).contiguous())

    frame_rate = args.frame_rate if args.frame_rate is not None else items[0].frame_rate

    embeddings_processor = load_embeddings_processor(base_checkpoint, device=device, dtype=torch.bfloat16)
    cached_embeddings = []
    for item in items:
        video_context, audio_context = _connect_te(item.te_path, embeddings_processor, device)
        cached_embeddings.append(
            CachedPromptEmbeddings(video_context_positive=video_context, audio_context_positive=audio_context)
        )
    del embeddings_processor
    torch.cuda.empty_cache()

    samples = [
        ValidationSample(
            prompt=f"probe:{item.sample_id}",
            video_dims=item.target_dims,
            seed=args.seed + item.manifest_index,
            conditions=[
                ReferenceConditionConfig(
                    video="precomputed://signal_latent",
                    downscale_factor=args.reference_downscale_factor,
                    temporal_scale_factor=args.reference_temporal_scale_factor,
                )
            ],
        )
        for item in items
    ]
    cached_media = [
        CachedSampleMedia(conditions={0: CachedConditionMedia(latent=item.reference_latent)}) for item in items
    ]
    config = ValidationConfig(
        samples=samples,
        video_dims=items[0].target_dims,
        frame_rate=frame_rate,
        seed=args.seed,
        inference_steps=1,
        interval=None,
        guidance_scale=1.0,
        stg_scale=0.0,
        stg_blocks=[],
        stg_mode="stg_v",
        generate_audio=False,
        generate_video=True,
        skip_initial_validation=True,
    )

    transformer = load_finetuned_transformer(
        base_checkpoint=base_checkpoint,
        trained_checkpoint=args.trained_checkpoint.resolve(),
        device=device,
    )
    sra_head, metadata = load_sra_head(
        args.sra_head.resolve(), transformer, device=device, dtype=torch.bfloat16, hidden_layer=args.ig_layer
    )
    loss_type = str(metadata.get("clean_rgb_sra_loss_type", "unknown"))
    calibration = default_calibration_for(loss_type) if args.ig_calibration == "auto" else args.ig_calibration
    hidden_layer = int(args.ig_layer or metadata["clean_rgb_sra_hidden_layer"])
    console.print(f"Head: layer={hidden_layer} loss={loss_type} calibration={calibration}")

    runner = IGValidationRunner(
        config=config,
        model_path=base_checkpoint,
        text_encoder_path=None,
        precomputed_embeddings=cached_embeddings,
        precomputed_media=cached_media,
        tiled_video_decode=False,
        sra_head=sra_head,
        ig_guider=IGGuider(scale=1.0),
        ig_hidden_layer=hidden_layer,
        ig_calibration=calibration,
    )

    by_sigma: dict[float, list[dict[str, float]]] = defaultdict(list)
    for sample, embeddings, media, latent in zip(samples, cached_embeddings, cached_media, target_latents, strict=True):
        rows = probe_intermediate(
            runner=runner,
            transformer=transformer,
            sample=sample,
            cached_embeddings=embeddings,
            cached_media=media,
            clean_target_latent=latent,
            sigmas=list(args.sigmas),
            device=device,
            calibration=calibration,
        )
        for row in rows:
            by_sigma[row["sigma"]].append(row)

    table = Table(title=f"IG probe (layer {hidden_layer}, {loss_type}, calibration={calibration})")
    for column in ("sigma", "mse_weak", "mse_strong", "weak/strong", "cos(w,s)", "cos(delta,resid)", "norm_ratio"):
        table.add_column(column, justify="right")
    summary = []
    for sigma in sorted(by_sigma, reverse=True):
        rows = by_sigma[sigma]
        mean = {key: statistics.fmean(row[key] for row in rows) for key in rows[0]}
        mean["mse_ratio"] = mean["mse_weak"] / max(mean["mse_strong"], 1e-9)
        summary.append(mean)
        table.add_row(
            f"{sigma:.2f}",
            f"{mean['mse_weak']:.4f}",
            f"{mean['mse_strong']:.4f}",
            f"{mean['mse_ratio']:.2f}x",
            f"{mean['cos_weak_strong']:.3f}",
            f"{mean['cos_delta_residual']:+.3f}",
            f"{mean['norm_ratio']:.3f}",
        )
    console.print(table)
    console.print(
        "[dim]Go/no-go: weak/strong roughly 1.5x-10x (bad but not broken), cos(w,s) clearly below 1.0, "
        "cos(delta,resid) positive. Apply IG over the sigma range where the last column holds.[/dim]"
    )

    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(
            json.dumps(
                {
                    "hidden_layer": hidden_layer,
                    "loss_type": loss_type,
                    "calibration": calibration,
                    "num_samples": len(samples),
                    "per_sigma": summary,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        console.print(f"[green]Wrote[/green] {args.report_path}")


if __name__ == "__main__":
    main()
