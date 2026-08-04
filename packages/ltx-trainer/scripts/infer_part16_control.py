#!/usr/bin/env python

"""Run one-stage Part16-controlled inference from a full fine-tuning checkpoint.

The official LTX-2.3 checkpoint supplies model metadata plus the Gemma/VAE
components. ``--trained-checkpoint`` is loaded as a transformer-only weight
overlay. Training-only ``clean_rgb_sra_head`` tensors are intentionally ignored.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import torch
from rich.console import Console
from safetensors.torch import load_file

from ltx_trainer.config import ReferenceConditionConfig, ValidationConfig, ValidationSample
from ltx_trainer.model_loader import load_transformer
from ltx_trainer.progress import TrainingProgress
from ltx_trainer.validation_runner import ValidationRunner

DEFAULT_NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"
console = Console()


def load_finetuned_transformer(
    *,
    base_checkpoint: Path,
    trained_checkpoint: Path,
    device: torch.device,
) -> torch.nn.Module:
    """Build LTX-2.3 from the official checkpoint and apply full-tune weights."""
    transformer = load_transformer(base_checkpoint, device=device, dtype=torch.bfloat16)
    state_dict = load_file(trained_checkpoint, device="cpu")
    inference_state = {name: value for name, value in state_dict.items() if "clean_rgb_sra_head." not in name}

    incompatible = transformer.load_state_dict(inference_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Fine-tuned checkpoint is incompatible with the official base model: "
            f"missing={sorted(incompatible.missing_keys)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )

    del state_dict, inference_state
    transformer.requires_grad_(False)
    transformer.eval()
    return transformer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a video conditioned on one Part16 control video.")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--trained-checkpoint", type=Path, required=True)
    parser.add_argument("--gemma-root", type=Path, required=True)
    parser.add_argument("--control-video", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--stg-scale", type=float, default=1.0)
    parser.add_argument("--stg-blocks", type=int, nargs="+", default=[29])
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reference-downscale-factor",
        type=int,
        default=1,
        help="Must match the reference preprocessing used for training.",
    )
    parser.add_argument(
        "--reference-temporal-scale-factor",
        type=int,
        default=1,
        help="Must match the reference preprocessing used for training.",
    )
    parser.add_argument(
        "--include-control",
        action="store_true",
        help="Save Part16 and generated video side by side.",
    )
    parser.add_argument("--disable-progress-bars", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    for name in ("base_checkpoint", "trained_checkpoint", "gemma_root", "control_video"):
        path = getattr(args, name)
        if not path.exists():
            parser.error(f"--{name.replace('_', '-')} does not exist: {path}")
    if not args.base_checkpoint.is_file() or not args.trained_checkpoint.is_file():
        parser.error("--base-checkpoint and --trained-checkpoint must be files")
    if not args.gemma_root.is_dir():
        parser.error("--gemma-root must be a directory")
    if not args.control_video.is_file():
        parser.error("--control-video must be a file")
    if args.width % 32 != 0 or args.height % 32 != 0:
        parser.error("--width and --height must be divisible by 32")
    if args.num_frames % 8 != 1:
        parser.error("--num-frames must satisfy num_frames % 8 == 1")
    if args.reference_downscale_factor < 1 or args.reference_temporal_scale_factor < 1:
        parser.error("reference scale factors must be at least 1")
    reference_width = args.width // args.reference_downscale_factor
    reference_height = args.height // args.reference_downscale_factor
    if reference_width % 32 != 0 or reference_height % 32 != 0:
        parser.error("scaled reference width and height must be divisible by 32")
    expected_suffix = ".png" if args.num_frames == 1 else ".mp4"
    if args.output_path.suffix.lower() != expected_suffix:
        parser.error(f"--output-path must end in {expected_suffix} for --num-frames {args.num_frames}")


@torch.inference_mode()
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(args, parser)

    if not torch.cuda.is_available():
        parser.error("CUDA is required for LTX-2.3 inference")
    device = torch.device("cuda")

    sample = ValidationSample(
        prompt=args.prompt,
        conditions=[
            ReferenceConditionConfig(
                video=str(args.control_video.resolve()),
                downscale_factor=args.reference_downscale_factor,
                temporal_scale_factor=args.reference_temporal_scale_factor,
                include_in_output=args.include_control,
            )
        ],
    )
    validation_config = ValidationConfig(
        samples=[sample],
        negative_prompt=args.negative_prompt,
        video_dims=(args.width, args.height, args.num_frames),
        frame_rate=args.frame_rate,
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

    base_checkpoint = args.base_checkpoint.resolve()
    runner = ValidationRunner(
        config=validation_config,
        model_path=base_checkpoint,
        text_encoder_path=args.gemma_root.resolve(),
    )
    transformer = load_finetuned_transformer(
        base_checkpoint=base_checkpoint,
        trained_checkpoint=args.trained_checkpoint.resolve(),
        device=device,
    )

    output_path = args.output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".part16_infer_", dir=output_path.parent) as temp_dir:
        with TrainingProgress(enabled=not args.disable_progress_bars, total_steps=1) as progress:
            results = runner.run(
                transformer=transformer,
                step=0,
                output_dir=Path(temp_dir),
                device=device,
                progress=progress,
            )
        if len(results) != 1:
            raise RuntimeError(f"Expected one generated video, received {len(results)} outputs")
        shutil.move(str(results[0][1]), output_path)

    console.print(f"[green]Saved Part16-controlled output to {output_path}[/green]")


if __name__ == "__main__":
    main()
