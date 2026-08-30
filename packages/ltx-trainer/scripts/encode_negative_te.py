#!/usr/bin/env python

"""Encode a negative prompt into the project's precomputed-TE file format.

One-time step for enabling CFG on the precomputed-embedding inference path:
loads the Gemma text encoder, encodes the (default: official
``DEFAULT_NEGATIVE_PROMPT``) text exactly like ``process_captions.py`` encodes
positive captions, and saves a ``.pt`` with ``video_prompt_embeds`` /
``audio_prompt_embeds`` / ``prompt_attention_mask`` — the same keys
``_connect_te`` expects, so one file serves every sample.

Example:

    uv run python scripts/encode_negative_te.py \\
      --base-checkpoint /path/to/ltx-2.3-22b-dev.safetensors \\
      --text-encoder-path /path/to/gemma \\
      --output /workspace/te/negative_default.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from rich.console import Console

from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT
from ltx_trainer.model_loader import load_embeddings_processor, load_text_encoder

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encode a negative prompt into a precomputed TE file.")
    parser.add_argument("--base-checkpoint", type=Path, required=True, help="LTX checkpoint (for the connector).")
    parser.add_argument("--text-encoder-path", type=Path, required=True, help="Gemma model directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output .pt path.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Negative prompt text. Defaults to the official DEFAULT_NEGATIVE_PROMPT.",
    )
    parser.add_argument("--load-in-8bit", action="store_true", help="Load Gemma in 8-bit (less VRAM).")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    prompt = args.prompt if args.prompt is not None else DEFAULT_NEGATIVE_PROMPT
    console.print(f"Encoding negative prompt ({len(prompt)} chars):")
    console.print(f"[dim]{prompt[:200]}{'...' if len(prompt) > 200 else ''}[/dim]")

    with console.status("[bold]Loading Gemma text encoder...", spinner="dots"):
        text_encoder = load_text_encoder(
            args.text_encoder_path,
            device=device,
            dtype=torch.bfloat16,
            load_in_8bit=args.load_in_8bit,
        )
        embeddings_processor = load_embeddings_processor(
            args.base_checkpoint.resolve(), device=device, dtype=torch.bfloat16
        )

    # Mirror process_captions.py exactly: encode -> feature_extractor, save
    # features *before* the connector (the connector runs at load time in
    # _connect_te via embeddings_processor.create_embeddings).
    with torch.inference_mode():
        encoded = text_encoder.encode([prompt], padding_side="left")
        hidden_states, prompt_attention_mask = encoded[0]
        video_prompt_embeds, audio_prompt_embeds = embeddings_processor.feature_extractor(
            hidden_states, prompt_attention_mask, "left"
        )

    payload = {
        "video_prompt_embeds": video_prompt_embeds[0].cpu().contiguous(),
        "prompt_attention_mask": prompt_attention_mask[0].cpu().contiguous(),
        "negative_prompt_text": prompt,
    }
    if audio_prompt_embeds is not None:
        payload["audio_prompt_embeds"] = audio_prompt_embeds[0].cpu().contiguous()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.rename(args.output)
    console.print(
        f"[green]Wrote[/green] {args.output}  "
        f"(video_prompt_embeds {tuple(payload['video_prompt_embeds'].shape)}, "
        f"mask {tuple(payload['prompt_attention_mask'].shape)})"
    )


if __name__ == "__main__":
    main()
