#!/usr/bin/env python

"""Consolidate a PyTorch Distributed Checkpoint (FSDP/DCP) into safetensors.

The input must be a DCP directory containing ``.metadata`` and ``*.distcp``
files, for example an output named ``pytorch_model_fsdp_0``.  The command is
intentionally single-process: it reconstructs the full state dict on CPU.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import types
from pathlib import Path


def _install_pathlib_pickle_compatibility() -> None:
    """Read DCP metadata saved by Python 3.12+ from Python 3.11.

    Python 3.12 moved ``PosixPath``'s pickle location to ``pathlib._local``.
    The training image currently uses Python 3.11, where that module does not
    exist.  Mapping it back to the equivalent public classes is sufficient for
    DCP's metadata pickle and does not affect model tensors.
    """

    if "pathlib._local" in sys.modules:
        return
    compatibility_module = types.ModuleType("pathlib._local")
    compatibility_module.PosixPath = pathlib.PosixPath
    compatibility_module.WindowsPath = pathlib.WindowsPath
    sys.modules["pathlib._local"] = compatibility_module


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="DCP/FSDP shard directory containing .metadata and *.distcp files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination single-file .safetensors checkpoint. Refuses to overwrite.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary consolidated .pt state dict for debugging.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    input_dir = args.input_dir.resolve()
    output = args.output.resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"DCP input directory does not exist: {input_dir}")
    if not (input_dir / ".metadata").is_file():
        raise FileNotFoundError(f"DCP input directory has no .metadata file: {input_dir}")
    if not any(input_dir.glob("*.distcp")):
        raise FileNotFoundError(f"DCP input directory has no *.distcp shard files: {input_dir}")
    if output.suffix != ".safetensors":
        raise ValueError(f"Output must use the .safetensors extension: {output}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_state_dict = output.with_suffix(".state_dict.pt.tmp")
    if temporary_state_dict.exists():
        raise FileExistsError(
            f"Found stale temporary file: {temporary_state_dict}. "
            "Inspect or remove it before retrying."
        )

    _install_pathlib_pickle_compatibility()
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    print(f"Consolidating DCP shards from: {input_dir}", flush=True)
    dcp_to_torch_save(input_dir, temporary_state_dict)

    print(f"Loading consolidated state dict: {temporary_state_dict}", flush=True)
    state_dict = torch.load(temporary_state_dict, map_location="cpu", weights_only=True)
    if set(state_dict) != {"model"} or not isinstance(state_dict["model"], dict):
        raise RuntimeError(f"Expected a top-level 'model' state dict, got keys: {list(state_dict)}")

    model_state_dict = state_dict["model"]
    if not model_state_dict:
        raise RuntimeError("The consolidated model state dict is empty")
    non_tensors = [name for name, value in model_state_dict.items() if not isinstance(value, torch.Tensor)]
    if non_tensors:
        raise RuntimeError(f"Model state dict contains non-tensor entries: {non_tensors[:10]}")

    # Safetensors requires contiguous tensors. This preserves the source dtype.
    model_state_dict = {name: value.contiguous() for name, value in model_state_dict.items()}
    print(f"Writing {len(model_state_dict):,} tensors to: {output}", flush=True)
    save_file(
        model_state_dict,
        output,
        metadata={
            "format": "pt",
            "source": str(input_dir),
            "conversion": "torch.distributed.checkpoint.format_utils.dcp_to_torch_save",
        },
    )

    with safe_open(output, framework="pt", device="cpu") as checkpoint:
        keys = list(checkpoint.keys())
        if len(keys) != len(model_state_dict):
            raise RuntimeError(
                f"Verification failed: wrote {len(keys):,} tensors, expected {len(model_state_dict):,}"
            )
        first_key = keys[0]
        first_tensor = checkpoint.get_tensor(first_key)
        print(
            f"Verified {len(keys):,} tensors; first={first_key} "
            f"shape={tuple(first_tensor.shape)} dtype={first_tensor.dtype}",
            flush=True,
        )

    print(f"Final size: {output.stat().st_size:,} bytes", flush=True)
    if args.keep_temp:
        print(f"Keeping temporary state dict: {temporary_state_dict}", flush=True)
    else:
        temporary_state_dict.unlink()
        print("Temporary state dict removed.", flush=True)
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
