"""Run the mixed fixed 56-sample ControlSFT evaluation across multiple GPUs."""

# ruff: noqa: PLR0912, PLR0915, T201
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Worker:
    gpu: str
    start_index: int
    num_samples: int


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
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=workspace
        / "outputs/ablation_stage1_baseline/checkpoints/model_weights_step_00800.safetensors",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=workspace / "checkpoints/ltx-2.3/ltx-2.3-22b-dev.safetensors",
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        help="Optional tuned VAE state dict used for full video decoding.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo / "datasets/controlsft_teacher_test1k/validation_fixed56.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace / "outputs/validation_fixed56/baseline_step_00800",
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
        help="Generate without injecting reference/Part16 control latents.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on manifest line {line_number}: {exc}") from exc
    if len(rows) != 56:
        raise ValueError(f"Fixed evaluation requires exactly 56 samples, got {len(rows)}")
    if [row.get("validation_index") for row in rows] != list(range(56)):
        raise ValueError("Manifest validation_index values must be exactly 0..55")
    if len({row.get("id") for row in rows}) != 56:
        raise ValueError("Manifest must contain 56 unique sample ids")
    expected_groups = ["train_focus32"] * 32 + ["test16"] * 16 + ["new_diving8"] * 8
    if [row.get("evaluation_group") for row in rows] != expected_groups:
        raise ValueError("Manifest groups must be ordered as train_focus32/test16/new_diving8")
    return rows


def _partition(total: int, gpus: list[str]) -> list[Worker]:
    quotient, remainder = divmod(total, len(gpus))
    workers = []
    start = 0
    for worker_index, gpu in enumerate(gpus):
        count = quotient + (1 if worker_index < remainder else 0)
        if count:
            workers.append(Worker(gpu=gpu, start_index=start, num_samples=count))
            start += count
    return workers


def _worker_command(args: argparse.Namespace, worker: Worker) -> list[str]:
    inference_script = Path(__file__).with_name("infer_part16_precomputed.py")
    command = [
        sys.executable,
        str(inference_script),
        "--base-checkpoint",
        str(args.base_checkpoint.resolve()),
        "--trained-checkpoint",
        str(args.checkpoint.resolve()),
        "--manifest-path",
        str(args.manifest.resolve()),
        "--output-dir",
        str(args.output_dir.resolve()),
        "--start-index",
        str(worker.start_index),
        "--num-samples",
        str(worker.num_samples),
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
        "--disable-progress-bars",
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.vae_checkpoint is not None:
        command.extend(["--vae-checkpoint", str(args.vae_checkpoint.resolve())])
    if args.no_reference_control:
        command.append("--no-reference-control")
    return command


def _expected_outputs(rows: list[dict], output_dir: Path) -> list[Path]:
    return [output_dir / f"line-{index}.mp4" for index, _row in enumerate(rows)]


def main() -> None:
    args = _build_parser().parse_args()
    if args.inference_steps < 1 or args.guidance_scale < 1.0 or args.stg_scale < 0.0:
        raise ValueError("Invalid inference/guidance/STG values")
    paths = [args.checkpoint, args.base_checkpoint, args.manifest]
    if args.vae_checkpoint is not None:
        paths.append(args.vae_checkpoint)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    rows = _load_manifest(args.manifest)
    workers = _partition(len(rows), args.gpus)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    print(f"Checkpoint: {args.checkpoint.resolve()}")
    print(f"Manifest:   {args.manifest.resolve()}")
    print(f"Output:     {args.output_dir.resolve()}")
    print(f"Video VAE:  {args.vae_checkpoint.resolve() if args.vae_checkpoint else 'base checkpoint'}")
    print("Decode:     full VAE decode (tiled decode disabled)")
    print(f"Control:    {'disabled' if args.no_reference_control else 'reference/Part16 enabled'}")
    for worker in workers:
        end = worker.start_index + worker.num_samples - 1
        print(f"GPU {worker.gpu}: samples {worker.start_index}..{end}")
        if args.dry_run:
            print("  " + " ".join(_worker_command(args, worker)))
    if args.dry_run:
        return

    processes: list[tuple[Worker, subprocess.Popen, object, Path]] = []
    started_at = time.monotonic()
    try:
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
            details = ", ".join(
                f"GPU {worker.gpu} exit={code} log={log_path}" for worker, code, log_path in failures
            )
            raise RuntimeError(f"Evaluation worker failure: {details}")
    except KeyboardInterrupt:
        print("Interrupted; terminating evaluation workers...", file=sys.stderr)
        for _worker, process, _handle, _path in processes:
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for _worker, process, log_handle, _path in processes:
            if process.poll() is None:
                process.terminate()
            if not log_handle.closed:
                log_handle.close()

    missing = [path for path in _expected_outputs(rows, args.output_dir) if not path.is_file()]
    if missing:
        raise RuntimeError(f"Evaluation finished but {len(missing)} outputs are missing: {missing[:3]}")
    elapsed = time.monotonic() - started_at
    print(f"Evaluation complete: 56/56 outputs in {args.output_dir.resolve()} ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
