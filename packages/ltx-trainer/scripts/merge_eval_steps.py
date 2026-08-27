#!/usr/bin/env python
"""Merge control-conditioned evaluation videos from four checkpoints.

Each input video is expected to be ``[control | generated_target]``. Portrait
samples are written as ``[control | 500 | 1000 | 1500 | 2000]`` in one row;
landscape samples are written as a 2x2 grid of the four generated targets.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image


STEPS = ("00500", "01000", "01500", "02000")


def _read_video(path: Path) -> tuple[list[np.ndarray], float]:
    container = av.open(str(path))
    stream = container.streams.video[0]
    fps = float(stream.average_rate or 24.0)
    frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    container.close()
    if not frames:
        raise RuntimeError(f"No video frames found in {path}")
    return frames, fps


def _resize(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    image = Image.fromarray(frame, mode="RGB")
    return np.asarray(image.resize((width, height), Image.Resampling.BICUBIC))


def _crop_target(frames: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    height, width, _ = frames[0].shape
    if width % 2:
        raise RuntimeError(f"Combined video width must be even, got {width}")
    midpoint = width // 2
    controls = [frame[:, :midpoint] for frame in frames]
    targets = [frame[:, midpoint:] for frame in frames]
    return controls, targets


def _tile(frames_by_video: list[list[np.ndarray]], width: int, height: int, columns: int) -> list[np.ndarray]:
    normalized = [
        [_resize(frame, width, height) for frame in frames]
        for frames in frames_by_video
    ]
    length = max(len(frames) for frames in normalized)
    output = []
    rows = (len(normalized) + columns - 1) // columns
    for index in range(length):
        tiles = [frames[min(index, len(frames) - 1)] for frames in normalized]
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        while len(tiles) < rows * columns:
            tiles.append(blank)
        row_images = [
            np.concatenate(tiles[row * columns : (row + 1) * columns], axis=1)
            for row in range(rows)
        ]
        output.append(np.concatenate(row_images, axis=0))
    return output


def _write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1001))
    stream.width = frames[0].shape[1]
    stream.height = frames[0].shape[0]
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "medium"}
    for array in frames:
        packet = stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24"))
        if packet:
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Directory containing 00500/01000/01500/02000")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=56)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(args.count):
        controls: list[np.ndarray] | None = None
        targets: list[list[np.ndarray]] = []
        fps = 24.0
        for step in STEPS:
            frames, current_fps = _read_video(args.root / step / f"line-{index}.mp4")
            if controls is None:
                controls, target = _crop_target(frames)
            else:
                _, target = _crop_target(frames)
            targets.append(target)
            fps = current_fps

        assert controls is not None
        target_height, target_width = targets[0][0].shape[:2]
        if target_width < target_height:
            merged_inputs = [controls] + targets
            merged = _tile(merged_inputs, target_width, target_height, columns=5)
        else:
            merged = _tile(targets, target_width, target_height, columns=2)
        output_path = args.output_dir / f"line-{index}.mp4"
        _write_video(output_path, merged, fps)
        print(f"Saved {output_path} ({merged[0].shape[1]}x{merged[0].shape[0]}, {len(merged)} frames)")


if __name__ == "__main__":
    main()
