import json
from pathlib import Path

import pytest
import torch

from ltx_trainer.config import DataConfig
from ltx_trainer.datasets import PrecomputedDataset


def _save_latent(path: Path, value: float) -> None:
    torch.save(
        {
            "latents": torch.full((2, 1, 1, 1), value),
            "num_frames": 1,
            "height": 1,
            "width": 1,
            "fps": 24.0,
        },
        path,
    )


def test_manifest_loads_explicit_absolute_paths(tmp_path: Path) -> None:
    target = tmp_path / "target.pt"
    reference = tmp_path / "reference.pt"
    condition = tmp_path / "condition.pt"
    _save_latent(target, 1.0)
    _save_latent(reference, 2.0)
    torch.save({"video_prompt_embeds": torch.ones(1, 2)}, condition)

    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "latents": str(target),
                "conditions": str(condition),
                "reference_latents": str(reference),
            }
        )
        + "\n"
    )

    dataset = PrecomputedDataset(
        str(tmp_path),
        data_sources={
            "conditions": "conditions",
            "latents": "video_latents",
            "reference_latents": "reference_latents",
        },
        manifest_path=str(manifest),
    )

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["idx"] == 0
    assert torch.equal(sample["video_latents"]["latents"], torch.ones(2, 1, 1, 1))
    assert torch.equal(sample["reference_latents"]["latents"], torch.full((2, 1, 1, 1), 2.0))
    assert torch.equal(sample["conditions"]["video_prompt_embeds"], torch.ones(1, 2))


def test_manifest_rejects_missing_required_field(tmp_path: Path) -> None:
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps({"latents": "/tmp/target.pt"}) + "\n")

    with pytest.raises(ValueError, match="missing required field 'conditions'"):
        PrecomputedDataset(
            str(tmp_path),
            data_sources={"latents": "video_latents", "conditions": "conditions"},
            manifest_path=str(manifest),
        )


def test_data_config_accepts_manifest_without_precomputed_subdirectories(tmp_path: Path) -> None:
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("{}\n")

    config = DataConfig(preprocessed_data_root=str(tmp_path), manifest_path=str(manifest))

    assert config.manifest_path == str(manifest.resolve())
