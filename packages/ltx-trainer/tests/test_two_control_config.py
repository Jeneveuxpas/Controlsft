from pathlib import Path

import yaml

from ltx_trainer.training_strategies.flexible import (
    FlexibleStrategyConfig,
    ReferenceConditionConfig,
)


def test_two_control_example_declares_ordered_reference_sources() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "v2v_two_control_ic_lora.yaml"
    )
    raw_config = yaml.safe_load(config_path.read_text())
    strategy = FlexibleStrategyConfig.model_validate(raw_config["training_strategy"])

    assert strategy.video is not None
    references = [
        condition
        for condition in strategy.video.conditions
        if isinstance(condition, ReferenceConditionConfig)
    ]
    assert [reference.latents_dir for reference in references] == [
        "reference_latents",
        "reference_latents_1",
    ]
    assert strategy.get_data_sources() == {
        "conditions": "conditions",
        "latents": "video_latents",
        "reference_latents": "reference_latents",
        "reference_latents_1": "reference_latents_1",
    }
