from ltx_trainer.training_strategies.flexible import (
    FirstFrameConditionConfig,
    ReferenceConditionConfig,
    _reference_conditions_in_prepend_order,
)


def test_reference_prepend_order_preserves_config_declaration_order() -> None:
    part16 = ReferenceConditionConfig(latents_dir="reference_latents")
    depth = ReferenceConditionConfig(latents_dir="reference_latents_1")
    conditions = [FirstFrameConditionConfig(), part16, depth]

    application_order = _reference_conditions_in_prepend_order(conditions)
    assert [condition.latents_dir for condition in application_order] == [
        "reference_latents_1",
        "reference_latents",
    ]

    token_order = ["target"]
    for condition in application_order:
        token_order.insert(0, condition.latents_dir)

    assert token_order == [
        "reference_latents",
        "reference_latents_1",
        "target",
    ]


def test_single_reference_application_order_is_unchanged() -> None:
    reference = ReferenceConditionConfig(latents_dir="reference_latents")

    application_order = _reference_conditions_in_prepend_order([reference])

    assert application_order == [reference]
