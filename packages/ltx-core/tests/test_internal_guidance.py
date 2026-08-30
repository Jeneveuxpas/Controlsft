import math

import torch

from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderFactory, MultiModalGuiderParams


def _shared_rescale(cond: torch.Tensor, pred: torch.Tensor, scale: float) -> torch.Tensor:
    factor = scale * (cond.std() / pred.std()) + (1 - scale)
    return pred * factor


def test_internal_guidance_is_combined_before_shared_rescale() -> None:
    cond = torch.tensor([[[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]]])
    uncond = torch.tensor([[[0.5, 1.0], [1.0, 2.0], [2.0, 3.0]]])
    perturbed = torch.tensor([[[0.0, 1.0], [2.0, 2.5], [5.0, 8.0]]])
    internal = torch.tensor([[[1.0, 2.0], [2.0, 4.0], [5.0, 9.0]]])
    params = MultiModalGuiderParams(
        cfg_scale=2.0,
        stg_scale=0.5,
        rescale_scale=0.7,
        internal_scale=1.4,
    )

    actual = MultiModalGuider(params).calculate(cond, uncond, perturbed, 0.0, internal=internal)

    base = cond + (params.cfg_scale - 1) * (cond - uncond) + params.stg_scale * (cond - perturbed)
    combined = base.clone()
    combined += (params.internal_scale - 1) * (cond - internal)
    expected = _shared_rescale(cond, combined, params.rescale_scale)

    torch.testing.assert_close(actual, expected)


def test_internal_leaves_reference_prefix_without_a_delta() -> None:
    cond = torch.tensor([[[1.0], [2.0], [4.0]]])
    internal = torch.tensor([[[1.0], [1.0], [3.0]]])
    guider = MultiModalGuider(MultiModalGuiderParams(internal_scale=1.5))

    actual = guider.calculate(cond, 0.0, 0.0, 0.0, internal=internal)

    expected = torch.tensor([[[1.0], [2.5], [4.5]]])
    torch.testing.assert_close(actual, expected)


def test_disabled_internal_guidance_preserves_existing_result() -> None:
    cond = torch.tensor([[[1.0, 2.0], [3.0, 5.0]]])
    uncond = torch.tensor([[[0.0, 1.0], [1.0, 2.0]]])
    internal = torch.full_like(cond, -100.0)
    params = MultiModalGuiderParams(cfg_scale=2.5, rescale_scale=0.7)
    guider = MultiModalGuider(params)

    without_internal = guider.calculate(cond, uncond, 0.0, 0.0)
    with_disabled_internal = guider.calculate(cond, uncond, 0.0, 0.0, internal=internal)

    torch.testing.assert_close(with_disabled_internal, without_internal)


def test_internal_scale_can_follow_the_official_sigma_schedule() -> None:
    internal_off = MultiModalGuiderParams(internal_scale=1.0)
    internal_on = MultiModalGuiderParams(internal_scale=1.4)
    sigma_low = 0.3
    factory = MultiModalGuiderFactory.from_dict(
        {
            math.inf: internal_off,
            0.7: internal_on,
            math.nextafter(sigma_low, -math.inf): internal_off,
        }
    )

    assert factory.params(0.8).internal_scale == 1.0
    assert factory.params(0.7).internal_scale == 1.4
    assert factory.params(0.3).internal_scale == 1.4
    assert factory.params(0.2).internal_scale == 1.0
