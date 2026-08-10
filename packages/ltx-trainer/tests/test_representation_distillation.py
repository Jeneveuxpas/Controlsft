import pytest
import torch
from pydantic import ValidationError

from ltx_trainer.distillation import RepresentationProjectionHead, compute_representation_distillation_loss
from ltx_trainer.timestep_samplers import ShiftedLogitNormalTimestepSampler, TimestepSampler
from ltx_trainer.training_strategies.flexible import (
    FlexibleStrategy,
    FlexibleStrategyConfig,
    ModalityConfig,
    ReferenceConditionConfig,
    RepresentationDistillationConfig,
)


class SequenceTimestepSampler(TimestepSampler):
    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.index = 0

    def sample(  # noqa: ARG002
        self, batch_size: int, seq_length: int | None = None, device: torch.device = None
    ) -> torch.Tensor:
        value = self.values[self.index]
        self.index += 1
        return torch.full((batch_size,), value, device=device)

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        return self.sample(batch.shape[0], batch.shape[1], batch.device)


def _latent_batch(batch_size: int = 2, height: int = 2, width: int = 2) -> dict:
    return {
        "latents": torch.randn(batch_size, 128, 1, height, width),
        "num_frames": torch.ones(batch_size, dtype=torch.int64),
        "height": torch.full((batch_size,), height, dtype=torch.int64),
        "width": torch.full((batch_size,), width, dtype=torch.int64),
        "fps": torch.full((batch_size,), 24.0),
    }


def _batch(batch_size: int = 2, height: int = 2, width: int = 2) -> dict:
    return {
        "video_latents": _latent_batch(batch_size, height, width),
        "reference_latents": _latent_batch(batch_size, 1, 2),
        "conditions": {
            "video_prompt_embeds": torch.randn(batch_size, 4, 8),
            "audio_prompt_embeds": None,
            "prompt_attention_mask": torch.ones(batch_size, 4, dtype=torch.int64),
        },
    }


def _strategy(noise_mode: str) -> FlexibleStrategy:
    return FlexibleStrategy(
        FlexibleStrategyConfig(
            video=ModalityConfig(is_generated=True, latents_dir="latents", conditions=[]),
            representation_distillation=RepresentationDistillationConfig(
                teacher_conditions=[ReferenceConditionConfig(latents_dir="reference_latents")],
                noise_mode=noise_mode,
                sra_timestep_max_gap=0.2,
                dual_timestep_second_probability=0.1,
            ),
        )
    )


def test_same_noise_distillation_pairs_target_tokens_and_loads_teacher_reference() -> None:
    strategy = _strategy("same")
    paired = strategy.prepare_distillation_inputs(_batch(), SequenceTimestepSampler([0.6]))

    student = paired.student
    teacher = paired.teacher
    assert strategy.config.get_data_sources() == {
        "conditions": "conditions",
        "latents": "video_latents",
        "reference_latents": "reference_latents",
    }
    assert student.video_target_start_index == 0
    assert teacher.video_target_start_index == 2
    assert torch.equal(student.video.latent, teacher.video.latent[:, 2:])
    assert torch.equal(student.video_targets, teacher.video_targets)
    assert torch.equal(student.video.timesteps, teacher.video.timesteps[:, 2:])
    assert paired.low_sigmas is None
    assert paired.second_timestep_mask is None


def test_dual_timestep_uses_shared_noise_and_preserves_unordered_student_t_s() -> None:
    torch.manual_seed(7)
    strategy = _strategy("dual_timestep")
    paired = strategy.prepare_distillation_inputs(
        _batch(batch_size=8, height=8, width=8),
        SequenceTimestepSampler([0.2, 0.8]),
    )
    student = paired.student
    teacher = paired.teacher
    student_x0 = student.video_clean_latents
    student_tau = student.video.timesteps.unsqueeze(-1)
    teacher_tau = teacher.video.timesteps[:, teacher.video_target_start_index :].unsqueeze(-1)
    teacher_xt = teacher.video.latent[:, teacher.video_target_start_index :]
    student_noise = (student.video.latent - (1 - student_tau) * student_x0) / student_tau
    teacher_noise = (teacher_xt - (1 - teacher_tau) * student_x0) / teacher_tau

    assert torch.allclose(student_noise, teacher_noise, atol=2e-5, rtol=2e-5)
    second_mask = paired.second_timestep_mask
    target_timesteps = student.video.timesteps
    assert torch.allclose(target_timesteps[second_mask], torch.full_like(target_timesteps[second_mask], 0.8))
    assert torch.allclose(target_timesteps[~second_mask], torch.full_like(target_timesteps[~second_mask], 0.2))
    assert torch.allclose(teacher_tau, torch.full_like(teacher_tau, 0.2))
    assert torch.allclose(student.video.sigma, torch.full_like(student.video.sigma, 0.2))
    assert torch.allclose(paired.low_sigmas, torch.full_like(paired.low_sigmas, 0.2))
    assert second_mask.float().mean().item() == pytest.approx(0.1, abs=0.05)


@pytest.mark.parametrize(("first", "second"), [(0.7, 0.3), (0.3, 0.7)])
def test_independent_high_low_sorts_student_and_teacher_timesteps(first: float, second: float) -> None:
    paired = _strategy("independent_high_low").prepare_distillation_inputs(
        _batch(), SequenceTimestepSampler([first, second])
    )
    teacher_start = paired.teacher.video_target_start_index

    assert torch.allclose(paired.student.video.timesteps, torch.full_like(paired.student.video.timesteps, 0.7))
    assert torch.allclose(
        paired.teacher.video.timesteps[:, teacher_start:],
        torch.full_like(paired.student.video.timesteps, 0.3),
    )
    assert torch.allclose(paired.student.video.sigma, torch.full_like(paired.student.video.sigma, 0.7))
    assert torch.allclose(paired.teacher.video.sigma, torch.full_like(paired.teacher.video.sigma, 0.3))
    assert paired.second_timestep_mask is None


def test_sra_teacher_timestep_is_below_student_by_at_most_configured_gap() -> None:
    torch.manual_seed(11)
    paired = _strategy("sra").prepare_distillation_inputs(
        _batch(batch_size=8), SequenceTimestepSampler([0.7])
    )
    teacher_start = paired.teacher.video_target_start_index
    student_timesteps = paired.student.video.timesteps
    teacher_timesteps = paired.teacher.video.timesteps[:, teacher_start:]

    assert torch.allclose(student_timesteps, torch.full_like(student_timesteps, 0.7))
    assert torch.all(teacher_timesteps <= student_timesteps)
    assert torch.all(teacher_timesteps >= student_timesteps - 0.2)
    assert torch.allclose(teacher_timesteps, paired.low_sigmas[:, None].expand_as(teacher_timesteps))
    assert paired.second_timestep_mask is None


def test_distillation_config_rejects_conditioned_student_and_probabilistic_teacher() -> None:
    reference = ReferenceConditionConfig(latents_dir="reference_latents")
    with pytest.raises(ValidationError, match="student must not have video conditions"):
        FlexibleStrategyConfig(
            video=ModalityConfig(is_generated=True, latents_dir="latents", conditions=[reference]),
            representation_distillation=RepresentationDistillationConfig(),
        )

    with pytest.raises(ValidationError, match="probability=1.0"):
        FlexibleStrategyConfig(
            video=ModalityConfig(is_generated=True, latents_dir="latents"),
            representation_distillation=RepresentationDistillationConfig(
                teacher_conditions=[ReferenceConditionConfig(latents_dir="reference_latents", probability=0.5)]
            ),
        )


def test_projection_and_cosine_loss_only_backpropagate_to_student() -> None:
    head = RepresentationProjectionHead(input_dim=8, hidden_dim=4)
    student = torch.randn(2, 3, 8, requires_grad=True)
    teacher = torch.randn(2, 3, 8, requires_grad=True)
    projected = head(student)
    loss, metrics = compute_representation_distillation_loss(
        projected,
        teacher,
        torch.tensor([[True, True, False], [True, False, False]]),
        loss_weight=0.8,
        warmup_steps=100,
        global_step=50,
        loss_type="cosine",
    )
    loss.mean().backward()

    assert student.grad is not None
    assert teacher.grad is None
    assert all(parameter.grad is not None for parameter in head.parameters())
    assert torch.allclose(metrics["train/representation_distillation_weight"], torch.tensor(0.4))


@pytest.mark.parametrize(
    ("loss_type", "expected"),
    [
        ("cosine", 1.0),
        ("l1", 1.0),
        ("l2", 1.0),
    ],
)
def test_representation_distillation_loss_types(loss_type: str, expected: float) -> None:
    student = torch.tensor([[[1.0, 0.0], [100.0, 100.0]]], requires_grad=True)
    teacher = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]], requires_grad=True)
    loss, _ = compute_representation_distillation_loss(
        student,
        teacher,
        torch.tensor([[True, False]]),
        loss_weight=1.0,
        warmup_steps=0,
        global_step=0,
        loss_type=loss_type,
    )

    assert torch.allclose(loss, torch.tensor([expected]))
    loss.mean().backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_video_high_noise_sampler_override_is_bounded_and_optional() -> None:
    sampler = ShiftedLogitNormalTimestepSampler(
        uniform_prob=0.0,
        high_noise_probability=1.0,
        high_noise_min=0.95,
    )
    samples = sampler.sample(batch_size=256, seq_length=2048)

    assert torch.all(samples >= 0.95)
    assert torch.all(samples <= 1.0)
    with pytest.raises(ValueError, match="high_noise_probability"):
        ShiftedLogitNormalTimestepSampler(high_noise_probability=1.1)
