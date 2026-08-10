import torch


class TimestepSampler:
    """Base class for timestep samplers.
    Timestep samplers are used to sample timesteps for diffusion models.
    They should implement both sample() and sample_for() methods.
    """

    def sample(self, batch_size: int, seq_length: int | None = None, device: torch.device = None) -> torch.Tensor:
        """Sample timesteps for a batch.
        Args:
            batch_size: Number of timesteps to sample
            seq_length: (optional) Length of the sequence being processed
            device: Device to place the samples on
        Returns:
            Tensor of shape (batch_size,) containing timesteps
        """
        raise NotImplementedError

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        """Sample timesteps for a specific batch tensor.
        Args:
            batch: Input tensor of shape (batch_size, seq_length, ...)
        Returns:
            Tensor of shape (batch_size,) containing timesteps
        """
        raise NotImplementedError


class UniformTimestepSampler(TimestepSampler):
    """Samples timesteps uniformly between min_value and max_value (default 0 and 1)."""

    def __init__(self, min_value: float = 0.0, max_value: float = 1.0):
        self.min_value = min_value
        self.max_value = max_value

    def sample(  # noqa: ARG002
        self, batch_size: int, seq_length: int | None = None, device: torch.device = None
    ) -> torch.Tensor:
        return torch.rand(batch_size, device=device) * (self.max_value - self.min_value) + self.min_value

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim != 3:
            raise ValueError(f"Batch should have 3 dimensions, got {batch.ndim}")

        return self.sample(batch.shape[0], device=batch.device)


class ShiftedLogitNormalTimestepSampler(TimestepSampler):
    """
    Samples timesteps from a stretched shifted logit-normal distribution,
    where the shift is determined by the sequence length.
    The stretching normalizes samples between percentile bounds to ensure
    the distribution covers [0, 1] more evenly. A uniform fallback prevents
    collapse at high token counts. An optional high-noise override supports
    the low-SNR coverage experiment used for Self-Flow video training.
    """

    def __init__(
        self,
        std: float = 1.0,
        eps: float = 1e-3,
        uniform_prob: float = 0.1,
        high_noise_probability: float = 0.0,
        high_noise_min: float = 0.95,
    ):
        if not 0.0 <= high_noise_probability <= 1.0:
            raise ValueError("high_noise_probability must be in [0, 1]")
        if not 0.0 <= high_noise_min < 1.0:
            raise ValueError("high_noise_min must be in [0, 1)")
        self.std = std
        self.eps = eps
        self.uniform_prob = uniform_prob
        self.high_noise_probability = high_noise_probability
        self.high_noise_min = high_noise_min
        # Percentile values for stretching (scaled by std)
        # 99.9th percentile of standard normal ≈ 3.0902
        # 0.5th percentile of standard normal ≈ -2.5758
        self.normal_999_percentile = 3.0902 * std
        self.normal_005_percentile = -2.5758 * std

    def sample(self, batch_size: int, seq_length: int, device: torch.device = None) -> torch.Tensor:
        """Sample timesteps for a batch from a stretched shifted logit-normal distribution.
        Args:
            batch_size: Number of timesteps to sample
            seq_length: Length of the sequence being processed, used to determine the shift
            device: Device to place the samples on
        Returns:
            Tensor of shape (batch_size,) containing timesteps sampled from a stretched
            shifted logit-normal distribution, where the shift is determined by seq_length
        """
        mu = self._get_shift_for_sequence_length(seq_length)

        # Sample from shifted logit-normal
        normal_samples = torch.randn((batch_size,), device=device) * self.std + mu
        logitnormal_samples = torch.sigmoid(normal_samples)

        # Compute percentile bounds for stretching
        percentile_999 = torch.sigmoid(torch.tensor(mu + self.normal_999_percentile, device=device))
        percentile_005 = torch.sigmoid(torch.tensor(mu + self.normal_005_percentile, device=device))

        # Stretch to [0, 1] range by normalizing between percentiles
        zero_terminal_raw = (logitnormal_samples - percentile_005) / (percentile_999 - percentile_005)

        # Reflect small values around eps for numerical stability
        stretched_logit = torch.where(
            zero_terminal_raw >= self.eps,
            zero_terminal_raw,
            2 * self.eps - zero_terminal_raw,
        )
        stretched_logit = torch.clamp(stretched_logit, 0, 1)

        # Mix with uniform samples (uniform_prob of the time)
        uniform = (1 - self.eps) * torch.rand((batch_size,), device=device) + self.eps
        prob = torch.rand((batch_size,), device=device)

        sampled = torch.where(prob > self.uniform_prob, stretched_logit, uniform)
        if self.high_noise_probability > 0.0:
            high_noise = self.high_noise_min + (1.0 - self.high_noise_min) * torch.rand(
                (batch_size,), device=device
            )
            use_high_noise = torch.rand((batch_size,), device=device) < self.high_noise_probability
            sampled = torch.where(use_high_noise, high_noise, sampled)
        return sampled

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        """Sample timesteps for a specific batch tensor.
        Args:
            batch: Input tensor of shape (batch_size, seq_length, ...)
        Returns:
            Tensor of shape (batch_size,) containing timesteps sampled from a shifted
            logit-normal distribution, where the shift is determined by the sequence length
            of the input batch
        Raises:
            ValueError: If the input batch does not have 3 dimensions
        """
        if batch.ndim != 3:
            raise ValueError(f"Batch should have 3 dimensions, got {batch.ndim}")

        batch_size, seq_length, _ = batch.shape
        return self.sample(batch_size, seq_length, device=batch.device)

    @staticmethod
    def _get_shift_for_sequence_length(
        seq_length: int,
        min_tokens: int = 1024,
        max_tokens: int = 4096,
        min_shift: float = 0.95,
        max_shift: float = 2.05,
    ) -> float:
        # Calculate the shift value for a given sequence length using linear interpolation
        # between min_shift and max_shift based on sequence length.
        m = (max_shift - min_shift) / (max_tokens - min_tokens)  # Calculate slope
        b = min_shift - m * min_tokens  # Calculate y-intercept
        shift = m * seq_length + b  # Apply linear equation y = mx + b
        return shift


SAMPLERS = {
    "uniform": UniformTimestepSampler,
    "shifted_logit_normal": ShiftedLogitNormalTimestepSampler,
}


def example() -> None:
    # noinspection PyUnresolvedReferences
    import matplotlib.pyplot as plt  # noqa: PLC0415

    sampler = ShiftedLogitNormalTimestepSampler()
    for seq_length in [1024, 2048, 4096, 8192]:
        samples = sampler.sample(batch_size=1_000_000, seq_length=seq_length)

        # plot the histogram of the samples
        plt.hist(samples.numpy(), bins=100, density=True)
        plt.title(f"Timestep Samples for Sequence Length {seq_length}")
        plt.xlabel("Timestep")
        plt.ylabel("Density")
        plt.show()


if __name__ == "__main__":
    example()
