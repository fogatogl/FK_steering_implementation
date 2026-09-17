import torch
import torch.nn as nn

from smc.models import CifarDDPM


class DummyUNet(nn.Module):
    def forward(self, x, t):
        return 0.1 * x


class DummyScheduler:
    def __init__(self, timesteps=5):
        self.timesteps = timesteps
        self.device = "cpu"
        self.sqrt_alpha_hat = torch.linspace(1.0, 0.1, timesteps)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.sqrt_alpha_hat**2)

    def sample_step(self, model, x, t_idx, generator=None):
        eps = model(x, t_idx)
        return x - 0.05 * eps, eps


def test_predict_x0_uses_x_t_not_x_t_minus_one():
    """Regression: x̂0 is computed from the step's input, not its output."""
    scheduler = DummyScheduler()
    ddpm = CifarDDPM(unet=DummyUNet(), scheduler=scheduler, device="cpu")
    t = 3
    gen = torch.Generator().manual_seed(0)
    state = ddpm.initial_state(k=2, generator=gen)
    x_t = state["x"].clone()

    state = ddpm.step(state, t, generator=gen)
    x0_hat = ddpm.predict_x0(state, t, clamp=False)

    expected = (x_t - scheduler.sqrt_one_minus_alpha_hat[t] * state["eps"]) / scheduler.sqrt_alpha_hat[t]
    from_x_prev = (state["x"] - scheduler.sqrt_one_minus_alpha_hat[t] * state["eps"]) / scheduler.sqrt_alpha_hat[t]
    assert torch.allclose(x0_hat, expected, atol=1e-6)
    assert not torch.allclose(x0_hat, from_x_prev, atol=1e-4)
