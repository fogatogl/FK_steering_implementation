"""The notebook -> smc/ refactor does not change the images.

Marked `slow`: T=1000 steps on the trained model.
"""
import pytest
import torch

from smc.models import CifarDDPM
from smc.sampling import sample
from smc.scheduler import NoiseScheduler
from smc.unet import UNet

K = 4
SEED = 2024


@pytest.fixture(scope="module")
def model_and_scheduler(checkpoint):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = checkpoint["config"]
    unet = UNet(in_channels=3, n_feat=config["n_feat"]).to(device)
    unet.load_state_dict(checkpoint["model"])
    unet.eval()
    scheduler = NoiseScheduler(timesteps=config["timesteps"],
                               beta_start=config["beta1"], beta_end=config["beta2"],
                               device=device)
    return unet, scheduler


class SchedulerNotebook(NoiseScheduler):
    """`sample_step` as it was in demo_DDPM.ipynb: noise from randn_like."""

    @torch.no_grad()
    def sample_step(self, model, x_t, t_idx, generator=None):
        t = torch.full((x_t.shape[0],), t_idx, device=self.device, dtype=torch.long)
        pred_noise = model(x_t, t)
        mean = self.sqrt_inv_alpha[t_idx] * (x_t - self.eps_coef[t_idx] * pred_noise)
        if t_idx > 0:
            return mean + self.sigma[t_idx] * torch.randn_like(x_t), pred_noise
        return mean, pred_noise


@pytest.mark.slow
def test_original_sampler_vs_diffusion_model_interface(model_and_scheduler):
    unet, scheduler = model_and_scheduler
    ddpm = CifarDDPM(unet=unet, scheduler=scheduler, device=scheduler.device)

    torch.manual_seed(SEED)
    x_ref = sample(unet, scheduler, n_samples=K, channels=3, size=32)

    torch.manual_seed(SEED)
    state = ddpm.initial_state(k=K, generator=None)
    for t_idx in reversed(range(scheduler.timesteps)):
        state = ddpm.step(state, t_idx, generator=None)

    assert torch.allclose(x_ref, state["x"], atol=1e-5)


@pytest.mark.slow
def test_refactor_noise_identical_to_notebook(model_and_scheduler):
    """`randn(shape, generator=None)` consumes the RNG like `randn_like`: otherwise
    the images would have changed and the reference FID would no longer apply."""
    unet, scheduler = model_and_scheduler
    notebook = SchedulerNotebook(timesteps=scheduler.timesteps,
                                 beta_start=scheduler.beta[0].item(),
                                 beta_end=scheduler.beta[-1].item(),
                                 device=scheduler.device)

    torch.manual_seed(SEED)
    x_current = sample(unet, scheduler, n_samples=K, channels=3, size=32)

    torch.manual_seed(SEED)
    x_notebook = sample(unet, notebook, n_samples=K, channels=3, size=32)

    assert torch.equal(x_current, x_notebook)
