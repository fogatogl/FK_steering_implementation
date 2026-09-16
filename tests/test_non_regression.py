"""Le refactor notebook -> smc/ ne change pas les images (exercice 1.1).

Marqués `slow` : T=1000 pas sur le modèle entraîné.
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
def modele_et_schedule(checkpoint):
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
    """`sample_step` tel qu'il était dans demo_DDPM.ipynb : bruit par randn_like."""

    @torch.no_grad()
    def sample_step(self, model, x_t, t_idx, generator=None):
        t = torch.full((x_t.shape[0],), t_idx, device=self.device, dtype=torch.long)
        pred_noise = model(x_t, t)
        mean = self.sqrt_inv_alpha[t_idx] * (x_t - self.eps_coef[t_idx] * pred_noise)
        if t_idx > 0:
            return mean + self.sigma[t_idx] * torch.randn_like(x_t), pred_noise
        return mean, pred_noise


@pytest.mark.slow
def test_sampler_original_vs_interface_diffusion_model(modele_et_schedule):
    unet, scheduler = modele_et_schedule
    ddpm = CifarDDPM(unet=unet, scheduler=scheduler, device=scheduler.device)

    torch.manual_seed(SEED)
    x_ref = sample(unet, scheduler, n_samples=K, channels=3, size=32)

    torch.manual_seed(SEED)
    state = ddpm.initial_state(k=K, generator=None)
    for t_idx in reversed(range(scheduler.timesteps)):
        state = ddpm.step(state, t_idx, generator=None)

    assert torch.allclose(x_ref, state["x"], atol=1e-5)


@pytest.mark.slow
def test_bruit_du_refactor_identique_au_notebook(modele_et_schedule):
    """`randn(shape, generator=None)` consomme le RNG comme `randn_like` : sinon
    les images auraient changé et le FID de référence ne s'appliquerait plus."""
    unet, scheduler = modele_et_schedule
    notebook = SchedulerNotebook(timesteps=scheduler.timesteps,
                                 beta_start=scheduler.beta[0].item(),
                                 beta_end=scheduler.beta[-1].item(),
                                 device=scheduler.device)

    torch.manual_seed(SEED)
    x_actuel = sample(unet, scheduler, n_samples=K, channels=3, size=32)

    torch.manual_seed(SEED)
    x_notebook = sample(unet, notebook, n_samples=K, channels=3, size=32)

    assert torch.equal(x_actuel, x_notebook)
