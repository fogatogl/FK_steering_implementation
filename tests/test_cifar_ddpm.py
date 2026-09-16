import pytest
import torch
import torch.nn as nn

from smc.models import CifarDDPM


class DummyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1))

    def forward(self, x, t):
        return 0.1 * x


class DummyScheduler:
    def __init__(self, timesteps=5, device="cpu"):
        self.timesteps = timesteps
        self.device = device
        self.sqrt_alpha_hat = torch.linspace(1.0, 0.1, timesteps, device=device)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.sqrt_alpha_hat**2)

    def sample_step(self, model, x, t_idx, generator=None):
        eps = model(x, t_idx)
        return x - 0.05 * eps, eps


@pytest.fixture
def dummy_setup():
    device = "cpu"
    unet = DummyUNet()
    scheduler = DummyScheduler(timesteps=5, device=device)
    return unet, scheduler, CifarDDPM(unet=unet, scheduler=scheduler, device=device)


def test_predict_x0_utilise_x_t_et_non_x_t_moins_un(dummy_setup):
    """Régression : x̂0 doit porter sur l'entrée du step, pas sur sa sortie."""
    _, scheduler, ddpm = dummy_setup
    t = 3
    gen = torch.Generator(device=ddpm.device).manual_seed(0)
    state = ddpm.initial_state(k=2, generator=gen)
    x_t = state["x"].clone()

    state = ddpm.step(state, t, generator=gen)
    x0_hat = ddpm.predict_x0(state, t, clamp=False)

    attendu = ((x_t - scheduler.sqrt_one_minus_alpha_hat[t] * state["eps"])
               / scheduler.sqrt_alpha_hat[t])
    assert torch.allclose(x0_hat, attendu, atol=1e-6)
    depuis_x_prev = ((state["x"] - scheduler.sqrt_one_minus_alpha_hat[t] * state["eps"])
                     / scheduler.sqrt_alpha_hat[t])
    assert not torch.allclose(x0_hat, depuis_x_prev, atol=1e-4)


def test_predict_x0_refuse_un_t_incoherent(dummy_setup):
    _, _, ddpm = dummy_setup
    gen = torch.Generator(device=ddpm.device).manual_seed(0)
    state = ddpm.step(ddpm.initial_state(k=2, generator=gen), 3, generator=gen)

    with pytest.raises(ValueError):
        ddpm.predict_x0(state, 2)


def test_predict_x0_borne_dans_le_domaine_des_images_par_defaut(dummy_setup):
    _, scheduler, ddpm = dummy_setup
    t = scheduler.timesteps - 1
    x_t = torch.full((2, 3, 4, 4), 5.0)
    state = {"x": x_t, "x_t": x_t, "eps": torch.zeros_like(x_t), "t": t}

    brut = ddpm.predict_x0(state, clamp=False)

    assert brut.abs().max() > 1.0
    assert torch.equal(ddpm.predict_x0(state), brut.clamp(-1.0, 1.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pas de GPU")
def test_initial_state_refuse_un_generateur_cpu_sur_un_modele_cuda():
    ddpm = CifarDDPM(unet=DummyUNet().cuda(),
                     scheduler=DummyScheduler(device="cuda"),
                     device="cuda")

    with pytest.raises(ValueError, match="générateur sur 'cpu'"):
        ddpm.initial_state(k=2, generator=torch.Generator())
