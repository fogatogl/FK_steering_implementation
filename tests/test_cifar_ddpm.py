import pytest
import torch
import torch.nn as nn
from smc.models import CifarDDPM
from smc.sampling import sample


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
        x_prev = x - 0.05 * eps
        return x_prev, eps


@pytest.fixture
def dummy_setup():
    device = "cpu"
    unet = DummyUNet()
    scheduler = DummyScheduler(timesteps=5, device=device)
    ddpm = CifarDDPM(unet=unet, scheduler=scheduler, device=device)
    return unet, scheduler, ddpm


def test_step_loop_matches_sample_structure(dummy_setup):
    unet, scheduler, ddpm = dummy_setup
    seed = 1234
    n_samples = 4

    torch.manual_seed(seed)
    x_direct = sample(unet, scheduler, n_samples=n_samples, channels=3, size=32)

    gen = torch.Generator(device=scheduler.device).manual_seed(seed)
    state = ddpm.initial_state(k=n_samples, generator=gen)
    for t_idx in reversed(range(scheduler.timesteps)):
        state = ddpm.step(state, t_idx, generator=gen)

    assert torch.allclose(x_direct, state["x"], atol=1e-6)


def test_initial_state_shape_and_device(dummy_setup):
    _, _, ddpm = dummy_setup
    k = 4
    gen = torch.Generator(device=ddpm.device).manual_seed(42)

    state = ddpm.initial_state(k=k, generator=gen)

    assert state["x"].shape == (4, 3, 32, 32)
    assert state["x"].device == torch.device(ddpm.device)
    assert state["eps"] is None


def test_step_shape_preserved_and_eps_populated(dummy_setup):
    _, _, ddpm = dummy_setup
    gen = torch.Generator(device=ddpm.device).manual_seed(42)
    state = ddpm.initial_state(k=2, generator=gen)

    next_state = ddpm.step(state, t=4, generator=gen)

    assert next_state["x"].shape == state["x"].shape
    assert next_state["eps"] is not None
    assert next_state["eps"].shape == state["x"].shape


def test_predict_x0_domain(dummy_setup):
    _, scheduler, ddpm = dummy_setup
    t = 0
    k = 2

    x0_true = torch.clamp(torch.randn(k, 3, 32, 32), -1.0, 1.0)
    eps_true = torch.randn_like(x0_true)

    sqrt_alpha = scheduler.sqrt_alpha_hat[t]
    sqrt_one_minus_alpha = scheduler.sqrt_one_minus_alpha_hat[t]
    x_t = sqrt_alpha * x0_true + sqrt_one_minus_alpha * eps_true

    state = {"x": x_t, "eps": eps_true}
    x0_hat = ddpm.predict_x0(state, t=t)

    assert torch.allclose(x0_hat, x0_true, atol=1e-5)
    assert x0_hat.min() >= -1.5
    assert x0_hat.max() <= 1.5


def test_network_call_count(dummy_setup):
    unet, scheduler, ddpm = dummy_setup
    t_steps = scheduler.timesteps
    gen = torch.Generator(device=ddpm.device).manual_seed(42)

    call_count = 0
    original_forward = unet.forward

    def counting_forward(x, t):
        nonlocal call_count
        call_count += 1
        return original_forward(x, t)

    unet.forward = counting_forward

    state = ddpm.initial_state(k=2, generator=gen)
    for t_idx in reversed(range(t_steps)):
        state = ddpm.step(state, t_idx, generator=gen)
        _ = ddpm.predict_x0(state, t_idx)

    assert call_count == t_steps


def test_predict_x0_utilise_x_t_et_non_x_t_moins_un(dummy_setup):
    """Régression : x̂0 doit porter sur l'entrée du step, pas sur sa sortie.

    `eps` est prédit depuis x_t ; appliquer Tweedie à x_{t-1} avec les
    coefficients de t mélangeait deux dates.
    """
    _, scheduler, ddpm = dummy_setup
    t = 3
    gen = torch.Generator(device=ddpm.device).manual_seed(0)
    state = ddpm.initial_state(k=2, generator=gen)
    x_t = state["x"].clone()

    state = ddpm.step(state, t, generator=gen)
    x0_hat = ddpm.predict_x0(state, t)

    attendu = ((x_t - scheduler.sqrt_one_minus_alpha_hat[t] * state["eps"])
               / scheduler.sqrt_alpha_hat[t])
    assert torch.allclose(x0_hat, attendu, atol=1e-6)
    # Et surtout : ce n'est pas ce qu'on obtiendrait depuis x_{t-1}.
    depuis_x_prev = ((state["x"] - scheduler.sqrt_one_minus_alpha_hat[t] * state["eps"])
                     / scheduler.sqrt_alpha_hat[t])
    assert not torch.allclose(x0_hat, depuis_x_prev, atol=1e-4)


def test_predict_x0_deduit_t_de_l_etat(dummy_setup):
    _, _, ddpm = dummy_setup
    gen = torch.Generator(device=ddpm.device).manual_seed(0)
    state = ddpm.step(ddpm.initial_state(k=2, generator=gen), 3, generator=gen)

    assert torch.allclose(ddpm.predict_x0(state), ddpm.predict_x0(state, 3))


def test_predict_x0_refuse_un_t_incoherent(dummy_setup):
    _, _, ddpm = dummy_setup
    gen = torch.Generator(device=ddpm.device).manual_seed(0)
    state = ddpm.step(ddpm.initial_state(k=2, generator=gen), 3, generator=gen)

    with pytest.raises(ValueError):
        ddpm.predict_x0(state, 2)


def test_predict_x0_refuse_un_etat_sans_eps(dummy_setup):
    _, _, ddpm = dummy_setup
    gen = torch.Generator(device=ddpm.device).manual_seed(0)
    state = ddpm.initial_state(k=2, generator=gen)

    with pytest.raises(ValueError):
        ddpm.predict_x0(state, 4)


# Le test de non-régression sur les poids réels vit dans tests/test_checkpoint.py
# (test_equivalence_sample_et_cifar_ddpm_sur_modele_reel) : il y trouve le
# checkpoint via la fixture partagée plutôt qu'un chemin relatif inexistant.
