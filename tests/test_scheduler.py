import pytest
import torch
import torch.nn as nn

from smc.scheduler import NoiseScheduler


class ConstantNoiseModel(nn.Module):
    """Prédit toujours le même bruit : rend sample_step déterministe à t=0."""

    def __init__(self, value=0.3):
        super().__init__()
        self.value = value
        self.last_t = None

    def forward(self, x, t):
        self.last_t = t
        return torch.full_like(x, self.value)


@pytest.fixture
def scheduler():
    return NoiseScheduler(timesteps=50, device="cpu")


# --- Constantes du schedule --------------------------------------------------

def test_alpha_hat_vaut_le_produit_cumule_des_alpha(scheduler):
    """alpha_hat est calculé par cumsum de log : doit égaler cumprod(alpha)."""
    attendu = torch.cumprod(scheduler.alpha, dim=0)

    assert torch.allclose(scheduler.alpha_hat, attendu, atol=1e-6)


def test_alpha_hat_decroissant_dans_zero_un(scheduler):
    alpha_hat = scheduler.alpha_hat

    assert torch.all(alpha_hat > 0.0)
    assert torch.all(alpha_hat <= 1.0)
    assert torch.all(alpha_hat[1:] < alpha_hat[:-1])


# --- Forward : q(x_t | x_0) --------------------------------------------------

def test_add_noise_applique_la_formule(scheduler):
    x0 = torch.randn(4, 3, 32, 32)
    bruit = torch.randn_like(x0)
    t = torch.tensor([0, 10, 25, 49])

    x_t, bruit_rendu = scheduler.add_noise(x0, t, noise=bruit)

    s1 = scheduler.sqrt_alpha_hat[t].view(-1, 1, 1, 1)
    s2 = scheduler.sqrt_one_minus_alpha_hat[t].view(-1, 1, 1, 1)
    assert torch.allclose(x_t, s1 * x0 + s2 * bruit, atol=1e-6)
    assert bruit_rendu is bruit


def test_add_noise_preserve_la_variance_unite(scheduler):
    """x0 ~ N(0,1) et bruit ~ N(0,1) => x_t ~ N(0,1) quel que soit t."""
    x0 = torch.randn(512, 3, 8, 8)
    t = torch.full((512,), 40, dtype=torch.long)

    x_t, _ = scheduler.add_noise(x0, t)

    assert abs(x_t.var().item() - 1.0) < 0.05


def test_add_noise_au_dernier_pas_efface_le_signal():
    sched = NoiseScheduler(timesteps=1000, device="cpu")
    x0 = torch.randn(8, 3, 32, 32)
    bruit = torch.randn_like(x0)
    t = torch.full((8,), 999, dtype=torch.long)

    x_t, _ = sched.add_noise(x0, t, noise=bruit)

    # Il ne doit quasiment plus rester que du bruit.
    assert sched.sqrt_alpha_hat[999].item() < 0.01
    assert torch.allclose(x_t, bruit, atol=0.05)


# --- Reverse : p(x_{t-1} | x_t) ---------------------------------------------

def test_sample_step_deterministe_a_t_zero(scheduler):
    """À t=0 aucun bruit n'est ajouté : deux appels doivent coïncider."""
    model = ConstantNoiseModel()
    x = torch.randn(2, 3, 32, 32)

    x_a, eps_a = scheduler.sample_step(model, x, 0)
    x_b, _ = scheduler.sample_step(model, x, 0)

    assert torch.equal(x_a, x_b)
    attendu = scheduler.sqrt_inv_alpha[0] * (x - scheduler.eps_coef[0] * eps_a)
    assert torch.allclose(x_a, attendu, atol=1e-6)


def test_sample_step_reproductible_avec_generator(scheduler):
    model = ConstantNoiseModel()
    x = torch.randn(2, 3, 32, 32)

    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    x_a, _ = scheduler.sample_step(model, x, 10, generator=g1)
    x_b, _ = scheduler.sample_step(model, x, 10, generator=g2)

    assert torch.equal(x_a, x_b)


def test_sample_step_inverse_add_noise_avec_un_modele_oracle(scheduler):
    """Un modèle qui connaît le vrai bruit ramène la moyenne vers x_0."""
    t = 30
    x0 = torch.randn(4, 3, 8, 8)
    bruit = torch.randn_like(x0)
    x_t, _ = scheduler.add_noise(x0, torch.full((4,), t, dtype=torch.long), noise=bruit)

    class Oracle(nn.Module):
        def forward(self, x, t_batch):
            return bruit

    # Moyenne du pas inverse, sans le terme de bruit ajouté.
    mean = scheduler.sqrt_inv_alpha[t] * (x_t - scheduler.eps_coef[t] * bruit)
    x_prev, _ = scheduler.sample_step(Oracle(), x_t, t)

    ecart_moyenne = (x_prev - mean).std().item()
    assert abs(ecart_moyenne - scheduler.sigma[t].item()) < 0.05
    # La moyenne doit être plus proche de x_0 que ne l'était x_t.
    assert (mean - x0).abs().mean() < (x_t - x0).abs().mean()
