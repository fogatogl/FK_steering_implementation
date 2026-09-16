"""Non-régression du sampler : le refactor notebook -> smc/ ne change pas les images.

C'est le livrable de l'exercice 1.1 (docs/plan_code_fk_pgdlm.md:70) : « boucler
`step` de T à 0 doit produire les mêmes images que ton ancien sampler à seed
égale ». Les tests unitaires sur mocks valident les pièces ; seuls ceux-ci
valident l'assemblage, sur les vrais poids et le schedule complet.

Marqués `slow` : ils font tourner T=1000 pas sur le modèle entraîné.
"""
import pytest
import torch

from smc.models import CifarDDPM
from smc.sampling import sample
from smc.scheduler import NoiseScheduler
from smc.unet import UNet

K = 4
SEED = 2024


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def modele_et_schedule(checkpoint):
    """UNet entraîné + schedule complet, sur le device disponible.

    Reconstruit plutôt que de réutiliser la fixture `real_unet` (CPU, portée
    session) : `.to()` la déplacerait pour tous les autres tests.
    """
    device = _device()
    config = checkpoint["config"]
    unet = UNet(in_channels=3, n_feat=config["n_feat"]).to(device)
    unet.load_state_dict(checkpoint["model"])
    unet.eval()
    scheduler = NoiseScheduler(timesteps=config["timesteps"],
                               beta_start=config["beta1"], beta_end=config["beta2"],
                               device=device)
    return unet, scheduler


class SchedulerNotebook(NoiseScheduler):
    """`sample_step` tel qu'il était dans demo_DDPM.ipynb, avant le refactor.

    Seule différence avec la version actuelle : le bruit vient de
    `randn_like(x_t)` et non de `randn(x_t.shape, generator=...)`.
    """

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
    """`sample()` et la boucle `CifarDDPM` produisent les mêmes images.

    Schedule complet (T=1000), poids réels, k=4 : c'est le test de
    non-régression du refactor, pas une vérification de structure.
    """
    unet, scheduler = modele_et_schedule
    ddpm = CifarDDPM(unet=unet, scheduler=scheduler, device=scheduler.device)

    torch.manual_seed(SEED)
    x_ref = sample(unet, scheduler, n_samples=K, channels=3, size=32)

    torch.manual_seed(SEED)
    state = ddpm.initial_state(k=K, generator=None)
    for t_idx in reversed(range(scheduler.timesteps)):
        state = ddpm.step(state, t_idx, generator=None)

    assert state["x"].shape == x_ref.shape
    assert torch.allclose(x_ref, state["x"], atol=1e-5)


@pytest.mark.slow
def test_bruit_du_refactor_identique_au_notebook(modele_et_schedule):
    """`randn(shape, generator=None)` consomme le RNG comme `randn_like`.

    Le refactor a remplacé `randn_like(x_t)` par `randn(x_t.shape,
    generator=...)` dans NoiseScheduler.sample_step. Si cela avait décalé le
    RNG, les images auraient changé sans que rien ne le signale -- et le FID de
    référence ne s'appliquerait plus. Sur T=1000 pas, les trajectoires doivent
    être identiques *bit à bit*, pas seulement proches.
    """
    unet, scheduler = modele_et_schedule
    # Betas relus sur le scheduler testé plutôt que laissés aux valeurs par
    # défaut : un checkpoint entraîné avec d'autres betas comparerait sinon
    # deux schedules différents.
    notebook = SchedulerNotebook(timesteps=scheduler.timesteps,
                                 beta_start=scheduler.beta[0].item(),
                                 beta_end=scheduler.beta[-1].item(),
                                 device=scheduler.device)

    torch.manual_seed(SEED)
    x_actuel = sample(unet, scheduler, n_samples=K, channels=3, size=32)

    torch.manual_seed(SEED)
    x_notebook = sample(unet, notebook, n_samples=K, channels=3, size=32)

    assert torch.equal(x_actuel, x_notebook)


@pytest.mark.slow
def test_images_generees_dans_le_domaine_attendu(modele_et_schedule):
    """Garde-fou : une trajectoire complète doit rendre des images plausibles."""
    unet, scheduler = modele_et_schedule

    torch.manual_seed(SEED)
    x = sample(unet, scheduler, n_samples=K, channels=3, size=32)

    assert torch.isfinite(x).all()
    assert x.abs().max() < 5.0          # pas de divergence
    assert x.std() > 0.2                # pas d'effondrement vers une constante
