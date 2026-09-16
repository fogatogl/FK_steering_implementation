import pytest
import torch
import torch.nn as nn

from smc.sampling import sample, sample_trajectory
from smc.scheduler import NoiseScheduler


class ModeleTracable(nn.Module):
    """Renvoie un bruit constant et compte ses appels."""

    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1))
        self.appels = 0

    def forward(self, x, t):
        self.appels += 1
        return torch.zeros_like(x) + self.param


@pytest.fixture
def setup():
    return ModeleTracable(), NoiseScheduler(timesteps=10, device="cpu")


def test_sample_forme_et_finitude(setup):
    model, sched = setup

    x = sample(model, sched, n_samples=3, channels=3, size=32)

    assert x.shape == (3, 3, 32, 32)
    assert torch.isfinite(x).all()


def test_sample_fait_un_appel_reseau_par_pas(setup):
    model, sched = setup

    sample(model, sched, n_samples=2)

    assert model.appels == sched.timesteps


def test_sample_restaure_le_mode_entrainement(setup):
    model, sched = setup
    model.train()

    sample(model, sched, n_samples=2)

    assert model.training is True


# --- sample_trajectory -------------------------------------------------------

def test_trajectory_renvoie_les_instantanes_demandes(setup):
    model, sched = setup
    steps = [9, 5, 0]

    snaps = sample_trajectory(model, sched, steps_to_save=steps,
                              n_samples=2, progress=False)

    indices = [t for t, _ in snaps]
    assert indices == [9, 5, 0, None]


def test_trajectory_dernier_instantane_est_la_sortie_finale(setup):
    model, sched = setup

    torch.manual_seed(7)
    snaps = sample_trajectory(model, sched, steps_to_save=[], n_samples=2, progress=False)
    torch.manual_seed(7)
    x_final = sample(model, sched, n_samples=2)

    t_final, x_traj = snaps[-1]
    assert t_final is None
    assert torch.allclose(x_traj, x_final, atol=1e-6)
