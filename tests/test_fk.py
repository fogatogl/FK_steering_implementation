import pytest
import torch
from smc.fk import fk_steer


class ControlledDummyModel:
    def __init__(self, timesteps):
        self.timesteps = list(timesteps)

    def initial_state(self, k, generator=None):
        return {"x": torch.linspace(1.0, 2.0, steps=k)}

    def step(self, state, t, generator=None):
        return {"x": state["x"] * 1.15}

    def predict_x0(self, state):
        return state["x"] * 1.1


def dummy_reward(x):
    return 0.5 * x


def accumulate_along_lineage(logG_stack, ancestors_stack):
    T, k = logG_stack.shape
    device = logG_stack.device
    curr_slots = torch.arange(k, device=device)
    total_logg = torch.zeros(k, device=device)

    for step in reversed(range(T)):
        curr_slots = ancestors_stack[step, curr_slots]
        total_logg += logG_stack[step, curr_slots]

    return total_logg, curr_slots


@pytest.mark.parametrize(
    "potential,lam",
    [
        ("difference", 3.5),
        ("max", 3.5),
        ("sum", 0.8),
    ],
)
def test_telescoping_with_active_non_collapsed_resampling(potential, lam):
    """Vérifie l'invariant télescopique avec brassage généalogique contrôlé."""
    k = 16
    timesteps = [4, 3, 2, 1, 0]
    gen = torch.Generator().manual_seed(0)

    model = ControlledDummyModel(timesteps)
    final_x, out = fk_steer(
        model=model,
        reward=dummy_reward,
        k=k,
        lam=lam,
        potential=potential,
        generator=gen,
        resample_threshold=0.7,
    )

    # 1. Vérification que le rééchantillonnage s'est bien produit
    assert out["n_resamplings"] > 0, f"Aucun rééchantillonnage pour {potential}"

    # 2. Reconstitution le long de la généalogie
    lineage_logg, initial_ancestors = accumulate_along_lineage(
        out["logG"], out["ancestors"]
    )

    # 3. Garde anti-test décoratif : plusieurs lignées distinctes doivent survivre
    n_surviving = torch.unique(initial_ancestors).numel()
    assert 1 < n_surviving < k, (
        f"Généalogie non discriminante pour {potential} (survivants : {n_surviving}/{k})."
    )

    # 4. Vérification de l'invariant \prod G_t == exp(\lambda r(x_0))
    expected = lam * dummy_reward(final_x)
    torch.testing.assert_close(
        lineage_logg,
        expected,
        atol=1e-5,
        rtol=1e-5,
        msg=f"Échec de l'invariant télescopique pour {potential}",
    )


def test_invariant_lambda_zero():
    """Vérifie qu'à lambda=0, la dynamique reste strictement neutre."""
    k = 8
    timesteps = [3, 2, 1, 0]
    gen = torch.Generator().manual_seed(0)

    model = ControlledDummyModel(timesteps)
    _, out = fk_steer(
        model=model,
        reward=dummy_reward,
        k=k,
        lam=0.0,
        potential="difference",
        generator=gen,
        resample_threshold=0.5,
    )

    torch.testing.assert_close(
        out["logG"], torch.zeros_like(out["logG"]), atol=1e-6, rtol=1e-6
    )
    assert out["n_resamplings"] == 0