import pytest
import torch
import torch.nn as nn

from smc.ema import EMA


class PetitModele(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 2, bias=True)
        # Buffer entier : EMA doit l'ignorer (filtre sur is_floating_point).
        self.register_buffer("compteur", torch.tensor(0, dtype=torch.long))


@pytest.fixture
def modele():
    m = PetitModele()
    with torch.no_grad():
        m.lin.weight.fill_(1.0)
        m.lin.bias.fill_(0.0)
    return m


def test_buffers_non_flottants_exclus(modele):
    ema = EMA(modele, decay=0.9)

    assert "compteur" not in ema.shadow


def test_update_applique_la_formule_exponentielle(modele):
    decay = 0.9
    ema = EMA(modele, decay=decay)

    with torch.no_grad():
        modele.lin.weight.fill_(2.0)
    ema.update(modele)

    attendu = decay * 1.0 + (1 - decay) * 2.0
    assert torch.allclose(ema.shadow["lin.weight"],
                          torch.full_like(modele.lin.weight, attendu), atol=1e-6)


def test_copy_to_ecrit_le_shadow_dans_le_modele(modele):
    ema = EMA(modele, decay=0.9)
    with torch.no_grad():
        modele.lin.weight.fill_(2.0)
    ema.update(modele)
    valeur_ema = ema.shadow["lin.weight"].clone()

    ema.copy_to(modele)

    assert torch.allclose(modele.lin.weight, valeur_ema, atol=1e-6)


def test_state_dict_round_trip(modele):
    ema = EMA(modele, decay=0.9)
    with torch.no_grad():
        modele.lin.weight.fill_(2.0)
    ema.update(modele)
    sauvegarde = {k: v.clone() for k, v in ema.state_dict().items()}

    nouvelle = EMA(PetitModele(), decay=0.9)
    nouvelle.load_state_dict(sauvegarde)

    assert nouvelle.shadow.keys() == ema.shadow.keys()
    for k in sauvegarde:
        assert torch.allclose(nouvelle.shadow[k], sauvegarde[k], atol=1e-6)
