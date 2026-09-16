import pytest
import torch

from smc.unet import UNet, gn


# n_feat=64 : plus petite largeur où gn() laisse >= 2 canaux par groupe partout,
# donc où le conditionnement temporel survit (cf.
# test_conditionnement_temporel_mort_si_un_seul_canal_par_groupe).
N_FEAT = 64


@pytest.fixture
def unet():
    torch.manual_seed(0)
    return UNet(in_channels=3, n_feat=N_FEAT).eval()


@pytest.fixture
def unet_perturbe(unet):
    """out_conv est initialisé à zéro : on le réveille pour tester le reste."""
    with torch.no_grad():
        unet.out_conv.weight.normal_(0.0, 0.1)
        unet.out_conv.bias.normal_(0.0, 0.1)
    return unet


def test_forward_preserve_la_forme(unet):
    x = torch.randn(2, 3, 32, 32)
    t = torch.tensor([10, 900])

    with torch.no_grad():
        y = unet(x, t)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_elements_du_batch_independants(unet_perturbe):
    x = torch.randn(3, 3, 32, 32)
    t = torch.tensor([0, 5, 900])

    with torch.no_grad():
        y_batch = unet_perturbe(x, t)
        y_seul = unet_perturbe(x[1:2], t[1:2])

    assert torch.allclose(y_batch[1], y_seul[0], atol=1e-5)


@pytest.mark.parametrize("n_feat", [64, 128])
def test_conditionnement_temporel_vivant_aux_largeurs_utilisables(n_feat):
    torch.manual_seed(0)
    unet = UNet(in_channels=3, n_feat=n_feat).eval()
    with torch.no_grad():
        unet.out_conv.weight.normal_(0.0, 0.1)
    x = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        y_tot = unet(x, torch.tensor([0, 0]))
        y_tard = unet(x, torch.tensor([999, 999]))

    assert (y_tot - y_tard).abs().max() > 0.1


def test_conditionnement_temporel_mort_si_un_seul_canal_par_groupe():
    """Limite connue : sous n_feat=32, le réseau ignore totalement t.

    ResidualBlock ajoute time_proj (constant par canal) *avant* norm2. Quand
    gn() retombe à un canal par groupe, la GroupNorm devient une InstanceNorm
    par canal et retranche exactement ce décalage : le signal temporel est
    annulé. Ce test fige la limite pour qu'un futur n_feat trop petit ne passe
    pas inaperçu -- n_feat=128 (le modèle entraîné) n'est pas concerné.
    """
    assert gn(16).num_groups == 16  # 1 canal / groupe
    torch.manual_seed(0)
    unet = UNet(in_channels=3, n_feat=16).eval()
    with torch.no_grad():
        unet.out_conv.weight.normal_(0.0, 0.1)
    x = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        y_tot = unet(x, torch.tensor([0, 0]))
        y_tard = unet(x, torch.tensor([999, 999]))

    assert torch.allclose(y_tot, y_tard, atol=1e-4)
