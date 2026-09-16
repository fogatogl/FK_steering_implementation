"""Intégrité des poids entraînés.

Les poids ne sont pas versionnés (cf. .gitignore) : ils vivent hors du dépôt et
sont miroités sur S3 par scripts/sync_s3.sh. Ces tests se *skippent* proprement
là où ils sont absents (CI, autre machine) -- ils ne doivent jamais échouer pour
cette raison. Surcharger le chemin avec DDPM_WEIGHTS_DIR si besoin.
"""
import pytest
import torch

from smc.scheduler import NoiseScheduler
from smc.unet import UNet
from tests.conftest import require_checkpoint

CLES_ATTENDUES = {"epoch", "model", "optimizer", "scaler", "ema", "history", "config"}
CLES_CONFIG = {"n_feat", "timesteps", "beta1", "beta2"}


def test_checkpoint_present_et_lisible(checkpoint_path, checkpoint):
    assert checkpoint_path.is_file()
    assert checkpoint_path.stat().st_size > 100 * 1024**2  # ~129 Mo
    assert CLES_ATTENDUES.issubset(checkpoint.keys())


def test_config_du_checkpoint(checkpoint):
    config = checkpoint["config"]

    assert CLES_CONFIG.issubset(config.keys())
    assert config["n_feat"] == 128
    assert config["timesteps"] == 1000
    assert config["beta1"] == pytest.approx(1e-4)
    assert config["beta2"] == pytest.approx(0.02)


def test_poids_chargeables_strictement(checkpoint):
    """La state_dict doit entrer dans UNet sans clé manquante ni superflue."""
    unet = UNet(in_channels=3, n_feat=checkpoint["config"]["n_feat"])

    manquantes, superflues = unet.load_state_dict(checkpoint["model"], strict=True)

    assert manquantes == []
    assert superflues == []


def test_poids_finis_et_non_degeneres(checkpoint):
    for nom, tenseur in checkpoint["model"].items():
        assert torch.isfinite(tenseur).all(), f"NaN/Inf dans {nom}"
    # Le réseau a bien appris : out_conv est zero-init, il ne doit plus l'être.
    assert checkpoint["model"]["out_conv.weight"].abs().max() > 0.0


def test_shadow_ema_aligne_sur_le_modele(checkpoint):
    cles_modele = {k for k, v in checkpoint["model"].items() if v.dtype.is_floating_point}

    assert set(checkpoint["ema"].keys()) == cles_modele
    for nom, tenseur in checkpoint["ema"].items():
        assert tenseur.shape == checkpoint["model"][nom].shape
        assert torch.isfinite(tenseur).all(), f"NaN/Inf dans ema[{nom}]"


def test_modele_entraine_predit_un_bruit_plausible(real_unet):
    """Sur un x_t construit par le forward, la prédiction doit corréler au bruit."""
    sched = NoiseScheduler(timesteps=1000, device="cpu")
    torch.manual_seed(0)
    x0 = torch.rand(4, 3, 32, 32) * 2 - 1  # domaine [-1, 1] de l'entraînement
    bruit = torch.randn_like(x0)
    t = torch.full((4,), 500, dtype=torch.long)
    x_t, _ = sched.add_noise(x0, t, noise=bruit)

    with torch.no_grad():
        eps_pred = real_unet(x_t, t)

    correlation = torch.corrcoef(torch.stack([eps_pred.flatten(), bruit.flatten()]))[0, 1]
    assert correlation > 0.5


def test_checkpoint_finetune_class3_present_et_coherent(checkpoint):
    """Le second jeu de poids (fine-tuning classe 3) doit rester compatible."""
    chemin = require_checkpoint("ft_class3/ddpm_last.pt")
    ft = torch.load(chemin, map_location="cpu", weights_only=False)

    assert CLES_ATTENDUES.issubset(ft.keys())
    assert ft["config"]["n_feat"] == checkpoint["config"]["n_feat"]
    assert set(ft["model"].keys()) == set(checkpoint["model"].keys())

    unet = UNet(in_channels=3, n_feat=ft["config"]["n_feat"])
    unet.load_state_dict(ft["model"], strict=True)

    # Le fine-tuning a bien bougé les poids par rapport au modèle de base.
    ecart = max((ft["model"][k] - checkpoint["model"][k]).abs().max().item()
                for k in ft["model"])
    assert ecart > 0.0


@pytest.mark.slow
def test_equivalence_sample_et_cifar_ddpm_sur_modele_reel(real_unet):
    """sample() et la boucle CifarDDPM doivent produire la même trajectoire.

    Schedule raccourci : l'équivalence est structurelle, elle ne dépend pas du
    nombre de pas, et 1000 pas sur CPU seraient inutilement longs.
    """
    from smc.models import CifarDDPM
    from smc.sampling import sample

    sched = NoiseScheduler(timesteps=20, device="cpu")
    ddpm = CifarDDPM(unet=real_unet, scheduler=sched, device="cpu")
    k, seed = 2, 2024

    torch.manual_seed(seed)
    x_ref = sample(real_unet, sched, n_samples=k, channels=3, size=32)

    torch.manual_seed(seed)
    state = ddpm.initial_state(k=k, generator=None)
    for t_idx in reversed(range(sched.timesteps)):
        state = ddpm.step(state, t_idx, generator=None)

    assert torch.allclose(x_ref, state["x"], atol=1e-5)
