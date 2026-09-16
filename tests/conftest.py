"""Fixtures partagées : accès aux poids réels (hors dépôt, cf. .gitignore)."""
import os
from pathlib import Path

import pytest
import torch


def weights_dir() -> Path:
    """Répertoire des poids entraînés.

    Ils ne sont pas versionnés (S3 est la source de vérité, cf.
    scripts/sync_s3.sh) : on autorise un surchargement par variable
    d'environnement pour les machines où le projet n'est pas dans
    /home/onyxia/work/ddpm.
    """
    return Path(os.environ.get("DDPM_WEIGHTS_DIR", "/home/onyxia/work/ddpm/weights"))


def require_checkpoint(relative_path: str) -> Path:
    path = weights_dir() / relative_path
    if not path.exists():
        pytest.skip(f"Poids absents ({path}) : `scripts/sync_s3.sh pull` pour les récupérer")
    return path


@pytest.fixture(scope="session")
def checkpoint_path() -> Path:
    return require_checkpoint("ddpm_last.pt")


@pytest.fixture(scope="session")
def checkpoint(checkpoint_path):
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


@pytest.fixture(scope="session")
def real_unet(checkpoint):
    """UNet entraîné, en eval, sur CPU."""
    from smc.unet import UNet

    unet = UNet(in_channels=3, n_feat=checkpoint["config"]["n_feat"])
    unet.load_state_dict(checkpoint["model"])
    unet.eval()
    return unet
