import os
from pathlib import Path

import pytest
import torch


def weights_dir():
    # Poids hors dépôt (S3 fait foi, cf. scripts/sync_s3.sh).
    return Path(os.environ.get("DDPM_WEIGHTS_DIR", "/home/onyxia/work/ddpm/weights"))


def require_checkpoint(relative_path):
    path = weights_dir() / relative_path
    if not path.exists():
        pytest.skip(f"Poids absents ({path}) : `scripts/sync_s3.sh pull`")
    return path


@pytest.fixture(scope="session")
def checkpoint():
    return torch.load(require_checkpoint("ddpm_last.pt"),
                      map_location="cpu", weights_only=False)
