import os
from pathlib import Path

import pytest
import torch


def weights_dir():
    # Weights live outside the repo and are not distributed; DDPM_WEIGHTS_DIR points at them.
    return Path(os.environ.get("DDPM_WEIGHTS_DIR", "/home/onyxia/work/ddpm/weights"))


def require_checkpoint(relative_path):
    path = weights_dir() / relative_path
    if not path.exists():
        pytest.skip(f"weights missing ({path}): set DDPM_WEIGHTS_DIR")
    return path


@pytest.fixture(scope="session")
def checkpoint():
    return torch.load(require_checkpoint("ddpm_last.pt"),
                      map_location="cpu", weights_only=False)
