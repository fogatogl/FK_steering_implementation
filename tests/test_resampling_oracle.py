import numpy as np
import particles.resampling as pr
import torch

from smc.resampling import resample_multinomial, resample_systematic


def test_oracle_resample_multinomial_frequencies():
    w_torch = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float32)
    w_np = np.array([0.5, 0.25, 0.25], dtype=np.float64)
    k = 4000

    gen = torch.Generator().manual_seed(42)
    idx_torch = resample_multinomial(w_torch, k, generator=gen)
    freqs_torch = torch.bincount(idx_torch, minlength=3).float() / k

    np.random.seed(42)
    idx_oracle = pr.multinomial(w_np, M=k)
    freqs_oracle = torch.tensor(np.bincount(idx_oracle, minlength=3) / k, dtype=torch.float32)

    assert torch.allclose(freqs_torch, freqs_oracle, atol=0.02)


def test_oracle_resample_systematic_frequencies():
    w_torch = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float32)
    w_np = np.array([0.5, 0.25, 0.25], dtype=np.float64)
    k = 4000

    gen = torch.Generator().manual_seed(42)
    idx_torch = resample_systematic(w_torch, k, generator=gen)
    freqs_torch = torch.bincount(idx_torch, minlength=3).float() / k

    np.random.seed(42)
    idx_oracle = pr.systematic(w_np, M=k)
    freqs_oracle = torch.tensor(np.bincount(idx_oracle, minlength=3) / k, dtype=torch.float32)
    assert torch.allclose(freqs_torch, freqs_oracle, atol=2.0 / k)