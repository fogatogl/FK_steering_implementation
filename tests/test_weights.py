import torch
from smc.weights import normalize_logw


def test_normalize_logw_numerical_stability():
    entree = torch.tensor([1000.0, 1001.0])
    w, _ = normalize_logw(entree)
    attendu = torch.tensor([0.2689, 0.7311])

    assert not torch.isnan(w).any()
    assert torch.allclose(w, attendu, atol=1e-3)


def test_normalize_logw_zeros_and_incremental():
    entree = torch.zeros(5)
    w, log_ell = normalize_logw(entree)
    attendu = torch.full((5,), 0.2)

    assert torch.allclose(w, attendu, atol=1e-6)
    assert torch.isclose(w.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(log_ell, torch.tensor(0.0), atol=1e-6)


def test_normalize_logw_inf():
    entree = torch.tensor([0.0, -float("inf")])
    w, _ = normalize_logw(entree)
    attendu = torch.tensor([1.0, 0.0])

    assert not torch.isnan(w).any()
    assert torch.allclose(w, attendu, atol=1e-6)