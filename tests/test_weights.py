import torch
from smc.weights import normalize_logw, ess, should_resample


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


def test_ess_uniform():
    assert abs(ess(torch.full((8,), 1 / 8)) - 8.0) < 1e-5


def test_ess_degenerate():
    assert abs(ess(torch.tensor([1.0, 0.0, 0.0, 0.0])) - 1.0) < 1e-5


def test_should_resample():
    w_uniform = torch.full((8,), 1.0 / 8.0)
    w_degenerate = torch.tensor([1.0, 0.0, 0.0, 0.0])

    assert not should_resample(w_uniform, threshold=0.5)
    assert should_resample(w_degenerate, threshold=0.5)


def test_normalize_logw_invariant_par_translation():
    """Les poids normalisés ne dépendent pas d'une constante additive ;
    log_ell, lui, se décale exactement de cette constante."""
    logw = torch.tensor([0.0, 1.0, 2.0])

    w, log_ell = normalize_logw(logw)
    w_decale, log_ell_decale = normalize_logw(logw + 5.0)

    assert torch.allclose(w, w_decale, atol=1e-6)
    assert torch.isclose(log_ell_decale, log_ell + 5.0, atol=1e-5)


def test_normalize_logw_tous_les_poids_nuls_donne_nan():
    """Limite connue : si toutes les particules sont mortes (logW = -inf),
    la normalisation produit des NaN. L'appelant doit détecter log_ell = -inf
    avant de se servir de w."""
    w, log_ell = normalize_logw(torch.full((3,), -float("inf")))

    assert torch.isnan(w).all()
    assert log_ell == -float("inf")
