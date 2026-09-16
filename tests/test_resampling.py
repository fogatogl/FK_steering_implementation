import pytest
import torch

from smc.resampling import resample_multinomial, resample_systematic


def test_resample_multinomial_frequencies():
    w = torch.tensor([0.5, 0.25, 0.25])
    k = 4000
    gen = torch.Generator().manual_seed(42)

    idx = resample_multinomial(w, k, generator=gen)
    freqs = torch.bincount(idx, minlength=3).float() / k

    assert torch.allclose(freqs, w, atol=0.02)


def test_resample_systematic_frequencies():
    w = torch.tensor([0.5, 0.25, 0.25])
    k = 4000
    gen = torch.Generator().manual_seed(42)

    idx = resample_systematic(w, k, generator=gen)
    freqs = torch.bincount(idx, minlength=3).float() / k

    assert torch.allclose(freqs, w, atol=0.02)


def test_resample_systematic_lower_variance_than_multinomial():
    w = torch.tensor([0.5, 0.25, 0.25])
    k = 4000
    n_runs = 200

    gen_multi = torch.Generator().manual_seed(0)
    comptes_multi = [
        torch.bincount(resample_multinomial(w, k, generator=gen_multi), minlength=3)[0].item()
        for _ in range(n_runs)
    ]

    gen_syst = torch.Generator().manual_seed(0)
    comptes_syst = [
        torch.bincount(resample_systematic(w, k, generator=gen_syst), minlength=3)[0].item()
        for _ in range(n_runs)
    ]

    var_multi = torch.tensor(comptes_multi, dtype=torch.float).var()
    var_syst = torch.tensor(comptes_syst, dtype=torch.float).var()

    assert var_syst < var_multi


def test_resample_systematic_sanity_check_two_particles():
    w = torch.tensor([0.5, 0.5])
    k = 2
    gen = torch.Generator().manual_seed(42)

    idx = resample_systematic(w, k, generator=gen)

    assert torch.equal(idx, torch.tensor([0, 1]))


@pytest.mark.parametrize("resample", [resample_multinomial, resample_systematic])
def test_indices_valides(resample):
    w = torch.tensor([0.5, 0.25, 0.25])
    gen = torch.Generator().manual_seed(0)

    idx = resample(w, 10, generator=gen)

    assert idx.shape == (10,)
    assert idx.dtype == torch.long
    assert idx.min() >= 0
    assert idx.max() < w.shape[0]


@pytest.mark.parametrize("resample", [resample_multinomial, resample_systematic])
def test_poids_degeneres_ne_selectionnent_que_le_survivant(resample):
    w = torch.tensor([0.0, 1.0, 0.0])
    gen = torch.Generator().manual_seed(0)

    idx = resample(w, 8, generator=gen)

    assert torch.all(idx == 1)


def test_systematic_comptes_proches_de_l_esperance():
    """Le rééchantillonnage systématique borne |compte - k*w| a 1."""
    w = torch.tensor([0.5, 0.25, 0.25])
    k = 40

    for graine in range(10):
        gen = torch.Generator().manual_seed(graine)
        comptes = torch.bincount(resample_systematic(w, k, generator=gen), minlength=3).float()
        assert torch.all((comptes - k * w).abs() <= 1.0)
