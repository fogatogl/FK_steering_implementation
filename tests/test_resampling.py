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