import torch
from smc.resampling import resample_multinomial, resample_systematic
from smc.weights import normalize_logw


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


# Poids dont la cumsum fp32 s'arrête à 1 − 1,9e−6 : le dernier point du peigne
# passait alors au-delà du dernier bord et searchsorted renvoyait k.
LOGW_CUMSUM_DEFICIENTE = torch.tensor(
    [33.247059, -13.070757, 2.053682, 29.95665, 1.97585, 14.369114, -3.183539, 5.904665]
)


def test_peigne_reste_dans_les_bornes_quand_la_cumsum_rate_un(monkeypatch):
    w, _ = normalize_logw(LOGW_CUMSUM_DEFICIENTE)
    k = w.numel()
    monkeypatch.setattr(torch, "rand", lambda *a, **kw: torch.tensor([1.0 - 2.0 ** -24]))

    assert int(resample_systematic(w, k).max()) < k
