import torch


def resample_multinomial(w, k, generator=None):
    """`k` indices d'ancêtres tirés selon `w` normalisé, avec remise."""
    return torch.multinomial(w, k, replacement=True, generator=generator)


def resample_systematic(w, k, generator=None):
    """`k` indices d'ancêtres par peigne régulier. `w` doit sommer à 1."""
    u = torch.rand(1, generator=generator, device=w.device).item() / k
    cumw = torch.cumsum(w, dim=0)

    # En fp32 la cumsum s'arrête parfois sous 1 : le dernier point du peigne
    # passe alors au-delà et searchsorted renvoie k, hors bornes.
    cumw[-1] = 1.0

    points = u + torch.arange(start=0, end=k, dtype=w.dtype, device=w.device) / k
    idx = torch.searchsorted(cumw, points, right=False)
    return idx
