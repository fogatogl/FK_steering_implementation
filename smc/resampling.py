import torch


def resample_multinomial(w, k, generator=None):
    return torch.multinomial(w, k, replacement=True, generator=generator)


def resample_systematic(w, k, generator=None):
    u = torch.rand(1, generator=generator, device=w.device).item() / k
    cumw = torch.cumsum(w, dim=0)
    points = u + torch.arange(start=0, end=k, dtype=w.dtype, device=w.device) / k
    idx = torch.searchsorted(cumw, points, right=False)
    return idx