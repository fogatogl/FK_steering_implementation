import torch


def resample_multinomial(w, k, generator=None):
    """`k` ancestor indices drawn from normalised `w`, with replacement."""
    return torch.multinomial(w, k, replacement=True, generator=generator)


def resample_systematic(w, k, generator=None):
    """`k` ancestor indices from a regular comb. `w` must sum to 1."""
    u = torch.rand(1, generator=generator, device=w.device).item() / k
    cumw = torch.cumsum(w, dim=0)

    # In fp32 the cumsum sometimes stops short of 1: the last comb point then
    # falls beyond it and searchsorted returns k, out of bounds.
    cumw[-1] = 1.0

    points = u + torch.arange(start=0, end=k, dtype=w.dtype, device=w.device) / k
    idx = torch.searchsorted(cumw, points, right=False)
    return idx
