import torch


def reward(x):
    m = torch.mean(x, dim=(2, 3))
    # seed 2024, n=64
    sigma = 0.1962
    r = m[:, 0] - 0.5*(m[:, 1] + m[:, 2])
    return r/sigma



