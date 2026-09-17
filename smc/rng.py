"""One generator per experiment, created on the model's device.

PyTorch refuses to pair a generator and a tensor living on different devices,
and a bare `torch.Generator()` is a CPU generator: the trap only shows up on
GPU. No function in `smc/` creates a generator, it receives one — or `None`,
and then draws from the global RNG.
"""
import torch


def make_generator(seed, device="cpu"):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def check_generator(generator, device):
    if generator is None:
        return
    expected = torch.device(device).type
    got = generator.device.type
    if got != expected:
        raise ValueError(f"generator on '{got}' for a computation on '{expected}': "
                         f"use make_generator(seed, device='{expected}')")
