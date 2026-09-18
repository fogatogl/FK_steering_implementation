"""Pixel-space DDPMs from the Hub behind the same interface as the CIFAR model.

Google's DDPM checkpoints (celebahq-256, church-256, ...) share the CIFAR
training schedule — linear beta 1e-4 -> 0.02, T = 1000 — so the schedulers and
all of smc/ apply unchanged; only the network and the image shape differ. The
UNet is a black box eps_theta(x, t); fp16 autocast cuts memory 8.3 -> 5.4 GiB
at batch 16 on a T4.
"""
import torch
import torch.nn as nn

from smc.models import CifarDDPM
from smc.rng import check_generator
from smc.scheduler import DDIMScheduler, NoiseScheduler


class HubUNet(nn.Module):
    def __init__(self, repo, fp16=True):
        super().__init__()
        from diffusers import UNet2DModel
        self.unet = UNet2DModel.from_pretrained(repo)
        self.fp16 = fp16
        self.shape = (self.unet.config.in_channels, self.unet.config.sample_size,
                      self.unet.config.sample_size)

    def forward(self, x, t):
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16 and x.is_cuda):
            return self.unet(x, t).sample.float()


class HubDDPM(CifarDDPM):
    def initial_state(self, k, generator):
        check_generator(generator, self.device)
        x = torch.randn(k, *self.unet.shape, generator=generator, device=self.device)
        return {"x": x, "eps": None, "x_t": None, "t": None}


def load_hub_model(repo, device, steps=None, eta=0.0, fp16=True):
    unet = HubUNet(repo, fp16).to(device).eval()
    kw = dict(timesteps=1000, beta_start=1e-4, beta_end=0.02, device=device)
    scheduler = DDIMScheduler(steps=steps, eta=eta, **kw) if steps else NoiseScheduler(**kw)
    return HubDDPM(unet=unet, scheduler=scheduler, device=device)
