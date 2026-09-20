import torch
import numpy as np
from PIL import Image


class ImageRewardSD:
    def __init__(self, vae, scorer, prompt, chunk=4):
        self.vae, self.scorer, self.prompt, self.chunk = vae, scorer, prompt, chunk

    @torch.no_grad()
    def decode(self, latents):
        imgs = []
        for z in latents.split(self.chunk):          # évite un pic VRAM à k grand
            img = self.vae.decode(z / self.vae.config.scaling_factor).sample
            imgs.append((img / 2 + 0.5).clamp(0, 1))
        return torch.cat(imgs)

    def __call__(self, latents):
        img = self.decode(latents)
        arr = (img.permute(0, 2, 3, 1).float().cpu().numpy() * 255).round().astype(np.uint8)
        pil = [Image.fromarray(a) for a in arr]
        scores = self.scorer.score(self.prompt, pil)
        if not isinstance(scores, list):        # k = 1 : float nu
            scores = [scores]
        return torch.tensor(scores, device=latents.device, dtype=torch.float32)