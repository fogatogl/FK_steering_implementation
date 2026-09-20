import torch
from abc import ABC, abstractmethod

from diffusers.utils.torch_utils import randn_tensor

from smc.rng import check_generator

class DiffusionModel(ABC):
    @property
    @abstractmethod
    def T(self): ...

    @property
    @abstractmethod
    def timesteps(self): ...

    @abstractmethod
    def initial_state(self, k, generator): ...

    @abstractmethod
    def step(self, state, t, generator): ...

    @abstractmethod
    def predict_x0(self, state, t=None): ...


class CifarDDPM(DiffusionModel):
    @property
    def T(self):
        return len(self.scheduler.schedule)

    @property
    def timesteps(self):
        return self.scheduler.schedule

    def __init__(self, unet, scheduler, device):
        self.unet = unet
        self.scheduler = scheduler
        self.device = device

    def initial_state(self, k, generator):
        check_generator(generator, self.device)
        x = torch.randn(k, 3, 32, 32, generator=generator, device=self.device)
        return {"x": x, "eps": None, "x_t": None, "t": None}

    def step(self, state, t, generator):
        x = state["x"]
        with torch.no_grad():
            x_prev, eps = self.scheduler.sample_step(self.unet, x, t, generator=generator)
        # eps was predicted from x: keep them paired, with their timestep.
        return {"x": x_prev, "eps": eps, "x_t": x, "t": t}

    def predict_x0(self, state, t=None, clamp=True):
        """Tweedie: x̂0 = (x_t − √(1−ᾱ_t) ε̂) / √ᾱ_t, clamped to the image range by default."""
        if t is None:
            t = state["t"]
        elif state.get("t") is not None and t != state["t"]:
            raise ValueError(f"t={t} but the state is dated t={state['t']}")
        # A hand-built state may have no x_t: x is then the state at step t.
        x_t = state["x_t"] if state.get("x_t") is not None else state["x"]
        x0 = (x_t - self.scheduler.sqrt_one_minus_alpha_hat[t] * state["eps"]) / self.scheduler.sqrt_alpha_hat[t]
        return x0.clamp(-1.0, 1.0) if clamp else x0


class StableDiffusion:
    """Même contrat que CifarDDPM : état = {x, eps, x_t, t}. Le texte ne sort jamais de cet objet."""

    def __init__(self, pipe, guidance_scale=7.5, num_steps=100, eta=1.0):
        self.pipe = pipe
        self.unet, self.vae, self.scheduler = pipe.unet, pipe.vae, pipe.scheduler
        self.guidance, self.eta = guidance_scale, eta
        self.device, self.dtype = pipe.device, pipe.unet.dtype

        self.scheduler.set_timesteps(num_steps, device=self.device)
        self.timesteps = self.scheduler.timesteps
        self.text_emb = None

    def set_prompt(self, prompt, negative_prompt=""):
        cond, uncond = self.pipe.encode_prompt(
            prompt, self.device, num_images_per_prompt=1,
            do_classifier_free_guidance=True, negative_prompt=negative_prompt,
        )
        self.text_emb = (uncond, cond)

    def initial_state(self, k, generator):
        check_generator(generator, self.device)
        shape = (k, self.unet.config.in_channels, 64, 64)
        x = randn_tensor(shape, generator=generator, device=self.device, dtype=self.dtype)
        x = x * self.scheduler.init_noise_sigma
        return {"x": x, "eps": None, "x_t": None, "t": None}

    @torch.no_grad()
    def step(self, state, t, generator):
        assert self.text_emb is not None, "appelle set_prompt d'abord"
        x = state["x"]
        k = x.shape[0]
        uncond, cond = self.text_emb
        emb = torch.cat([uncond.expand(k, -1, -1), cond.expand(k, -1, -1)])

        x_in = self.scheduler.scale_model_input(torch.cat([x, x]), t)
        eps_u, eps_c = self.unet(x_in, t, encoder_hidden_states=emb).sample.chunk(2)
        eps = eps_u + self.guidance * (eps_c - eps_u)

        out = self.scheduler.step(eps, t, x, eta=self.eta, generator=generator)
        return {"x": out.prev_sample, "eps": eps, "x_t": x, "t": int(t)}

    def predict_x0(self, state):
        a = self.scheduler.alphas_cumprod[state["t"]].to(state["x_t"])
        return (state["x_t"] - (1 - a).sqrt() * state["eps"]) / a.sqrt()   # latent, pas de clamp