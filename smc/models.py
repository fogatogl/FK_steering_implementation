import torch
from abc import ABC, abstractmethod

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
