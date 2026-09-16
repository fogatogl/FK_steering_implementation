import torch
from abc import ABC, abstractmethod


class DiffusionModel(ABC):
    @abstractmethod
    def initial_state(self, k, generator): ...

    @abstractmethod
    def step(self, state, t, generator): ...

    @abstractmethod
    def predict_x0(self, state, t): ...


class CifarDDPM(DiffusionModel):
    def __init__(self, unet, scheduler, device):
        self.unet = unet
        self.scheduler = scheduler
        self.device = device

    def initial_state(self, k, generator):
        x = torch.randn(k, 3, 32, 32, generator=generator, device=self.device)
        return {"x": x, "eps": None}

    def step(self, state, t, generator):
        x = state["x"]
        with torch.no_grad():
            x_prev, eps = self.scheduler.sample_step(self.unet, x, t)
        return {"x": x_prev, "eps": eps}

    def predict_x0(self, state, t):
        x = state["x"]
        eps = state["eps"]
        return (x - self.scheduler.sqrt_one_minus_alpha_hat[t]*eps) / self.scheduler.sqrt_alpha_hat[t]