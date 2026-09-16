import torch
from abc import ABC, abstractmethod

from smc.rng import check_generator


class DiffusionModel(ABC):
    @abstractmethod
    def initial_state(self, k, generator): ...

    @abstractmethod
    def step(self, state, t, generator): ...

    @abstractmethod
    def predict_x0(self, state, t=None): ...


class CifarDDPM(DiffusionModel):
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
        # eps a été prédit depuis x : on les garde appariés, avec leur date t.
        return {"x": x_prev, "eps": eps, "x_t": x, "t": t}

    def predict_x0(self, state, t=None, clamp=True):
        """x̂0 = (x_t − √(1−ᾱ_t)·ε̂_t) / √ᾱ_t, sur `x_t` (entrée du dernier step)
        et non sur sa sortie x_{t-1}, puisque ε̂ a été prédit depuis x_t.

        `clamp=True` borne à [-1, 1], le domaine des images : c'est cette
        valeur que la reward score à chaque pas.
        """
        if state["eps"] is None:
            raise ValueError("predict_x0 exige un `eps` : appeler step() d'abord.")

        t_etat = state.get("t")
        if t is None:
            t = t_etat
        elif t_etat is not None and t != t_etat:
            raise ValueError(
                f"predict_x0 appelé avec t={t} sur un état daté de t={t_etat}."
            )
        if t is None:
            raise ValueError("predict_x0 exige un `t`, absent de l'état et des arguments.")

        # x_t absent d'un état construit à la main : `x` est alors l'état au pas t.
        x_t = state.get("x_t") if state.get("x_t") is not None else state["x"]
        eps = state["eps"]
        x0 = (x_t - self.scheduler.sqrt_one_minus_alpha_hat[t] * eps) / self.scheduler.sqrt_alpha_hat[t]
        return x0.clamp(-1.0, 1.0) if clamp else x0
