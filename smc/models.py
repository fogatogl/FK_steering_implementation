import torch
from abc import ABC, abstractmethod


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
        x = torch.randn(k, 3, 32, 32, generator=generator, device=self.device)
        return {"x": x, "eps": None, "x_t": None, "t": None}

    def step(self, state, t, generator):
        x = state["x"]
        with torch.no_grad():
            x_prev, eps = self.scheduler.sample_step(self.unet, x, t, generator=generator)
        # `x` est l'état au pas `t`, `x_prev` celui au pas `t-1`. `eps` a été
        # prédit à partir de `x` : on garde les deux appariés, sans quoi
        # predict_x0 mélangerait x_{t-1} et un epsilon daté de t.
        return {"x": x_prev, "eps": eps, "x_t": x, "t": t}

    def predict_x0(self, state, t=None):
        """Estimateur de Tweedie x̂0 = (x_t − √(1−ᾱ_t)·ε̂_t) / √ᾱ_t.

        Porte sur `x_t`, l'entrée du dernier `step`, et non sur `state["x"]`
        qui en est la sortie x_{t-1} : ε̂ ayant été prédit depuis x_t, les deux
        doivent porter le même `t`. `t` est déduit de l'état si on l'omet.
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

        # `x_t` est absent des états construits à la main : on retombe alors sur
        # `x`, qui est bien l'état au pas `t` puisque aucun step n'a eu lieu.
        x_t = state.get("x_t") if state.get("x_t") is not None else state["x"]
        eps = state["eps"]
        return (x_t - self.scheduler.sqrt_one_minus_alpha_hat[t] * eps) / self.scheduler.sqrt_alpha_hat[t]
