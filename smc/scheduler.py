import torch


class NoiseScheduler:
    def __init__(self, timesteps, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.device = device

        self.beta = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumsum(self.alpha.log(), dim=0).exp()

        # Forward
        self.sqrt_alpha_hat = self.alpha_hat.sqrt()
        self.sqrt_one_minus_alpha_hat = (1.0 - self.alpha_hat).sqrt()

        # Reverse
        self.sqrt_inv_alpha = (1.0 / self.alpha).sqrt()
        self.eps_coef = self.beta / self.sqrt_one_minus_alpha_hat
        self.sigma = self.beta.sqrt()

        self.schedule = list(range(timesteps - 1, -1, -1))

    def add_noise(self, x_0, t, noise=None):
        """q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_0)
        s1 = self.sqrt_alpha_hat[t].view(-1, 1, 1, 1)
        s2 = self.sqrt_one_minus_alpha_hat[t].view(-1, 1, 1, 1)
        return s1 * x_0 + s2 * noise, noise

    @torch.no_grad()
    def sample_step(self, model, x_t, t_idx, generator=None):
        """One step of p(x_{t-1} | x_t)."""
        t = torch.full((x_t.shape[0],), t_idx, device=self.device, dtype=torch.long)
        pred_noise = model(x_t, t)
        mean = self.sqrt_inv_alpha[t_idx] * (x_t - self.eps_coef[t_idx] * pred_noise)
        if t_idx > 0:
            return mean + self.sigma[t_idx] * torch.randn(x_t.shape, generator=generator, device=x_t.device), pred_noise
        return mean, pred_noise


class DDIMScheduler(NoiseScheduler):
    """Song, Meng & Ermon 2021. Same alpha_hat tables as the DDPM, a sub-sequence
    of `steps` timesteps, and the eta-parametrised reverse step (their eq. 12).
    eta=1 on the full schedule gives the DDPM posterior mean with the beta-tilde
    variance; the DDPM above uses beta, so only the means coincide."""

    def __init__(self, timesteps, steps, eta, beta_start=1e-4, beta_end=0.02, device="cpu"):
        super().__init__(timesteps, beta_start, beta_end, device)
        self.eta = eta
        # Uniform spacing; the paper uses quadratic on CIFAR-10. `unique`
        # removes the duplicates rounding can create when steps is close to T.
        tau = torch.linspace(0, timesteps - 1, steps).round().long().unique()
        self.schedule = tau.flip(0).tolist()
        # t -> the timestep that follows it; None after the last one, where
        # alpha_hat_prev is 1 and x_s is x0_hat itself.
        self.prev = dict(zip(self.schedule, self.schedule[1:] + [None]))

    @torch.no_grad()
    def sample_step(self, model, x_t, t_idx, generator=None):
        """One step of x_t -> x_s with s = self.prev[t_idx]. Returns (x_s, eps)."""
        s = self.prev[t_idx]
        t = torch.full((x_t.shape[0],), t_idx, device=self.device, dtype=torch.long)
        eps = model(x_t, t)

        # 1. ᾱ_t et estimation du point propre x̂0
        alpha_hat_t = self.alpha_hat[t_idx]
        sqrt_alpha_hat_t = self.sqrt_alpha_hat[t_idx]
        sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t_idx]

        x0_hat = (x_t - sqrt_one_minus_alpha_hat_t * eps) / sqrt_alpha_hat_t

        # 2. Dernier pas (t_idx == 0 -> s is None) : ᾱ_s = 1, x_s = x̂0 pur
        if s is None:
            return x0_hat, eps

        # 3. Étape intermédiaire : extraction de ᾱ_s
        alpha_hat_s = self.alpha_hat[s]

        # 4. Calcul de σ_s avec clamp contre instabilités numériques
        # σ_s = η * √((1 - ᾱ_s) / (1 - ᾱ_t)) * √(1 - ᾱ_t / ᾱ_s)
        sigma_sq = (
            (self.eta**2)
            * ((1.0 - alpha_hat_s) / (1.0 - alpha_hat_t))
            * (1.0 - alpha_hat_t / alpha_hat_s)
        )
        sigma = torch.sqrt(torch.clamp(sigma_sq, min=0.0))

        # 5. Direction pointant vers x_t : √(1 - ᾱ_s - σ_s²)
        # clamp(min=0.0) indispensable lorsque η -> 1 où le terme tend vers 0 par le bas
        dir_xt_sq = torch.clamp(1.0 - alpha_hat_s - sigma**2, min=0.0)
        dir_xt = torch.sqrt(dir_xt_sq) * eps

        # 6. Assemblage de l'équation (12)
        x_s = torch.sqrt(alpha_hat_s) * x0_hat + dir_xt

        # 7. Bruit résiduel stochastique
        if self.eta > 0.0:
            noise = torch.randn(
                x_t.shape, generator=generator, device=x_t.device, dtype=x_t.dtype
            )
            x_s = x_s + sigma * noise

        return x_s, eps