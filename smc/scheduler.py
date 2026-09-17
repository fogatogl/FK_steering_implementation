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
