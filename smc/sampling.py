import torch
from tqdm.auto import tqdm


@torch.no_grad()
def sample_trajectory(model, scheduler, steps_to_save, n_samples=4,
                      channels=3, size=32, progress=True):
    was_training = model.training
    model.eval()

    x = torch.randn(n_samples, channels, size, size, device=scheduler.device)
    target = set(steps_to_save)
    snapshots = []

    it = reversed(range(scheduler.timesteps))
    if progress:
        it = tqdm(it, total=scheduler.timesteps, desc="sampling", leave=False)

    for t_idx in it:
        if t_idx in target:
            snapshots.append((t_idx, x.clone()))
        x, _ = scheduler.sample_step(model, x, t_idx)

    snapshots.append((None, x.clone()))
    model.train(was_training)
    return snapshots


@torch.no_grad()
def sample(model, scheduler, n_samples=16, channels=3, size=32):
    was_training = model.training
    model.eval()
    x = torch.randn(n_samples, channels, size, size, device=scheduler.device)
    for t_idx in tqdm(reversed(range(scheduler.timesteps)),
                      total=scheduler.timesteps, desc="sampling", leave=False):
        x, _ = scheduler.sample_step(model, x, t_idx)
    model.train(was_training)
    return x
