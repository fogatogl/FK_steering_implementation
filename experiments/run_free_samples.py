"""Free sampling + reward: the data behind figure 0.

Scores go to results/, images to samples/ (S3). `--rescore` rewrites the JSON
from the .pt without a GPU: use it when sigma_ref changes in smc/rewards.py.
From the root: `python -m experiments.run_free_samples`.
"""
import argparse
import json
from pathlib import Path

import torch

from smc.models import CifarDDPM
from smc.rewards import reward
from smc.rng import make_generator
from smc.scheduler import NoiseScheduler
from smc.unet import UNet

ROOT = Path(__file__).resolve().parent.parent


def load_model(weights_path, device, ema=True):
    """EMA weights by default: those of the notebook FID (79.2 / 50.7) and of
    everything after 17/09. The lambda sweep and figures 0-2 ran on the raw
    weights (`ema=False`), see docs/decisions.md."""
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    unet = UNet(in_channels=3, n_feat=config["n_feat"]).to(device)
    unet.load_state_dict(ckpt["ema"] if ema else ckpt["model"])
    unet.eval()
    scheduler = NoiseScheduler(timesteps=config["timesteps"],
                               beta_start=config["beta1"], beta_end=config["beta2"],
                               device=device)
    return CifarDDPM(unet=unet, scheduler=scheduler, device=device), config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--weights", default="/home/onyxia/work/ddpm/weights/ddpm_last.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tag", default="free")
    p.add_argument("--no-ema", action="store_true", help="raw weights, as before 17/09")
    p.add_argument("--rescore", action="store_true",
                   help="rewrite the JSON from the existing .pt, no sampling")
    args = p.parse_args()

    images_path = ROOT / "samples" / f"{args.tag}_seed{args.seed}.pt"
    json_path = ROOT / "results" / f"{args.tag}_samples_seed{args.seed}.json"

    if args.rescore:
        x = torch.load(images_path, map_location="cpu")
        old = json.loads(json_path.read_text())
        config, timesteps = old["config"], old["timesteps"]
        device = old.get("device", "unknown")
    else:
        ddpm, config = load_model(args.weights, args.device, ema=not args.no_ema)
        gen = make_generator(args.seed, device=args.device)
        state = ddpm.initial_state(args.n, generator=gen)
        for t_idx in reversed(range(ddpm.scheduler.timesteps)):
            state = ddpm.step(state, t_idx, generator=gen)
        x = state["x"].cpu()
        timesteps, device = ddpm.scheduler.timesteps, args.device
        Path(ROOT / "samples").mkdir(exist_ok=True)
        torch.save(x, images_path)

    r = reward(x).cpu()

    json_path.write_text(json.dumps({
        "tag": args.tag,
        "seed": args.seed,
        "n": len(x),
        "timesteps": timesteps,
        "model_calls": len(x) * timesteps,
        "device": device,
        "weights": args.weights,
        "ema": not args.no_ema,
        "config": {k: v for k, v in config.items() if isinstance(v, (int, float, str))},
        "reward": r.tolist(),
        "images": str(images_path.relative_to(ROOT)),
    }, indent=2))

    print(f"{json_path}  mean={r.mean():.4f} std={r.std():.4f} "
          f"min={r.min():.4f} max={r.max():.4f}")


if __name__ == "__main__":
    main()
