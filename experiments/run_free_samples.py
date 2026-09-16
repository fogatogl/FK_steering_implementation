"""Échantillonnage libre + reward : les données de la figure 0.

Scores dans results/, images dans samples/ (S3). `--rescore` réécrit le JSON
depuis le .pt sans GPU : à passer quand σ_ref change dans smc/rewards.py.
Depuis la racine : `python -m experiments.run_free_samples`.
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

RACINE = Path(__file__).resolve().parent.parent


def charger_modele(chemin_poids, device):
    ckpt = torch.load(chemin_poids, map_location=device, weights_only=False)
    config = ckpt["config"]
    unet = UNet(in_channels=3, n_feat=config["n_feat"]).to(device)
    unet.load_state_dict(ckpt["model"])
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
    p.add_argument("--rescore", action="store_true",
                   help="réécrit le JSON depuis le .pt existant, sans échantillonner")
    args = p.parse_args()

    chemin_images = RACINE / "samples" / f"{args.tag}_seed{args.seed}.pt"
    chemin_json = RACINE / "results" / f"{args.tag}_samples_seed{args.seed}.json"

    if args.rescore:
        x = torch.load(chemin_images, map_location="cpu")
        ancien = json.loads(chemin_json.read_text())
        config, timesteps = ancien["config"], ancien["timesteps"]
        device = ancien.get("device", "inconnu")
    else:
        ddpm, config = charger_modele(args.weights, args.device)
        gen = make_generator(args.seed, device=args.device)
        state = ddpm.initial_state(args.n, generator=gen)
        for t_idx in reversed(range(ddpm.scheduler.timesteps)):
            state = ddpm.step(state, t_idx, generator=gen)
        x = state["x"].cpu()
        timesteps, device = ddpm.scheduler.timesteps, args.device
        Path(RACINE / "samples").mkdir(exist_ok=True)
        torch.save(x, chemin_images)

    r = reward(x).cpu()

    chemin_json.write_text(json.dumps({
        "tag": args.tag,
        "seed": args.seed,
        "n": len(x),
        "timesteps": timesteps,
        "model_calls": len(x) * timesteps,
        "device": device,
        "weights": args.weights,
        "config": {k: v for k, v in config.items() if isinstance(v, (int, float, str))},
        "reward": r.tolist(),
        "images": str(chemin_images.relative_to(RACINE)),
    }, indent=2))

    print(f"{chemin_json}  mean={r.mean():.4f} std={r.std():.4f} "
          f"min={r.min():.4f} max={r.max():.4f}")


if __name__ == "__main__":
    main()
