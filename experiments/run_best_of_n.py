"""Sweep best-of-n (exercice 1.3) : les données de la figure 1.

Générateur dérivé de (seed, n) et consigné dans l'enregistrement : une cellule
seule se rejoue à l'identique, même si la formule change ensuite.
Depuis la racine : `python -m experiments.run_best_of_n`.
"""
import argparse
import json
from pathlib import Path

import torch

from experiments.run_free_samples import charger_modele
from smc.fk import best_of_n
from smc.rewards import reward
from smc.rng import make_generator

RACINE = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026])
    p.add_argument("--weights", default="/home/onyxia/work/ddpm/weights/ddpm_last.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=str(RACINE / "results" / "bestofn.json"))
    p.add_argument("--save-images", action="store_true",
                   help="garde le meilleur échantillon de chaque cellule dans samples/")
    args = p.parse_args()

    model, config = charger_modele(args.weights, args.device)

    runs, images = [], {}
    total = sum(args.n) * len(args.seeds) * model.T
    faits = 0
    for seed in args.seeds:
        for n in args.n:
            seed_effectif = seed * 1000 + n
            gen = make_generator(seed_effectif, device=args.device)
            x_best, r_best, appels = best_of_n(model, reward, n, gen)
            runs.append({"methode": "best_of_n", "n": n, "seed": seed,
                         "seed_effectif": seed_effectif, "r_best": r_best,
                         "n_model_calls": appels})
            if args.save_images:
                images[f"seed{seed}_n{n}"] = x_best
            faits += appels
            print(f"seed={seed} n={n:2d}  r_best={r_best:+.4f}  "
                  f"[{faits}/{total} appels]", flush=True)

    chemin_images = None
    if args.save_images:
        (RACINE / "samples").mkdir(exist_ok=True)
        chemin_images = RACINE / "samples" / "bestofn_best.pt"
        torch.save(images, chemin_images)

    sortie = Path(args.out)
    sortie.parent.mkdir(exist_ok=True)
    sortie.write_text(json.dumps({
        "runs": runs,
        "seeds": args.seeds,
        "n": args.n,
        "T": model.T,
        "device": args.device,
        "weights": args.weights,
        "images": str(chemin_images.relative_to(RACINE)) if chemin_images else None,
        "config": {k: v for k, v in config.items() if isinstance(v, (int, float, str))},
    }, indent=2))
    print(sortie)


if __name__ == "__main__":
    main()
