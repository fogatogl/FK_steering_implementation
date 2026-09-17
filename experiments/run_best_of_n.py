"""Best-of-n sweep: the data behind figure 1.

The generator is derived from (seed, n) and recorded with the run: a single
cell can be replayed exactly, even if the formula changes later.
From the root: `python -m experiments.run_best_of_n`.
"""
import argparse
import json
from pathlib import Path

import torch

from experiments.run_free_samples import load_model
from smc.fk import best_of_n
from smc.rewards import reward
from smc.rng import make_generator

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026])
    p.add_argument("--weights", default="/home/onyxia/work/ddpm/weights/ddpm_last.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=str(ROOT / "results" / "bestofn.json"))
    p.add_argument("--save-images", action="store_true",
                   help="keep the best sample of each cell in samples/")
    p.add_argument("--no-ema", action="store_true", help="raw weights, as before 17/09")
    args = p.parse_args()

    model, config = load_model(args.weights, args.device, ema=not args.no_ema)

    runs, images = [], {}
    total = sum(args.n) * len(args.seeds) * model.T
    done = 0
    for seed in args.seeds:
        for n in args.n:
            effective_seed = seed * 1000 + n
            gen = make_generator(effective_seed, device=args.device)
            x, info = best_of_n(model, reward, n, gen)
            best = info["rewards"].argmax()
            calls = n * model.T
            runs.append({"method": "best_of_n", "n": n, "seed": seed,
                         "effective_seed": effective_seed,
                         "r_best": info["rewards"][best].item(), "n_model_calls": calls})
            if args.save_images:
                images[f"seed{seed}_n{n}"] = x[best].cpu()
            done += calls
            print(f"seed={seed} n={n:2d}  r_best={runs[-1]['r_best']:+.4f}  "
                  f"[{done}/{total} calls]", flush=True)

    images_path = None
    if args.save_images:
        (ROOT / "samples").mkdir(exist_ok=True)
        images_path = ROOT / "samples" / "bestofn_best.pt"
        torch.save(images, images_path)

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "runs": runs,
        "seeds": args.seeds,
        "n": args.n,
        "T": model.T,
        "device": args.device,
        "weights": args.weights,
        "ema": not args.no_ema,
        "images": str(images_path.relative_to(ROOT)) if images_path else None,
        "config": {k: v for k, v in config.items() if isinstance(v, (int, float, str))},
    }, indent=2))
    print(out)


if __name__ == "__main__":
    main()
