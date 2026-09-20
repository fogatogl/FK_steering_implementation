"""Lambda sweep: the data behind figure 2.

One flat record per run (method, potential, lam, resampler, k, seed): two
result files concatenate, a diverged point reruns alone.

Each run restarts from a fresh generator seeded on (seed, k). Best-of-N and FK
at lambda=0 therefore see the same initial noise and consume the RNG in the
same order: their `r_max` coincide, up to GPU non-determinism (~1e-6 in fp32;
two identical calls are already not bit-for-bit equal).

From the root: `python scripts/run_sweep_lambda.py`.
"""
import argparse
import json
from pathlib import Path

import torch

from experiments.run_free_samples import load_model
from smc.fk import best_of_n, fk_steer
from smc.resampling import resample_multinomial, resample_systematic
from smc.rewards import make_classifier_reward, reward as red_reward
from smc.rng import make_generator

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = Path("/home/onyxia/work/ddpm/weights")
RESAMPLERS = {"systematic": resample_systematic, "multinomial": resample_multinomial}


def make_reward(args):
    if args.reward == "red":
        return red_reward
    return make_classifier_reward(args.classifier_weights, args.target, args.device)


def sample_from_weights(x, w, r, generator):
    idx = torch.multinomial(w, num_samples=1, generator=generator).item()
    return x[idx], r[idx].item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, nargs="+", default=[16])
    p.add_argument("--lam", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
    p.add_argument("--potentials", nargs="+", default=["difference", "max", "sum"])
    p.add_argument("--resamplers", nargs="+", default=list(RESAMPLERS),
                   choices=list(RESAMPLERS))
    p.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026])
    p.add_argument("--reward", default="red", choices=["red", "classifier"])
    p.add_argument("--target", type=int, default=3, help="target class of the classifier reward")
    p.add_argument("--classifier-weights", default=str(WEIGHTS / "classifier_small_seed0.pt"))
    p.add_argument("--weights", default="/home/onyxia/work/ddpm/weights/ddpm_last.pt",
                   help="a checkpoint, or hub:<repo> for a pretrained DDPM")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=str(ROOT / "results" / "sweep_lambda.json"))
    p.add_argument("--save-images", action="store_true",
                   help="keep the k final particles of each run in samples/")
    p.add_argument("--no-ema", action="store_true",
                   help="raw weights: those of results/sweep_lambda.json from 17/09")
    p.add_argument("--steps", type=int, default=None, help="DDIM with this many steps")
    p.add_argument("--eta", type=float, default=0.0)
    args = p.parse_args()

    model, config = load_model(args.weights, args.device, ema=not args.no_ema,
                               steps=args.steps, eta=args.eta)
    reward = make_reward(args)
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    images_path = ROOT / "samples" / f"{out.stem}.pt" if args.save_images else None

    # Ecrit apres chaque run : une nuit qui meurt a 90 % ne doit pas tout perdre.
    # Les images et les ancetres aussi, pas seulement le JSON.
    def dump():
        if args.save_images:
            (ROOT / "samples").mkdir(exist_ok=True)
            torch.save(images, images_path)
        out.write_text(json.dumps({
        "runs": runs,
        "k": args.k,
        "lam": args.lam,
        "potentials": args.potentials,
        "resamplers": args.resamplers,
        "seeds": args.seeds,
        "T": model.T,
        "sampler": "ddim" if args.steps else "ddpm",
        "eta": args.eta if args.steps else None,
        "device": args.device,
        "weights": args.weights,
        "ema": not args.no_ema,
        "reward": args.reward,
        "target": args.target if args.reward == "classifier" else None,
        "classifier_weights": args.classifier_weights if args.reward == "classifier" else None,
        "images": str(images_path.relative_to(ROOT)) if images_path else None,
        "config": {k: v for k, v in config.items() if isinstance(v, (int, float, str))},
    }, indent=2))

    runs, images = [], {}
    per_k = 1 + len(args.potentials) * len(args.lam) * len(args.resamplers)
    total = len(args.k) * len(args.seeds) * per_k
    for k in args.k:
        common = {"k": k, "n_model_calls": k * model.T, "reward": args.reward,
                  "target": args.target if args.reward == "classifier" else None,
                  "steps": model.T, "eta": args.eta if args.steps else None}
        for seed in args.seeds:
            effective_seed = seed * 1000 + k

            gen = make_generator(effective_seed, device=args.device)
            x, info = best_of_n(model, reward, k, gen)
            r = info["rewards"]
            _, r_sample = sample_from_weights(x, info["weights"], r, gen)
            runs.append({"method": "best_of_n", "potential": None, "lam": None,
                         "resampler": None, "seed": seed, "effective_seed": effective_seed,
                         "r_sample": r_sample, "r_max": r.max().item(),
                         "ess_min": None, "n_resamplings": None, "ess_trace": None, **common})
            if args.save_images:
                images[f"best_of_n_k{k}_seed{seed}"] = x.cpu()
            print(f"[{len(runs):3d}/{total}] k={k:<3d} seed={seed} best_of_n                    "
                  f"r_sample={r_sample:+.4f}  r_max={r.max().item():+.4f}", flush=True)
            dump()

            for name in args.resamplers:
                for pot in args.potentials:
                    for lam in args.lam:
                        gen = make_generator(effective_seed, device=args.device)
                        x, info = fk_steer(model, reward, k, lam, pot, gen,
                                           resampler=RESAMPLERS[name])
                        r = info["rewards"]
                        _, r_sample = sample_from_weights(x, info["weights"], r, gen)
                        tag = f"{pot}_{name}_k{k}_lam{lam:g}_seed{seed}"
                        runs.append({"method": "fk", "potential": pot, "lam": lam,
                                     "resampler": name, "seed": seed, "effective_seed": effective_seed,
                                     "r_sample": r_sample, "r_max": r.max().item(),
                                     "ess_min": info["ess_min"],
                                     "n_resamplings": info["n_resamplings"],
                                     # ESS(t) porte la figure 2 ; les ancetres pesent T*k
                                     # entiers et partent dans le .pt avec les images.
                                     "ess_trace": [round(v, 4) for v in info["ess_trace"].tolist()],
                                     "tag": tag, **common})
                        if args.save_images:
                            images[tag] = x.cpu()
                            images[f"ancestors_{tag}"] = info["ancestors"].cpu()
                        print(f"[{len(runs):3d}/{total}] k={k:<3d} seed={seed} {pot:10s} {name:11s} "
                              f"lam={lam:<5g} r_sample={r_sample:+.4f}  "
                              f"r_max={r.max().item():+.4f}  ess_min={info['ess_min']:5.2f}  "
                              f"resampl={info['n_resamplings']}", flush=True)
                        dump()

    dump()
    print(out)


if __name__ == "__main__":
    main()
