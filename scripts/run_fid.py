"""Class-3 FID: references, free images (base / fine-tuned), FK images, FID.

Same protocol as the notebook (79.2 / 50.7 / floor 21.7): EMA weights, 2048
images, the 5000 training cats as reference, uint8 PNG. Each stage writes at a
fine grain (batch for `gen`, run for `fk`) and skips what exists: after a cut,
rerun the same command. Results accumulate in results/fid.json.
"""
import argparse
import json
import math
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.run_free_samples import load_model
from smc.fk import fk_steer
from smc.resampling import resample_multinomial, resample_systematic
from smc.rewards import make_classifier_reward, reward as red_reward
from smc.rng import make_generator

ROOT = Path(__file__).resolve().parent.parent
DDPM = Path("/home/onyxia/work/ddpm")
FID_DIR, DATA_DIR = DDPM / "fid", DDPM / "dataset"
RESULTS = ROOT / "results" / "fid.json"
RESAMPLERS = {"systematic": resample_systematic, "multinomial": resample_multinomial}

# The classifier reward goes here once it exists: one entry = one factory
# args -> callable(x) -> (k,).
REWARDS = {"rouge": lambda args: red_reward,
           "classifier": lambda args: make_classifier_reward(args.classifier_weights, args.target,
                                                             args.device)}


def save_pngs(x, folder, start):
    """x in [-1, 1] -> uint8 PNG, exactly like the notebook's dump_samples."""
    folder.mkdir(parents=True, exist_ok=True)
    arr = ((x * .5 + .5).clamp(0, 1) * 255).round().to(torch.uint8)
    arr = arr.permute(0, 2, 3, 1).cpu().numpy()
    for i in range(len(arr)):
        Image.fromarray(arr[i]).save(folder / f"{start + i:05d}.png")


def complete(folder, start, n):
    return all((folder / f"{start + i:05d}.png").exists() for i in range(n))


def refs(args):
    from torchvision.datasets import CIFAR10
    train = CIFAR10(str(DATA_DIR), train=True, download=True)
    test = CIFAR10(str(DATA_DIR), train=False, download=True)
    c = args.classe
    sets = {
        f"ref_class{c}": [train[i][0] for i, y in enumerate(train.targets) if y == c],
        f"ref_class{c}_test": [test[i][0] for i, y in enumerate(test.targets) if y == c],
        "ref_test": [test[i][0] for i in range(len(test))],
    }
    # Floor at the same N as the generated images: a subset of the reference,
    # hence optimistic (overlap). `ref_class{c}_test` is disjoint but N=1000.
    sets[f"ref_class{c}_sub{args.n}"] = sets[f"ref_class{c}"][:args.n]
    for name, images in sets.items():
        folder = FID_DIR / name
        if complete(folder, 0, len(images)):
            print(f"{name}: {len(images)} already there"); continue
        folder.mkdir(parents=True, exist_ok=True)
        for j, img in enumerate(images):
            img.save(folder / f"{j:05d}.png")
        print(f"{name}: {len(images)} written")


def gen(args):
    model, _ = load_model(args.weights, args.device, ema=not args.no_ema,
                          steps=args.steps, eta=args.eta)
    folder = FID_DIR / f"gen_{args.tag}{suffix(args)}"
    for b in range(math.ceil(args.n / args.batch)):
        start = b * args.batch
        size = min(args.batch, args.n - start)
        if complete(folder, start, size):
            continue
        t0 = time.time()
        g = make_generator(args.seed * 1000 + b, args.device)
        state = model.initial_state(size, g)
        for t in model.timesteps:
            state = model.step(state, t, g)
        save_pngs(state["x"], folder, start)
        print(f"{folder.name}: batch {b} ({start}-{start + size}) in {time.time() - t0:.0f}s",
              flush=True)


def suffix(args):
    return f"_ddim{args.steps}_eta{args.eta:g}" if args.steps else ""


def fk(args):
    model, _ = load_model(args.weights, args.device, ema=not args.no_ema,
                          steps=args.steps, eta=args.eta)
    r = REWARDS[args.reward](args)
    tag = f"fk_{args.reward}_{args.potential}_lam{args.lam:g}_k{args.k}{suffix(args)}"
    folder = FID_DIR / f"gen_{tag}"
    runs = folder / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    for run in range(math.ceil(args.n / args.k)):
        f = runs / f"run{run:04d}.pt"
        if f.exists():
            continue
        t0 = time.time()
        g = make_generator(args.seed * 100000 + run, args.device)
        x, info = fk_steer(model, r, args.k, args.lam, args.potential, g,
                           resampler=RESAMPLERS[args.resampler])
        # PNGs first: the .pt only exists once the images are complete.
        save_pngs(x, folder, run * args.k)
        torch.save({"x": x.cpu(), "weights": info["weights"].cpu(),
                    "rewards": info["rewards"].cpu(), "ess_min": info["ess_min"],
                    "n_resamplings": info["n_resamplings"], "lam": args.lam,
                    "potential": args.potential, "reward": args.reward,
                    "resampler": args.resampler, "k": args.k, "seed": args.seed,
                    "ema": not args.no_ema, "steps": model.T,
                    "eta": args.eta if args.steps else None}, f)
        print(f"{tag}: run {run} ess_min={info['ess_min']:.2f} "
              f"resampl={info['n_resamplings']} in {time.time() - t0:.0f}s", flush=True)


def statistics(folder, inception, device):
    """Inception (mu, sigma) of the folder, cached in fid/<name>.npz."""
    from pytorch_fid.fid_score import compute_statistics_of_path
    cache = folder.with_suffix(".npz")
    if cache.exists():
        d = np.load(cache)
        return d["mu"], d["sigma"]
    mu, sigma = compute_statistics_of_path(str(folder), inception, 64, 2048, device,
                                           num_workers=4)
    np.savez(cache, mu=mu, sigma=sigma)
    return mu, sigma


def frechet(mu1, s1, mu2, s2):
    """pytorch-fid 0.3.0 calls sqrtm(disp=False), removed in scipy 1.18."""
    from scipy import linalg
    diff = mu1 - mu2
    covmean = linalg.sqrtm(s1 @ s2)
    if not np.isfinite(covmean).all():
        offset = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + offset) @ (s2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def fid(args):
    from pytorch_fid.inception import InceptionV3
    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(args.device).eval()
    gen_dir, ref_dir = FID_DIR / args.gen, FID_DIR / args.ref
    n_gen, n_ref = len(list(gen_dir.glob("*.png"))), len(list(ref_dir.glob("*.png")))
    mu_g, s_g = statistics(gen_dir, inception, args.device)
    mu_r, s_r = statistics(ref_dir, inception, args.device)
    value = frechet(mu_g, s_g, mu_r, s_r)

    records = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    records = [e for e in records if not (e["gen"] == args.gen and e["ref"] == args.ref)]
    records.append({"gen": args.gen, "ref": args.ref, "n_gen": n_gen,
                    "n_ref": n_ref, "fid": value, "date": str(date.today())})
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(records, indent=2))
    print(f"FID {args.gen} ({n_gen}) vs {args.ref} ({n_ref}) = {value:.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["refs", "gen", "fk", "fid"])
    p.add_argument("--classe", type=int, default=3, help="target class")
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--weights", default=str(DDPM / "weights" / "ddpm_last.pt"))
    p.add_argument("--no-ema", action="store_true", help="raw weights, as in the lambda sweep")
    p.add_argument("--steps", type=int, default=None, help="DDIM with this many steps")
    p.add_argument("--eta", type=float, default=0.0)
    # gen
    p.add_argument("--tag", default="base")
    p.add_argument("--batch", type=int, default=256)
    # fk
    p.add_argument("--reward", default="rouge", choices=list(REWARDS))
    p.add_argument("--target", type=int, default=3)
    p.add_argument("--classifier-weights",
                   default=str(DDPM / "weights" / "classifier_small_seed0.pt"))
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--potential", default="difference")
    p.add_argument("--resampler", default="systematic", choices=list(RESAMPLERS))
    p.add_argument("--k", type=int, default=16)
    # fid
    p.add_argument("--gen", default="gen_base")
    p.add_argument("--ref", default="ref_class3")
    args = p.parse_args()
    {"refs": refs, "gen": gen, "fk": fk, "fid": fid}[args.stage](args)


if __name__ == "__main__":
    main()
