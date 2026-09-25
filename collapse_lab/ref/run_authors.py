"""R0. The authors' code, as released, on the repository's 100 prompts.

Pipeline `FKDStableDiffusion` and class `FKD` of zacharyhorvitz/Fk-Diffusion-Steering
(commit 9413005), read from /home/onyxia/work/fkd_ref/, never copied here (no licence).
Two configurations:
  paper    : lambda 10, k 4, max potential, 20-80-20 (the flags of launch.sh, table 1)
  defaults : lambda 10, k 4, diff potential, 5-30-5  (the defaults of launch_eval_runs.py)
The model is SD v1.5 fp16 (`stable-diffusion-v1-5/stable-diffusion-v1-5`, the mirror of
`runwayml/stable-diffusion-v1-5`, which left the Hub), DDIM eta 1, 100 steps, CFG 7.5 by the
pipeline's default, as in their code. ImageReward-v1.0 loaded by their `rm_load` and scored by
their `score_batched`, from the local cache. The LLM grader they import is replaced by a stub
module (`shim/google/genai.py`).

Two departures from their launcher, stated here: (1) the seed is reset per prompt
(`seed * 1000 + i`) instead of once per pass, so that the queue resumes after an interruption;
their seeds 42/43/44 set once would not give our x_T anyway; (2) one pass (one seed) instead
of three, and ImageReward alone (HPS with --hps).
Output: results/sd_authors_R0.json, pairable with bon4 / ctl by prompt_id.

  /home/onyxia/work/.venvs/sd/bin/python collapse_lab/ref/run_authors.py --config paper --limit 100
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
FKD = Path("/home/onyxia/work/fkd_ref/Fk-Diffusion-Steering/text_to_image")
REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONFIGS = {
    "paper":    dict(potential_type="max",  resample_frequency=20, resampling_t_start=20, resampling_t_end=80),
    "defaults": dict(potential_type="diff", resample_frequency=5,  resampling_t_start=5,  resampling_t_end=30),
}


def diversite(images, taille=64):
    a = [np.asarray(im.resize((taille, taille)), dtype=np.float32) / 255.0 for im in images]
    d = [float(np.sqrt(((x - y) ** 2).mean())) for i, x in enumerate(a) for y in a[i + 1:]]
    return round(sum(d) / len(d), 4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="paper", choices=list(CONFIGS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--hps", action="store_true")
    p.add_argument("--ir-cache", default="/home/onyxia/work/ir_cache")
    p.add_argument("--out", default="/home/onyxia/work/diffusion-models/results/sd_authors_R0.json")
    p.add_argument("--no-smc", action="store_true",
                   help="their pipeline without FK (fkd_args=None): four free draws, to compare with bon4")
    p.add_argument("--generator", action="store_true",
                   help="passes torch.Generator(seed_effective) to the pipeline; with --seed 2024, the same x_T and the "
                        "same DDIM noise as bon4 / ctl / R1 (the multinomial draws of their FKD stay on the global RNG)")
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).parent / "shim"))
    sys.path.insert(0, str(FKD))
    sys.path.insert(0, str(FKD / "fkd_diffusers"))
    import diffusers
    from diffusers import DDIMScheduler
    from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion
    import fkd_diffusers.rewards as rewards
    from fkd_diffusers.image_reward_utils import rm_load
    from fks_utils import do_eval

    prompts = json.loads((FKD / "prompt_files" / "benchmark_ir.json").read_text())[:args.limit]
    ours = json.loads((ROOT / "data" / "imagereward-benchmark-prompts.json").read_text())
    ours = ours if isinstance(ours, list) else next(v for v in ours.values() if isinstance(v, list))
    assert [e["id"] for e in prompts] == [e["id"] for e in ours[:args.limit]], "fichiers de prompts differents"

    out = Path(args.out)
    records = json.loads(out.read_text())["runs"] if out.exists() else []
    sampler = ("authors_free" if args.no_smc else f"authors_{args.config}") + ("_g" if args.generator else "")
    faits = {(r["prompt_id"], r["sampler"], r["seed"]) for r in records}

    pipe = FKDStableDiffusion.from_pretrained(REPO, torch_dtype=torch.float16, variant="fp16",
                                              safety_checker=None).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    # fkd_pipeline_sd does `from rewards import ...` (fkd_diffusers/ is on sys.path): the same
    # file exists twice in sys.modules. The same loaded model for both.
    import rewards as rewards_plat
    rewards.REWARDS_DICT["ImageReward"] = rewards_plat.REWARDS_DICT["ImageReward"] = rm_load(
        "ImageReward-v1.0", device="cuda", download_root=args.ir_cache)
    metrics = ["ImageReward"] + (["HumanPreference"] if args.hps else [])
    versions = {"diffusers": diffusers.__version__, "torch": torch.__version__, "fkd_commit": "9413005"}

    for i, item in enumerate(prompts):
        pid, prompt = item["id"], item["prompt"]
        if (pid, sampler, args.seed) in faits:
            continue
        effective = args.seed * 1000 + i
        torch.manual_seed(effective)
        torch.cuda.manual_seed_all(effective)
        fkd_args = None if args.no_smc else dict(
            lmbda=10.0, num_particles=4, use_smc=True, adaptive_resampling=False,
            time_steps=100, guidance_reward_fn="ImageReward", **CONFIGS[args.config])
        gen = torch.Generator("cuda").manual_seed(effective) if args.generator else None
        t0 = time.time()
        images = pipe([prompt] * 4, num_inference_steps=100, eta=1.0, fkd_args=fkd_args, generator=gen)[0]
        dt = time.time() - t0
        res = do_eval(prompt=[prompt] * 4, images=images, metrics_to_compute=metrics)
        ir = [round(float(v), 4) for v in res["ImageReward"]["result"]]
        # their terminal step resamples if ESS < k/2: the four images can be copies
        arrs = [np.asarray(im.resize((32, 32))) for im in images]
        n_distinct = len({a.tobytes() for a in arrs})
        rec = {
            "prompt_id": pid, "prompt": prompt, "sampler": sampler, "config": CONFIGS[args.config],
            "seed": args.seed, "seed_effective": effective, "k": 4, "lam": 10.0,
            "ir": ir, "ir_max": max(ir), "ir_mean": round(float(np.mean(ir)), 4),
            "n_distinct_images": n_distinct, "div_pix": diversite(images),
            "seconds": round(dt, 1), "versions": versions,
        }
        if args.hps:
            rec["hps"] = [round(float(v), 4) for v in res["HumanPreference"]["result"]]
        records.append(rec)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"runs": records}, indent=1))
        tmp.replace(out)
        torch.cuda.empty_cache()
        print(f"[{i + 1:3d}/{len(prompts)}] {pid:>12} {sampler} ir_max={max(ir):+.4f} ir={ir} "
              f"distinct={n_distinct} {dt:5.1f}s", flush=True)
    print(out)


if __name__ == "__main__":
    main()
