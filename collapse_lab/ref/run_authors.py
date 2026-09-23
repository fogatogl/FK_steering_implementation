"""R0. Le code des auteurs, tel quel, sur les 100 prompts du depot.

Pipeline `FKDStableDiffusion` et classe `FKD` de zacharyhorvitz/Fk-Diffusion-Steering
(commit 9413005), lus depuis /home/onyxia/work/fkd_ref/, jamais copies ici (pas de licence).
Deux configurations :
  paper    : lambda 10, k 4, potentiel max, 20-80-20 (les flags de launch.sh, la table 1)
  defaults : lambda 10, k 4, potentiel diff, 5-30-5   (les defauts de launch_eval_runs.py)
Le modele est SD v1.5 fp16 (`stable-diffusion-v1-5/stable-diffusion-v1-5`, le miroir de
`runwayml/stable-diffusion-v1-5` retire du Hub), DDIM eta 1, 100 pas, CFG 7.5 par defaut du
pipeline, comme chez eux. ImageReward-v1.0 charge par leur `rm_load` et score par leur
`score_batched`, depuis le cache local. Le grader LLM qu'ils importent est remplace par un
module factice (`shim/google/genai.py`).

Deux ecarts a leur lanceur, dits ici : (1) la graine est refixee par prompt
(`seed * 1000 + i`) au lieu d'une fois par passe, pour que la file reprenne apres une coupure ;
leurs graines 42/43/44 posees une fois ne rendent de toute facon pas nos x_T ; (2) une seule
passe (une graine) au lieu de trois, et ImageReward seul (HPS sur --hps).
Sortie : results/sd_authors_R0.json, appariable a bon4 / ctl par prompt_id.

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
                   help="leur pipeline sans FK (fkd_args=None) : quatre tirages libres, a comparer a bon4")
    p.add_argument("--generator", action="store_true",
                   help="passe torch.Generator(seed_effective) au pipeline : memes x_T et meme bruit DDIM que "
                        "bon4 / ctl / R1 (les tirages multinomiaux de leur FKD restent sur le RNG global)")
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
    # fkd_pipeline_sd fait `from rewards import ...` (fkd_diffusers/ est sur sys.path) : le meme
    # fichier existe deux fois dans sys.modules. Le meme modele charge pour les deux.
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
        # le pas terminal reechantillonne chez eux si ESS < k/2 : les quatre images peuvent etre des copies
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
