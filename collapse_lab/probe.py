"""Sonde causale : quand les lignees meurent-elles, et qu'est-ce qui les tue ?

Meme pipeline, memes prompts, memes seeds que scripts/run_sd_baseline.py --samplers fk4,
donc appariable avec results/sd_baseline.json et results/sd_variants/fk4_*.json.
Deux choses en plus : la matrice des ancetres est conservee, ce qu'aucun run sur
disque ne fait, et le bras `floor` reproduit le `reward_min_value = 0.0` du code
publie des auteurs (fkd_class.py : population_rs est initialise a ce plancher, et
le potentiel max lit max(r_t, population_rs)). Ce plancher s'obtient sans toucher
a smc/ : il suffit que la reward passee a fk_steer rende max(r, 0).

  BRAS      lambda  plancher  seuil   ce que le bras isole
  ctl        10       non      1.0    le fk4 de reference, controle du pilote
  floor      10       oui      1.0    le seul plancher des auteurs
  lam2        2       non      1.0    la seule force du tilt
  floor2      2       oui      1.0    les deux
  lam0        0       non      1.0    controle d'appariement : doit redonner bon4
  adapt      10*      non      1.0    lambda bisecte a ESS = k/2 aux pas non terminaux
  fadapt     10*      oui      1.0    le plancher et le lambda adaptatif ensemble

  * 10 est a la fois le plafond (lam_max) et le lambda du pas terminal. bisect_lambda
    rend lam_max des que l'ESS y est deja au-dessus de la cible, donc lambda_t ne peut
    que DESCENDRE sous 10 : ces deux bras testent "moins fort tot", pas le profil
    montant du constat 8. lam_max = 100 (le defaut 10*lam de fk_steer) testerait aussi
    "plus fort tard" ; c'est un autre bras, pas un autre reglage de celui-ci.
    La cible exp(lambda r(x_0)) ne bouge pas : le pas terminal garde lambda et acc
    porte la correction.

Depuis la racine :
  /home/onyxia/work/.venvs/sd/bin/python collapse_lab/probe.py --arms ctl floor --limit 20
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"

ARMS = {  # nom: (lam, plancher, seuil, lambda adaptatif)
    "ctl":    (10.0, False, 1.0, False),
    "floor":  (10.0, True,  1.0, False),
    "lam2":   (2.0,  False, 1.0, False),
    "floor2": (2.0,  True,  1.0, False),
    "lam0":   (0.0,  False, 1.0, False),
    "adapt":  (10.0, False, 1.0, True),
    "fadapt": (10.0, True,  1.0, True),
}


def racines(anc, jusqu_a):
    """Racine x_T de chaque case apres avoir applique les reechantillonnages 0..jusqu_a."""
    s = torch.arange(anc.shape[1], device=anc.device)
    for i in range(jusqu_a, -1, -1):
        s = anc[i][s]
    return s


def diversite(images, taille=64):
    if len(images) < 2:
        return None
    a = [np.asarray(im.resize((taille, taille)), dtype=np.float32) / 255.0 for im in images]
    d = [float(np.sqrt(((x - y) ** 2).mean())) for i, x in enumerate(a) for y in a[i + 1:]]
    return round(sum(d) / len(d), 4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", default=["ctl", "floor"], choices=list(ARMS))
    # C'est CE fichier, dans CET ordre, qui donne seed_effective = 2024000 + i et donc
    # les memes x_T que sd_baseline.json. prompts_subset_40.json a un autre ordre : la
    # sonde ne serait alors appariee a rien. Verifie ci-dessous plutot que suppose.
    p.add_argument("--prompts", default=str(ROOT / "data" / "imagereward-benchmark-prompts.json"))
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--schedule", type=int, nargs="+", default=[0, 20, 40, 60, 80])
    p.add_argument("--potential", default="max")
    p.add_argument("--ir-cache", default="/home/onyxia/work/ir_cache")
    p.add_argument("--out", default=str(ROOT / "collapse_lab" / "out" / "probe.json"))
    args = p.parse_args()

    data = json.loads(Path(args.prompts).read_text())
    if isinstance(data, dict):
        data = next(v for v in data.values() if isinstance(v, list))
    prompts = [(str(e.get("id", i)), e["prompt"]) for i, e in enumerate(data)][:args.limit]

    # Garde-fou d'appariement : la case j de ce run et la case j de bon4 ne partagent x_T
    # que si le generateur recoit le meme seed_effective. Une inversion d'ordre du fichier
    # de prompts invaliderait tout le depouillement sans rien casser visiblement.
    ref = json.loads((ROOT / "results" / "sd_baseline.json").read_text())["runs"]
    ref = {r["prompt_id"]: r for r in ref if r["sampler"] == "bon4" and r["seed"] == args.seed}
    for i, (pid, txt) in enumerate(prompts):
        if pid not in ref:
            raise SystemExit(f"{pid} absent de bon4 : rien a quoi apparier")
        if ref[pid]["seed_effective"] != args.seed * 1000 + i or ref[pid]["prompt"] != txt:
            raise SystemExit(
                f"appariement rompu a i={i} ({pid}) : bon4 porte "
                f"seed_effective={ref[pid]['seed_effective']}, ce run utiliserait "
                f"{args.seed * 1000 + i}. Mauvais fichier de prompts ou mauvais ordre.")
    print(f"appariement verifie sur {len(prompts)} prompts", flush=True)

    import ImageReward as RM
    from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline
    from smc.fk import fk_steer
    from smc.models import StableDiffusion
    from smc.rewards_sd import ImageRewardSD

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(out.read_text())["runs"] if out.exists() else []
    faits = {(r["prompt_id"], r["arm"]) for r in records}

    pipe = StableDiffusionPipeline.from_pretrained(
        REPO, torch_dtype=torch.float16, variant="fp16", safety_checker=None).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    ir = RM.load("ImageReward-v1.0", device="cuda", download_root=args.ir_cache)
    vae_r = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse",
                                          torch_dtype=torch.float16).to("cuda")

    sched = sorted(args.steps - 1 - t for t in args.schedule)
    total, fait = len(prompts) * len(args.arms), 0
    for i, (pid, prompt) in enumerate(prompts):
        for arm in args.arms:
            fait += 1
            if (pid, arm) in faits:
                continue
            lam, plancher, seuil, adapt = ARMS[arm]
            effective = args.seed * 1000 + i
            g = torch.Generator("cuda").manual_seed(effective)

            model = StableDiffusion(pipe, guidance_scale=args.guidance,
                                    num_steps=args.steps, eta=args.eta)
            model.set_prompt(prompt)
            brut = ImageRewardSD(vae_r, ir, prompt)
            # reward_min_value des auteurs : le plancher passe par la reward, pas par smc/.
            reward = (lambda z: torch.clamp(brut(z), min=0.0)) if plancher else brut

            t0 = time.time()
            x, info = fk_steer(model, reward, args.k, lam, args.potential, g,
                               resample_threshold=seuil, schedule=sched,
                               adaptive_lam=adapt,
                               ess_target=(0.5 * args.k if adapt else None),
                               lam_max=(lam if adapt else None))
            dt = time.time() - t0

            with torch.no_grad():
                img = pipe.vae.decode(x / pipe.vae.config.scaling_factor).sample
            images = pipe.image_processor.postprocess(img, output_type="pil")
            scores = ir.score(prompt, images)
            scores = scores if isinstance(scores, list) else [scores]

            anc = info["ancestors"]
            # nombre de racines distinctes apres chaque pas planifie : la trace que
            # n_lineages resume en un seul chiffre a la fin.
            trace = [int(torch.unique(racines(anc, m)).numel()) for m in sched]
            records.append({
                "prompt_id": pid, "prompt": prompt, "arm": arm, "lam": lam,
                "plancher": plancher, "adaptatif": adapt,
                "threshold": seuil, "seed": args.seed,
                "seed_effective": effective, "k": args.k, "potential": args.potential,
                "schedule_t": list(args.schedule), "schedule_idx": sched,
                "ir": [round(float(s), 4) for s in scores],
                "ir_max": round(float(max(scores)), 4),
                "div_pix": diversite(images),
                "lineages_trace": trace, "n_lineages": trace[-1],
                "root_slots": racines(anc, sched[-1]).tolist(),
                "n_resamplings": info["n_resamplings"],
                "ess_at_schedule": [round(info["ess_trace"][m].item(), 4) for m in sched],
                "lam_at_schedule": [round(info["lam_trace"][m].item(), 3) for m in sched],
                "r_at_schedule": [[round(v, 4) for v in row]
                                  for row in info["r_at_schedule"].tolist()],
                "logG_at_schedule": [[round(v, 4) for v in info["logG"][m].tolist()]
                                     for m in sched],
                "seconds": round(dt, 1),
            })
            tmp = out.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"runs": records}, indent=2))
            tmp.replace(out)
            torch.cuda.empty_cache()
            print(f"[{fait:3d}/{total}] {pid:>6} {arm:6s} ir_max={max(scores):+.4f} "
                  f"lignees={trace} ESS={[round(info['ess_trace'][m].item(),2) for m in sched]} "
                  f"resampl={info['n_resamplings']} {dt:5.1f}s", flush=True)
    print(out)


if __name__ == "__main__":
    main()
