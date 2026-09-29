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
import argparse, json, os, socket, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"

DEFAUT = dict(lam=10.0, plancher=False, threshold=1.0, adapt=False, lam_max=None,
              schedule_t=None,      # valeurs de t ; None = --schedule
              schedule_idx=None,    # indices de boucle directs, prioritaires sur schedule_t
              resampler="systematic", form="increment", reward_vae="ftmse",
              steps=None, potential=None)   # None = --steps, --potential
ARMS = {  # ce qui differe du defaut, et rien d'autre
    "ctl":    dict(),
    "floor":  dict(plancher=True),
    "lam2":   dict(lam=2.0),
    "floor2": dict(lam=2.0, plancher=True),
    "lam0":   dict(lam=0.0),
    "adapt":  dict(adapt=True),
    "fadapt": dict(plancher=True, adapt=True),
    # constat 14, "commencer le calendrier plus tard" : le pas t = 80 retire, rien d'autre.
    "late":   dict(schedule_t=[0, 20, 40, 60]),
    # docs/reference_config.md : les quatre choix d'implementation du code publie, un par un
    # puis ensemble (R1), sous la configuration du papier. threshold 1.5 = reechantillonner a
    # chaque pas planifie, poids uniformes compris, comme leur boucle non adaptative.
    "stat0":  dict(plancher=True, form="statistic"),
    "multi":  dict(resampler="multinomial", threshold=1.5),
    "vae":    dict(reward_vae="pipe"),
    "idx":    dict(schedule_idx=[20, 40, 60, 80, 99]),
    "R1":     dict(plancher=True, form="statistic", resampler="multinomial", threshold=1.5,
                   reward_vae="pipe", schedule_idx=[20, 40, 60, 80, 99]),
    # plan du 22/09, B2 et B3
    "thr05":  dict(plancher=True, threshold=0.5),
    "rise":   dict(plancher=True, adapt=True, lam_max=100.0),
    # session H, l'annexe D du papier : 200 pas, calendrier [180, 160, 140, 120, 0] lu en t
    # (ce que fait leur lanceur quand seul --num_inference_steps change). s = t / steps :
    # ctl reechantillonne a s = 0.8..0.2, d200 et pos100 a s = 0.9..0.6.
    "free200": dict(lam=0.0, steps=200),
    "d200":    dict(steps=200, schedule_t=[0, 120, 140, 160, 180]),
    "st200":   dict(steps=200, schedule_t=[0, 40, 80, 120, 160]),
    "pos100":  dict(schedule_t=[0, 60, 70, 80, 90]),
    # la recette de la figure 7 (lambda 2, difference, ESS < k/2), k = 4, sans plancher
    "recipeD": dict(lam=2.0, threshold=0.5, potential="difference", steps=200,
                    schedule_t=[0, 120, 140, 160, 180]),
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
    p.add_argument("--prompt-ids", nargs="*", default=None,
                   help="restreint aux prompt_id donnes (l'indice i, donc x_T, reste celui du fichier)")
    p.add_argument("--save-images", action="store_true",
                   help="PNG 512 px dans out/images/<arm>/<prompt_id>_<slot>.png + index.json, and the "
                        "guide's decoded Tweedie estimates at each scheduled step, WebP 128 px, "
                        "<prompt_id>_t<t>_<slot>.webp")
    p.add_argument("--redo", action="store_true",
                   help="rejoue les paires (prompt, bras) deja presentes et remplace leur record ; "
                        "exige --force, car un autre processus ne rend pas les memes chiffres "
                        "(constat 19) et l'incident du 22/09 a ecrase six records ainsi")
    p.add_argument("--force", action="store_true",
                   help="autorise --redo a remplacer des records ; sinon, passer un --out neuf par session")
    args = p.parse_args()
    if args.redo and not args.force:
        raise SystemExit("--redo remplace des records d'une autre session : passer --force, "
                         "ou ecrire dans un autre fichier avec --out")

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

    # Finding 19: the resampling path reproduces within a machine group, not across. The GPU is
    # whatever Onyxia allocates, so every record says which process and device produced it.
    session = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "host": socket.gethostname(), "pid": os.getpid(),
               "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
               "cudnn_benchmark": torch.backends.cudnn.benchmark,
               "deterministic": torch.are_deterministic_algorithms_enabled()}
    session_id = f"{session['started']}/{session['host']}/{session['pid']}"
    print(f"session {session_id} on {session['device']}", flush=True)

    import ImageReward as RM
    from PIL import Image
    from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline
    from smc.fk import fk_steer
    from smc.models import StableDiffusion
    from smc.resampling import resample_multinomial, resample_systematic
    from smc.rewards_sd import ImageRewardSD

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(out.read_text())["runs"] if out.exists() else []
    faits = {(r["prompt_id"], r["arm"]) for r in records}
    if args.prompt_ids:
        prompts = [(pid, txt) for pid, txt in prompts if pid in set(args.prompt_ids)]
        # l'indice i qui fixe seed_effective est celui du fichier, pas celui de la sous-liste
        indices = {pid: i for i, (pid, _) in enumerate(
            [(str(e.get("id", i)), e["prompt"]) for i, e in enumerate(data)])}
    else:
        indices = {pid: i for i, (pid, _) in enumerate(prompts)}
    img_dir = out.parent / "images"
    index_path = img_dir / "index.json"
    index = json.loads(index_path.read_text()) if (args.save_images and index_path.exists()) else []

    pipe = StableDiffusionPipeline.from_pretrained(
        REPO, torch_dtype=torch.float16, variant="fp16", safety_checker=None).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    ir = RM.load("ImageReward-v1.0", device="cuda", download_root=args.ir_cache)
    vae_r = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse",
                                          torch_dtype=torch.float16).to("cuda")

    total, fait = len(prompts) * len(args.arms), 0
    for pid, prompt in prompts:
        i = indices[pid]
        for arm in args.arms:
            fait += 1
            if (pid, arm) in faits and not args.redo:
                continue
            if args.redo:
                records = [r for r in records if not (r["prompt_id"] == pid and r["arm"] == arm)]
                index = [e for e in index if not (e["prompt_id"] == pid and e["arm"] == arm)]
            a = dict(DEFAUT, **ARMS[arm])
            lam, plancher, seuil, adapt = a["lam"], a["plancher"], a["threshold"], a["adapt"]
            steps, potential = a["steps"] or args.steps, a["potential"] or args.potential
            if a["schedule_idx"] is not None:
                sched = sorted(a["schedule_idx"])
                cal = [steps - 1 - m for m in sched]
            else:
                cal = list(args.schedule) if a["schedule_t"] is None else list(a["schedule_t"])
                sched = sorted(steps - 1 - t for t in cal)
            effective = args.seed * 1000 + i
            g = torch.Generator("cuda").manual_seed(effective)

            model = StableDiffusion(pipe, guidance_scale=args.guidance,
                                    num_steps=steps, eta=args.eta)
            model.set_prompt(prompt)
            # la reward guide decode avec ft-mse (decision 6) ou avec le VAE du pipeline (les auteurs)
            brut = ImageRewardSD(pipe.vae if a["reward_vae"] == "pipe" else vae_r, ir, prompt)
            # fk_steer calls the reward once per scheduled step, on the Tweedie estimates before
            # that step's resampling: keep them, decode them once the run is over.
            seen = []

            def guide(z):
                if args.save_images:
                    seen.append(z.detach().clone())
                return brut(z)
            # reward_min_value des auteurs : le plancher passe par la reward, pas par smc/.
            reward = (lambda z: torch.clamp(guide(z), min=0.0)) if plancher else guide
            resampler = resample_multinomial if a["resampler"] == "multinomial" else resample_systematic

            t0 = time.time()
            x, info = fk_steer(model, reward, args.k, lam, potential, g,
                               resample_threshold=seuil, resampler=resampler, schedule=sched,
                               potential_form=a["form"],
                               adaptive_lam=adapt,
                               ess_target=(0.5 * args.k if adapt else None),
                               lam_max=((a["lam_max"] or lam) if adapt else None))
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
            if args.save_images:
                (img_dir / arm).mkdir(parents=True, exist_ok=True)
                roots = racines(anc, sched[-1]).tolist()
                for j, im in enumerate(images):
                    path = img_dir / arm / f"{pid}_{j}.png"
                    im.save(path)
                    index.append({"arm": arm, "prompt_id": pid, "slot": j, "root": roots[j],
                                  "ir": round(float(scores[j]), 4), "path": str(path.relative_to(out.parent))})
                index_path.write_text(json.dumps(index, indent=1))
                # calls come in loop order, so t decreases: 80, 60, 40, 20, 0
                for t, z in zip(sorted(cal, reverse=True), seen, strict=True):
                    arr = (brut.decode(z).permute(0, 2, 3, 1).float().cpu().numpy() * 255).round()
                    for j, a8 in enumerate(arr.astype(np.uint8)):
                        Image.fromarray(a8).resize((128, 128), Image.LANCZOS).save(
                            img_dir / arm / f"{pid}_t{t}_{j}.webp", quality=85)
            records.append({
                "prompt_id": pid, "prompt": prompt, "arm": arm, "lam": lam,
                "plancher": plancher, "adaptatif": adapt, "lam_max": a["lam_max"],
                "resampler": a["resampler"], "form": a["form"], "reward_vae": a["reward_vae"],
                "threshold": seuil, "seed": args.seed,
                "seed_effective": effective, "k": args.k, "potential": potential,
                "steps": steps, "schedule_t": cal, "schedule_idx": sched,
                "ir": [round(float(s), 4) for s in scores],
                "ir_max": round(float(max(scores)), 4),
                "div_pix": diversite(images),
                "lineages_trace": trace, "n_lineages": trace[-1],
                "root_slots": racines(anc, sched[-1]).tolist(),
                # la matrice des ancetres aux pas planifies (les autres lignes sont l'identite) :
                # ce que F1 dessine, et ce que root_slots resume.
                "ancestors_at_schedule": [anc[m].tolist() for m in sched],
                "n_resamplings": info["n_resamplings"],
                "ess_at_schedule": [round(info["ess_trace"][m].item(), 4) for m in sched],
                "lam_at_schedule": [round(info["lam_trace"][m].item(), 3) for m in sched],
                "r_at_schedule": [[round(v, 4) for v in row]
                                  for row in info["r_at_schedule"].tolist()],
                "logG_at_schedule": [[round(v, 4) for v in info["logG"][m].tolist()]
                                     for m in sched],
                "seconds": round(dt, 1),
                "session_id": session_id, "session": session,
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
