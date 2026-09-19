"""Les deux lignes de référence de la table 1 (FK Steering, SD v1.5) : k=1 et best-of-N.

Un enregistrement plat par (prompt, échantillonneur, seed) ; relancer le script
reprend là où il s'est arrêté. L'échantillonneur est la seule pièce à remplacer :
une fonction qui rend N images PIL, plus une ligne dans SAMPLERS.

Depuis la racine :
`HF_HOME=/home/onyxia/work/hf_cache python scripts/run_sd_baseline.py`
"""
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def load_prompts(path, limit):
    if not path.exists():
        raise SystemExit(f"{path} manquant : le sous-ensemble de prompts n'est pas encore produit.")
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        listes = [v for v in data.values() if isinstance(v, list)]
        if not listes:
            raise SystemExit(f"{path} : aucune liste de prompts dans ce JSON.")
        data = listes[0]
    prompts = [(str(e.get("id", i)), e["prompt"]) for i, e in enumerate(data)]
    return prompts[:limit] if limit else prompts


def sample_independent(pipe, prompt, n, generator, args):
    return pipe([prompt] * n, num_inference_steps=args.steps, guidance_scale=args.guidance,
                eta=args.eta, height=args.size, width=args.size, generator=generator).images


# Un échantillonneur = (fonction, N) ; brancher FK plus tard tient en une ligne ici,
# la fonction devant rendre ses N particules finales comme images PIL.
SAMPLERS = {"k1": (sample_independent, 1), "bon4": (sample_independent, 4)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", default=str(ROOT / "data" / "prompts_subset_40.json"))
    p.add_argument("--samplers", nargs="+", default=list(SAMPLERS), choices=list(SAMPLERS))
    p.add_argument("--seeds", type=int, nargs="+", default=[2024])
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--slicing", action="store_true",
                   help="pic VRAM mesuré à 2,16 Gio sur 15 Go : inutile par défaut")
    p.add_argument("--ir-cache", default="/home/onyxia/work/ir_cache")
    p.add_argument("--limit", type=int, default=None, help="les n premiers prompts, pour un essai")
    p.add_argument("--out", default=str(ROOT / "results" / "sd_baseline.json"))
    args = p.parse_args()

    prompts = load_prompts(Path(args.prompts), args.limit)

    import torch
    import hpsv2
    import ImageReward as RM
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(out.read_text())["runs"] if out.exists() else []
    faits = {(r["prompt_id"], r["sampler"], r["seed"]) for r in records}

    pipe = StableDiffusionPipeline.from_pretrained(
        REPO, torch_dtype=torch.float16, variant="fp16", safety_checker=None,
    ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    if args.slicing:
        pipe.enable_attention_slicing()

    # Budget compté, pas déduit. Les N particules passent dans UN seul forward batché :
    # le rapport de 4 entre bon4 et k1 est porté par les lignes, pas par les appels.
    compteur = {"calls": 0, "rows": 0, "batch": 0}

    def count(mod, inp, out_):
        compteur["calls"] += 1
        compteur["rows"] += inp[0].shape[0]
        compteur["batch"] = inp[0].shape[0]  # 2N : le CFG double le batch vu par l'UNet

    pipe.unet.register_forward_hook(count)

    ir = RM.load("ImageReward-v1.0", device="cuda", download_root=args.ir_cache)

    commun = {"model": REPO, "dtype": "fp16", "steps": args.steps, "guidance": args.guidance,
              "eta": args.eta, "size": args.size, "scheduler": "ddim"}
    total = len(prompts) * len(args.samplers) * len(args.seeds)
    fait = 0
    for pid, prompt in prompts:
        for name in args.samplers:
            fn, n = SAMPLERS[name]
            for seed in args.seeds:
                fait += 1
                if (pid, name, seed) in faits:
                    continue
                compteur.update(calls=0, rows=0)
                g = torch.Generator("cuda").manual_seed(seed)
                t0 = time.time()
                images = fn(pipe, prompt, n, g, args)
                dt = time.time() - t0

                ir_scores = ir.score(prompt, images)  # prompt d'abord
                if not isinstance(ir_scores, list):  # à N=1 ImageReward squeeze et rend un float nu
                    ir_scores = [ir_scores]
                # hpsv2.score recharge 1,97 Go de checkpoint à chaque appel : un seul appel pour les N images
                hps = [float(s) for s in hpsv2.score(images, prompt, hps_version="v2.1")]
                best = max(range(len(images)), key=lambda i: ir_scores[i])

                records.append({
                    "prompt_id": pid, "prompt": prompt, "sampler": name, "n": n, "seed": seed,
                    "ir": ir_scores, "hps": hps,
                    # la table reporte la particule choisie par IR : son HPS, et non le max de HPS
                    "ir_max": ir_scores[best], "hps_at_ir_max": hps[best],
                    "n_unet_calls": compteur["calls"], "n_unet_rows": compteur["rows"],
                    "unet_batch": compteur["batch"],
                    "seconds": round(dt, 1), **commun,
                })
                out.write_text(json.dumps({"runs": records}, indent=2))
                # SD, ImageReward et le ViT-H de hpsv2 tiennent la VRAM ensemble, et
                # hpsv2.score recharge son checkpoint a chaque appel : sans ca la
                # fragmentation finit par declencher un OOM en pleine nuit.
                torch.cuda.empty_cache()
                print(f"[{fait:3d}/{total}] {pid:>4} {name:4s} seed={seed} "
                      f"ir_max={ir_scores[best]:+.4f} hps={hps[best]:.4f} "
                      f"unet={compteur['calls']}x{compteur['batch']}={compteur['rows']} "
                      f"{dt:5.1f}s", flush=True)

    print(out)


if __name__ == "__main__":
    main()
