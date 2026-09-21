"""Figure 6 : pour un prompt et une seed, les images que k1, best-of-4 et FK ont produites.

Le run ne garde pas les images ; les seeds sont dans le JSON, donc on les refait
ici avec les mêmes fonctions que run_sd_baseline.py et le même x_T. Les scores
affichés sont recalculés sur les images régénérées, pas copiés du JSON, et les
deux sont imprimés côte à côte pour vérifier que la régénération est fidèle.

Depuis scripts/ :
`HF_HOME=/home/onyxia/work/hf_cache python plot_fig6_sd_grid.py --cases 010856-0009:2025 001369-0084:2024 011623-0101:2024`

Les lignes en plus de k1/bon4/fk4 sont les variantes de l'écran (VARIANTES) : leurs
flags écrasent ceux de la ligne de commande, et leur JSON de référence est
results/sd_variants/<tag>.json.
`... --rows k1 bon4 fk4 T1 T1A05 --cases 005695-0057:2024`
"""
import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.patheffects
import matplotlib.patches
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from figstyle import BLUE, INK, INK_LIGHT, RED
from run_sd_baseline import FK, REPO, SAMPLERS, diversite, load_prompts

ROOT = Path(__file__).resolve().parent.parent
COULEUR = {"k1": INK_LIGHT, "bon4": BLUE, "fk4": RED}
# tag de l'écran -> (échantillonneur, flags de run_sd_baseline.py écrasés pour cette ligne)
VARIANTES = {
    "S60": ("fk4", {"fk_schedule": [0, 20, 40, 60]}),
    "S40": ("fk4", {"fk_schedule": [0, 20, 40]}),
    "L2": ("fk4", {"lam": 2.0}), "L5": ("fk4", {"lam": 5.0}), "L20": ("fk4", {"lam": 20.0}),
    "A05": ("fk4", {"fk_threshold": 0.5}),
    "T1": ("fk4", {"fk_lam_schedule": "linear"}),
    "T2": ("fk4", {"fk_lam_schedule": "quad"}),
    "T1A05": ("fk4", {"fk_lam_schedule": "linear", "fk_threshold": 0.5}),
    "T2A05": ("fk4", {"fk_lam_schedule": "quad", "fk_threshold": 0.5}),
    "T1t": ("fk4", {"fk_lam_schedule": "linear", "fk_lam_placement": "tempering"}),
    "T2t": ("fk4", {"fk_lam_schedule": "quad", "fk_lam_placement": "tempering"}),
    "T1tA05": ("fk4", {"fk_lam_schedule": "linear", "fk_lam_placement": "tempering", "fk_threshold": 0.5}),
    "T2tA05": ("fk4", {"fk_lam_schedule": "quad", "fk_lam_placement": "tempering", "fk_threshold": 0.5}),
}


def feuille(ax, images, scores, thumb, couleur):
    """Jusqu'à quatre vignettes sur une ligne, le score en haut à gauche, l'argmax encadré."""
    gap = max(3, thumb // 40)
    sheet = np.ones((thumb + 2 * gap, 4 * (thumb + gap) + gap, 3))
    for i, im in enumerate(images):
        arr = np.asarray(im.resize((thumb, thumb))) / 255.0
        x0 = gap + i * (thumb + gap)
        sheet[gap:gap + thumb, x0:x0 + thumb] = arr
    ax.imshow(sheet, interpolation="lanczos")
    best = int(np.argmax(scores))
    for i, s in enumerate(scores):
        x0 = gap + i * (thumb + gap)
        ax.text(x0 + 4, gap + 12, f"{s:+.2f}", fontsize=7.5, color="white", family="monospace",
                va="top", path_effects=[matplotlib.patheffects.withStroke(linewidth=1.6, foreground="black")])
    x0 = gap + best * (thumb + gap)
    ax.add_patch(matplotlib.patches.Rectangle((x0 - 1.5, gap - 1.5), thumb + 3, thumb + 3,
                                              fill=False, linewidth=2.2, edgecolor=couleur))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", required=True, help="prompt_id:seed, dans l'ordre d'affichage")
    p.add_argument("--json", default=str(ROOT / "results" / "sd_baseline.json"))
    p.add_argument("--prompts", default=str(ROOT / "data" / "imagereward-benchmark-prompts.json"))
    # les mêmes valeurs que run_sd_baseline.py : ce sont celles du JSON
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--lam", type=float, default=10.0)
    p.add_argument("--fk-schedule", type=int, nargs="+", default=[0, 20, 40, 60, 80])
    p.add_argument("--fk-threshold", type=float, default=1.0)
    p.add_argument("--fk-lam-schedule", default="constant", choices=["constant", "linear", "quad"])
    p.add_argument("--fk-lam-placement", default="terminal", choices=["terminal", "tempering"])
    p.add_argument("--rows", nargs="+", default=["k1", "bon4", "fk4"],
                   help="lignes, dans l'ordre : k1, bon4, fk4 ou un tag de VARIANTES")
    p.add_argument("--variants", default=str(ROOT / "results" / "sd_variants"))
    p.add_argument("--reward-vae", default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--ir-cache", default="/home/onyxia/work/ir_cache")
    p.add_argument("--thumb", type=int, default=150)
    p.add_argument("--out", default=str(ROOT / "figures" / "fig6_sd_grid.png"))
    p.add_argument("--keep", default=str(ROOT / "samples" / "sd_grid"), help="les PNG régénérés")
    args = p.parse_args()

    import ImageReward as RM
    from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline

    prompts = load_prompts(Path(args.prompts), None)
    index = {pid: (i, txt) for i, (pid, txt) in enumerate(prompts)}
    runs = json.loads(Path(args.json).read_text())["runs"]
    for tag in args.rows:
        if tag in VARIANTES and (Path(args.variants) / f"{tag}.json").exists():
            runs += json.loads((Path(args.variants) / f"{tag}.json").read_text())["runs"]

    pipe = StableDiffusionPipeline.from_pretrained(
        REPO, torch_dtype=torch.float16, variant="fp16", safety_checker=None,
    ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    ir = RM.load("ImageReward-v1.0", device="cuda", download_root=args.ir_cache)
    FK["vae"] = AutoencoderKL.from_pretrained(args.reward_vae, torch_dtype=torch.float16).to("cuda")
    FK["ir"] = ir

    keep = Path(args.keep)
    keep.mkdir(parents=True, exist_ok=True)
    ordre = args.rows
    resultats = []
    for case in args.cases:
        pid, seed = case.split(":")
        seed = int(seed)
        i, prompt = index[pid]
        effective = seed * 1000 + i
        par_sampler = {}
        for name in ordre:
            sampler, overrides = VARIANTES.get(name, (name, {}))
            fn, n = SAMPLERS[sampler]
            g = torch.Generator("cuda").manual_seed(effective)
            res = fn(pipe, prompt, n, g, argparse.Namespace(**{**vars(args), **overrides}))
            images = res[0] if isinstance(res, tuple) else res
            scores = ir.score(prompt, images)
            scores = [scores] if not isinstance(scores, list) else scores
            for j, im in enumerate(images):
                im.save(keep / f"{pid}_seed{seed}_{name}_{j}.png")
            enregistre = [r for r in runs if r["prompt_id"] == pid and r["seed"] == seed and r["sampler"] == sampler
                          and r.get("lam_schedule", "constant") == overrides.get("fk_lam_schedule", args.fk_lam_schedule)
                          and r.get("lam_placement", "terminal") == overrides.get("fk_lam_placement", args.fk_lam_placement)
                          and r.get("threshold", 1.0) == overrides.get("fk_threshold", args.fk_threshold)
                          and r.get("lam", args.lam) == overrides.get("lam", args.lam)
                          and r.get("schedule_t", args.fk_schedule) == overrides.get("fk_schedule", args.fk_schedule)]
            ref = [round(v, 2) for v in enregistre[0]["ir"]] if enregistre else None
            lignees = res[1].get("n_lineages") if isinstance(res, tuple) else None
            print(f"{pid} seed={seed} {name:5s} régénéré {[round(v, 2) for v in scores]}  json {ref}"
                  f"  div_pix={diversite(images)}  lignées={lignees}", flush=True)
            par_sampler[name] = (images, scores, lignees)
            torch.cuda.empty_cache()
        resultats.append((pid, seed, prompt, par_sampler))

    n_cas = len(resultats)
    fig, axes = plt.subplots(len(ordre), n_cas, figsize=(4.6 * n_cas, 1.55 * len(ordre) + 0.4), squeeze=False)
    for c, (pid, seed, prompt, par_sampler) in enumerate(resultats):
        for r_, name in enumerate(ordre):
            images, scores, lignees = par_sampler[name]
            feuille(axes[r_][c], images, scores, args.thumb, COULEUR.get(name, RED))
            # sous les vignettes : la diversité pixel, et pour FK le nombre de x_T distincts survivants
            note = f"div {diversite(images):.3f}" if len(images) > 1 else ""
            if lignees is not None:
                note += f"  ·  {lignees} lignée{'s' if lignees > 1 else ''}/{len(images)}"
            axes[r_][c].set_xlabel(note, fontsize=7.5, color=INK_LIGHT, family="monospace", labelpad=2)
            if c == 0:
                axes[r_][c].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=10,
                                       color=INK, labelpad=10)
        titre = prompt if len(prompt) <= 48 else prompt[:46] + "…"
        maxs = "  ".join(f"{n} {max(par_sampler[n][1]):+.2f}" for n in ordre)
        axes[0][c].set_title(f"{titre}\nseed {seed}   ·   {maxs}", fontsize=8.5, color=INK, loc="left", pad=6)

    fig.suptitle(f"Figure 6 — SD v1.5 : ce que {', '.join(ordre)} ont produit sur le même x_T",
                 fontsize=12, color=INK, x=0.012, ha="left", y=0.995)
    # Une ligne par variante : sans ça le lecteur lit les quatre bandes comme du lambda constant.
    rampe = {"constant": "λ constant", "linear": "λ_t = λ_T p", "quad": "λ_t = λ_T p²"}
    conf = []
    for name in ordre:
        if name not in VARIANTES:
            continue
        o = VARIANTES[name][1]
        conf.append(f"{name} : {rampe[o.get('fk_lam_schedule', args.fk_lam_schedule)]}, "
                    f"seuil {o.get('fk_threshold', args.fk_threshold):g}, "
                    f"{o.get('fk_lam_placement', args.fk_lam_placement)}")
    fig.text(0.012, 0.004,
             "score = ImageReward recalculé sur l'image régénérée  ·  cadre = particule que l'ImageReward désigne  ·  "
             f"λ_T={args.lam:g}, MAX, calendrier {args.fk_schedule}  ·  fk4 : λ constant, seuil {args.fk_threshold:g}"
             + ("  ·  " + "  ·  ".join(conf) if conf else ""),
             fontsize=8, color=INK_LIGHT, family="monospace")
    fig.tight_layout(rect=(0, 0.025, 1, 0.94))
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
