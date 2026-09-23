"""Figure F5 : le meme effondrement a trois echelles.

CIFAR 32 px (classifieur, k = 16, T = 1000), CelebA-HQ 256 px (classifieur lunettes, k = 16,
DDIM 50) et Stable Diffusion 512 px (ImageReward, k = 4, DDIM 100). Deux panneaux, lambda en
abscisse :
  gauche : ESS minimale le long du debruitage, divisee par k (1 = aucune degenerescence des
           poids, 1/k = une particule porte tout) ;
  droite : diversite pixel des finales, la RMSE moyenne par paire a 64 x 64 dans [0, 1], la
           definition de div_pix (scripts/run_sd_baseline.py), rapportee a sa valeur libre.
Donnees : results/sweep_lambda_classifier.json + samples/sweep_lambda_classifier.pt (CIFAR),
results/sweep_lambda_hub_classifier.json + samples/sweep_lambda_hub_classifier.pt (CelebA),
collapse_lab/out/probe.json bras lam0 / lam2 / ctl (SD, lambda 0 / 2 / 10, memes prompts).
Les .pt ne sont que dans le checkout principal (/home/onyxia/work/diffusion-models/samples).
Rien de code en dur : chaque point vient d'un fichier nomme ci-dessus.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle

ROOT = HERE.parent
MAIN = Path("/home/onyxia/work/diffusion-models")
SAMPLES = MAIN / "samples"


def div_pix(x):
    """x : (k, 3, H, W) dans [-1, 1]. RMSE moyenne par paire a 64 x 64 dans [0, 1]."""
    x = ((x + 1) / 2).clamp(0, 1)
    x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False, antialias=True)
    k = x.shape[0]
    d = [float(((x[i] - x[j]) ** 2).mean().sqrt()) for i in range(k) for j in range(i + 1, k)]
    return float(np.mean(d))


def echelle_pt(json_name, pt_name):
    d = json.loads((ROOT / "results" / json_name).read_text())
    runs = [r for r in d["runs"] if r["method"] != "best_of_n"] if "method" in d["runs"][0] else d["runs"]
    k = d["k"]
    imgs = torch.load(SAMPLES / pt_name, map_location="cpu", weights_only=False)
    lams = sorted({r["lam"] for r in runs})
    out = {}
    for lam in lams:
        rr = [r for r in runs if r["lam"] == lam]
        ess = np.array([r["ess_min"] / k for r in rr])
        cles = [f"difference_systematic_lam{lam:g}_seed{r['seed']}" for r in rr]
        dv = np.array([div_pix(imgs[c]) for c in cles if c in imgs])
        out[lam] = (ess.mean(), ess.std(ddof=1) / len(ess) ** .5, dv.mean(), dv.std(ddof=1) / len(dv) ** .5 if len(dv) > 1 else 0.0, len(rr))
    return out, k


def echelle_sd():
    runs = json.loads((ROOT / "collapse_lab" / "out" / "probe.json").read_text())["runs"]
    par = {}
    for r in runs:
        par.setdefault(r["arm"], {})[r["prompt_id"]] = r
    out = {}
    for arm in ("lam0", "lam2", "ctl"):
        d = par[arm]
        com = sorted(set(d) & set(par["lam0"]))          # memes prompts pour les trois lambda
        ess = np.array([min(d[c]["ess_at_schedule"]) / d[c]["k"] for c in com])
        dv = np.array([d[c]["div_pix"] for c in com])
        lam = d[com[0]]["lam"]
        out[lam] = (ess.mean(), ess.std(ddof=1) / len(ess) ** .5, dv.mean(), dv.std(ddof=1) / len(dv) ** .5, len(com))
    return out, 4


def main():
    cifar, k_c = echelle_pt("sweep_lambda_classifier.json", "sweep_lambda_classifier.pt")
    celeba, k_h = echelle_pt("sweep_lambda_hub_classifier.json", "sweep_lambda_hub_classifier.pt")
    sd, k_s = echelle_sd()
    print(__doc__.splitlines()[0], "\n")
    for nom, e, k in (("CIFAR 32 px", cifar, k_c), ("CelebA 256 px", celeba, k_h), ("SD 512 px", sd, k_s)):
        print(f"{nom}, k = {k}")
        print(f"  {'lambda':>6s} | {'ESS_min / k':>11s} | {'div_pix':>8s} {'/ libre':>8s} | n")
        libre = e[min(e)][2]
        for lam, (em, es, dm, ds, n) in sorted(e.items()):
            print(f"  {lam:6g} | {em:11.3f} | {dm:8.3f} {dm / libre:8.2f} | {n}")

    # panneau A5 : la cible elle-meme. ESS de exp(lambda * ir) sur les quatre tirages libres de
    # bon4 (100 prompts, seed 2024), contre lambda ; le point de fk4 a t = 80 a lambda = 10.
    base = json.loads((ROOT / "results" / "sd_baseline.json").read_text())["runs"]
    bon = [r for r in base if r["sampler"] == "bon4" and r["seed"] == 2024]
    ref = [r for r in json.loads((ROOT / "results" / "sd_ref_fields100.json").read_text())["runs"] if r["sampler"] == "fk4"]

    def ess_w(logw):
        w = np.exp(logw - logw.max()); w /= w.sum()
        return 1 / (w ** 2).sum()

    lams_c = [0.5, 1, 2, 5, 10]
    cible = {l: np.array([ess_w(l * np.array(r["ir"])) for r in bon]) / 4 for l in lams_c}
    fk80 = np.array([r["ess_at_schedule"][0] for r in ref]) / 4
    print(f"cible sur 4 tirages libres, ESS / k median par lambda : " + ", ".join(f"{l:g}: {np.median(v):.2f}" for l, v in cible.items())
          + f" ; fk4 a t = 80, lambda 10 : {np.median(fk80):.2f}")

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 3.6))
    a3.plot(lams_c, [np.median(cible[l]) for l in lams_c], marker="o", color=figstyle.INK, lw=1.4, label="cible : 4 tirages libres reponderes")
    a3.fill_between(lams_c, [np.percentile(cible[l], 25) for l in lams_c], [np.percentile(cible[l], 75) for l in lams_c], color=figstyle.INK, alpha=0.12)
    a3.errorbar([10], [np.median(fk80)], yerr=[[np.median(fk80) - np.percentile(fk80, 25)], [np.percentile(fk80, 75) - np.median(fk80)]],
                fmt="D", color=figstyle.RED, ms=6, capsize=3, label="fk4, premier pas planifie (t = 80)")
    a3.set_xscale("symlog", linthresh=0.5); a3.set_xticks([0.5, 1, 2, 5, 10]); a3.set_xticklabels(["0.5", "1", "2", "5", "10"], fontsize=8); a3.minorticks_off()
    a3.set_ylim(0, 1.05); a3.set_xlabel("lambda"); a3.set_ylabel("ESS / k (mediane, Q1-Q3)")
    a3.set_title("la cible : ce que quatre particules peuvent porter", fontsize=9)
    a3.legend(fontsize=7, loc="upper right")
    figstyle.dress(a3)
    styles = (("CIFAR 32 px, k=16", cifar, figstyle.BLUE, "o"),
              ("CelebA 256 px, k=16", celeba, figstyle.GREEN, "s"),
              ("SD 512 px, k=4", sd, figstyle.RED, "D"))
    tous = sorted(set(cifar) | set(celeba) | set(sd))
    for nom, e, col, mk in styles:
        lams = sorted(e)
        a1.errorbar(lams, [e[l][0] for l in lams], yerr=[e[l][1] for l in lams], color=col, marker=mk, ms=5, lw=1.4, capsize=2, label=nom)
        libre = e[lams[0]][2]
        a2.errorbar(lams, [e[l][2] / libre for l in lams], yerr=[e[l][3] / libre for l in lams], color=col, marker=mk, ms=5, lw=1.4, capsize=2, label=nom)
    # les trois echelles n'ont pas les memes lambda : axe symlog, lineaire sous 0.5
    for ax in (a1, a2):
        ax.set_xscale("symlog", linthresh=0.5)
        ax.set_xticks(tous)
        ax.set_xticklabels([f"{l:g}" for l in tous], fontsize=8)
        ax.minorticks_off()
    a1.set_ylabel("ESS minimale / k")
    a1.set_ylim(0, 1.05)
    a1.axhline(1 / 16, color=figstyle.INK_LIGHT, ls=":", lw=1)
    a1.axhline(1 / 4, color=figstyle.INK_LIGHT, ls=":", lw=1)
    a1.set_title("les poids : une particule porte tout", fontsize=9)
    a2.set_ylabel("diversite pixel des finales / libre")
    a2.set_ylim(0, 1.1)
    a2.set_title("les images : la meme forme aux trois echelles", fontsize=9)
    for ax in (a1, a2):
        ax.set_xlabel("lambda")
        figstyle.dress(ax)
    a1.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    out = ROOT / "figures" / "fig8_three_scales"
    for ext in ("png", "svg"):
        fig.savefig(f"{out}.{ext}", dpi=150)
    print(f"\n-> {out}.png")


if __name__ == "__main__":
    main()
