"""Figure 4 : ImageReward et HPS de la particule que l'ImageReward désigne, par échantillonneur.

Deux panneaux, chacun son axe : les deux métriques n'ont pas la même échelle et un axe
commun écraserait HPS. Les barres partent de 0, sans quoi un HPS zoomé ferait passer du
bruit pour un effet.

Depuis la racine : `python scripts/plot_fig4_sd.py`.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figstyle import BLUE, GREEN, INK, INK_LIGHT, RED, aggregate, dress, load_runs

ROOT = Path(__file__).resolve().parent.parent
COULEURS = [INK_LIGHT, BLUE, RED, GREEN]


def panneau(ax, etiquettes, moyennes, erreurs, ylabel):
    x = np.arange(len(etiquettes))
    barres = ax.bar(x, moyennes, width=0.6, zorder=3,
                    color=[COULEURS[i % len(COULEURS)] for i in x],
                    yerr=erreurs, capsize=5, ecolor=INK, error_kw={"elinewidth": 1.2})
    ax.bar_label(barres, fmt="%.3f", fontsize=9.5, color=INK, padding=4)
    ax.set_xticks(x)
    ax.set_xticklabels(etiquettes, color=INK, fontsize=9.5)
    ax.set_xlim(-0.6, len(etiquettes) - 0.4)
    ax.set_ylabel(ylabel, color=INK, fontsize=10)
    ax.margins(y=0.14)
    dress(ax, grid="y")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", nargs="+", default=[str(ROOT / "results" / "sd_baseline.json")])
    p.add_argument("--out", default=str(ROOT / "figures" / "fig4_sd_ir_hps.png"))
    args = p.parse_args()

    brut = load_runs(args.json)
    # une moyenne par prompt d'abord : plusieurs seeds d'un même prompt ne sont pas des
    # tirages indépendants, et l'erreur-type affichée se prend sur les prompts
    groupes = {}
    for r in brut:
        groupes.setdefault((r["sampler"], r["prompt_id"]), []).append(r)
    runs = [{"sampler": s, "n": g[0]["n"],
             "ir_max": np.mean([x["ir_max"] for x in g]),
             "hps_at_ir_max": np.mean([x["hps_at_ir_max"] for x in g])}
            for (s, _), g in groupes.items()]

    xs, ir, ir_sd, cnt = aggregate(runs, "sampler", "ir_max")
    _, hps, hps_sd, _ = aggregate(runs, "sampler", "hps_at_ir_max")
    # budget croissant plutôt qu'alphabétique : k1, bon4, puis FK quand il arrivera
    n_par_sampler = {r["sampler"]: r["n"] for r in runs}
    ordre = np.argsort([n_par_sampler[s] for s in xs], kind="stable")
    xs, ir, ir_sd, hps, hps_sd, cnt = (v[ordre] for v in (xs, ir, ir_sd, hps, hps_sd, cnt))

    sem = np.sqrt(cnt)
    etiquettes = [f"{s}\nN={n_par_sampler[s]}, {c} prompts" for s, c in zip(xs, cnt)]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.9))
    panneau(axes[0], etiquettes, ir, ir_sd / sem, "ImageReward")
    panneau(axes[1], etiquettes, hps, hps_sd / sem, "HPS v2.1")
    axes[0].set_title("la reward qui guide", fontsize=10, color=INK, loc="left", pad=8)
    axes[1].set_title("le juge, que rien n'optimise", fontsize=10, color=INK, loc="left", pad=8)

    fig.suptitle("Figure 4 — SD v1.5 : l'écart se creuse sur la reward qui guide, à peine sur le juge",
                 fontsize=12.5, color=INK, x=0.012, ha="left", y=0.98)
    fig.text(0.012, 0.005,
             f"seeds {sorted({r['seed'] for r in brut})}  ·  "
             f"meilleure particule au sens de l'ImageReward, son HPS lu au même indice  ·  "
             "barres d'erreur = erreur-type sur les prompts",
             fontsize=8.5, color=INK_LIGHT, family="monospace")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    for i, s in enumerate(xs):
        print(f"{s:6s} IR={ir[i]:+.4f} ± {ir_sd[i]/sem[i]:.4f}  "
              f"HPS={hps[i]:.4f} ± {hps_sd[i]/sem[i]:.4f}  ({cnt[i]} prompts)")
    print(out)


if __name__ == "__main__":
    main()
