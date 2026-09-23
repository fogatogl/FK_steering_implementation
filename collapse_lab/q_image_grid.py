"""Q. La grille d'images appariee : une ligne par bras, quatre images, cadre = racine x_T.

Deux modes.
  --choose : imprime les six prompt_id de la regle pre-enregistree (docs/protocol_sd.md,
             22/09) depuis out/probe.json a n = 40 : les deux prompts de rang 20 et 21 sur
             l'ir_max de ctl ; les deux ou ctl bat le plus bon4 ; parmi ceux ou floor2 garde
             quatre lignees, les deux au plus haut ir_max de floor2. Egalites par prompt_id.
  (defaut) : lit out/images/index.json et dessine, par prompt, une grille bras x 4 cases,
             cadre colore par racine (memes couleurs que F1, scripts/figstyle.py si present),
             ir sous chaque image, ir_max et lignees en marge. Sortie out/fig_grid_<pid>.png
             et la planche des six, out/fig_grid_all.png.
Rien n'est calcule ici : les ir viennent de index.json, les racines de root_slots.
"""
import argparse, json
from pathlib import Path

import numpy as np

from commun import bon4

LAB = Path(__file__).resolve().parent
OUT = LAB / "out"
# les bras rejoues dans la session du 22-23/09 portent le suffixe _b1 : memes seeds, mais le
# chemin smc n'est pas rejouable d'une session a l'autre (ASSESSMENT.md) ; R1 est de cette session
ARMS = ["lam0_b1", "ctl_b1", "floor2_b1", "R1"]
COULEURS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]   # racine 0..3, partagees avec F1


def par_bras():
    runs = json.loads((OUT / "probe.json").read_text())["runs"]
    par = {}
    for r in runs:
        par.setdefault(r["arm"], {})[r["prompt_id"]] = r
    return par


def choisir(par):
    bon = bon4()
    ctl = par["ctl"]
    ids = sorted(ctl)[:40]
    rang = sorted(ids, key=lambda c: (ctl[c]["ir_max"], c))
    medians = rang[19:21]
    gagne = sorted(ids, key=lambda c: (-(ctl[c]["ir_max"] - bon[c]["ir_max"]), c))
    gagne = [c for c in gagne if c not in medians][:2]
    f2 = par["floor2"]
    quatre = sorted((c for c in ids if c in f2 and f2[c]["n_lineages"] == 4 and c not in medians + gagne),
                    key=lambda c: (-f2[c]["ir_max"], c))[:2]
    return medians + gagne + quatre


def grille(pids, par):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    index = json.loads((OUT / "images" / "index.json").read_text())
    img = {(e["arm"], e["prompt_id"], e["slot"]): e for e in index}
    arms = [a for a in ARMS if any(k[0] == a for k in img)]
    fig_all, axes_all = plt.subplots(len(pids) * len(arms), 4, figsize=(4 * 2.2, len(pids) * len(arms) * 2.5))
    for p, pid in enumerate(pids):
        fig, axes = plt.subplots(len(arms), 4, figsize=(4 * 2.6, len(arms) * 2.9))
        for a, arm in enumerate(arms):
            r = par[arm][pid]
            for j in range(4):
                e = img[(arm, pid, j)]
                im = Image.open(OUT / e["path"])
                for ax in (axes[a, j], axes_all[p * len(arms) + a, j]):
                    ax.imshow(im)
                    ax.set_xticks([]); ax.set_yticks([])
                    for s in ax.spines.values():
                        s.set_edgecolor(COULEURS[e["root"]]); s.set_linewidth(4)
                    ax.set_xlabel(f"ir {e['ir']:+.2f}", fontsize=8)
            axes[a, 0].set_ylabel(f"{arm}\nir_max {r['ir_max']:+.2f}\n{r['n_lineages']} lignee(s)", fontsize=8)
            axes_all[p * len(arms) + a, 0].set_ylabel(f"{pid[-4:]} {arm}", fontsize=7)
        fig.suptitle(f"{pid} : {r['prompt'][:70]}", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / f"fig_grid_{pid}.png", dpi=110)
        plt.close(fig)
    fig_all.tight_layout()
    fig_all.savefig(OUT / "fig_grid_all.png", dpi=90)
    print(OUT / "fig_grid_all.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--choose", action="store_true")
    p.add_argument("--pids", nargs="*")
    args = p.parse_args()
    par = par_bras()
    if args.choose:
        print(" ".join(choisir(par)))
        return
    # la grille se dessine sur les prompts qui ont des images : la regle de choix a ete
    # appliquee a la nuit du 22/09, et le rejeu des ctl a change leurs ir_max (derive inter-session)
    index = json.loads((OUT / "images" / "index.json").read_text())
    pids = args.pids or sorted({e["prompt_id"] for e in index})
    grille(pids, par)


if __name__ == "__main__":
    main()
