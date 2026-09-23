"""O. L'arbre genealogique des quatre particules, un panneau par bras, meme prompt (figure F1).

Lit out/probe.json : `ancestors_at_schedule` (la matrice des ancetres aux pas planifies,
enregistree depuis le 22/09 ; les runs anterieurs ne l'ont pas et sont sautes),
`logG_at_schedule` (les poids normalises du pas, epaisseur des aretes), `ir` (sous chaque
feuille), `root_slots` (couleur = racine x_T, memes couleurs que la grille F2).
Niveaux de haut en bas : x_T, puis chaque pas planifie ; une arete relie la case j du pas m
a son parent anc[m][j] au pas precedent. Un noeud dont personne ne descend meurt la.

  python collapse_lab/o_ancestry_fig.py --pid 005695-0057 --arms lam0 ctl floor2 R1
"""
import argparse, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB = Path(__file__).resolve().parent
COULEURS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]   # racine 0..3, partagees avec q_image_grid.py


def racines_par_niveau(anc):
    """anc : (M, k). Renvoie, pour chaque niveau 0..M (0 = x_T), la racine de chaque case."""
    k = len(anc[0])
    niveaux = [list(range(k))]
    for m in range(len(anc)):
        prev = niveaux[-1]
        niveaux.append([prev[anc[m][j]] for j in range(k)])
    return niveaux


def dessiner(ax, r, titre):
    anc = r["ancestors_at_schedule"]
    lg = np.array(r["logG_at_schedule"])
    w = np.exp(lg - lg.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    ts = sorted(r["schedule_t"], reverse=True)
    k = r["k"]; M = len(anc)
    niv = racines_par_niveau(anc)
    for m in range(M):
        for j in range(k):
            parent = anc[m][j]
            # le poids qui a fait copier le parent : celui du parent au pas m
            ax.plot([parent, j], [m, m + 1], color=COULEURS[niv[m][parent]], lw=0.6 + 4.5 * w[m][parent], alpha=0.85, zorder=1)
    for m in range(M + 1):
        for j in range(k):
            vivant = m == M or any(anc[m][i] == j for i in range(k))
            ax.scatter(j, m, s=90 if vivant else 40, color=COULEURS[niv[m][j]],
                       edgecolor="k" if vivant else "none", lw=0.6, zorder=2, marker="o" if vivant else "x")
    for j in range(k):
        ax.text(j, M + 0.35, f"{r['ir'][j]:+.2f}", ha="center", fontsize=7)
    ax.set_yticks(range(M + 1))
    ax.set_yticklabels(["x_T"] + [f"t={t}" for t in ts], fontsize=7)
    ax.set_xticks(range(k)); ax.set_xticklabels([f"case {j}" for j in range(k)], fontsize=7)
    ax.invert_yaxis()
    ax.set_ylim(M + 0.7, -0.5)
    ax.set_title(f"{titre}\n{r['n_lineages']} racine(s), ir_max {r['ir_max']:+.2f}", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pid", required=True)
    p.add_argument("--arms", nargs="+", default=["lam0_b1", "ctl_b1", "floor2_b1", "R1"],
                   help="_b1 : les bras rejoues dans la session du 22-23/09, la seule qui porte les ancetres")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    runs = json.loads((LAB / "out" / "probe.json").read_text())["runs"]
    par = {(r["arm"], r["prompt_id"]): r for r in runs}
    arms = [a for a in args.arms if (a, args.pid) in par and "ancestors_at_schedule" in par[(a, args.pid)]]
    if not arms:
        raise SystemExit(f"aucun run de {args.pid} avec ancestors_at_schedule pour {args.arms}")
    fig, axes = plt.subplots(1, len(arms), figsize=(3.2 * len(arms), 4.2), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, a in zip(axes, arms):
        dessiner(ax, par[(a, args.pid)], a)
    fig.suptitle(f"{args.pid} : {par[(arms[0], args.pid)]['prompt'][:80]}", fontsize=9)
    fig.tight_layout()
    out = Path(args.out) if args.out else LAB / "out" / f"fig_ancestry_{args.pid}.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".svg"))
    print(out, "bras :", " ".join(arms))


if __name__ == "__main__":
    main()
