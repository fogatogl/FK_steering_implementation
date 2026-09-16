"""Figure 1 : r_best en fonction de n (best-of-n), 3 seeds, barres d'erreur.

Référence = E[max de n tirages] par bootstrap sur les images libres, rescorées
ici pour rester dans l'unité du sweep. En 1.4, la courbe FK = un appel de plus
à courbe(), filtré sur `methode`.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from smc.rewards import reward

ENCRE = "#2b2b2b"
ENCRE_FAIBLE = "#8a8a8a"
BEST_OF_N = "#4a6fa5"
REFERENCE = "#c0563a"


def charger(chemins):
    runs = []
    for c in chemins:
        d = json.loads(Path(c).read_text())
        runs += d["runs"] if isinstance(d, dict) else d
    return runs


def agreger(runs, cle="r_best"):
    """(ns triés, moyenne par n, écart-type par n, nb de seeds par n)."""
    ns = sorted({r["n"] for r in runs})
    moy, ecart, effectif = [], [], []
    for n in ns:
        v = np.array([r[cle] for r in runs if r["n"] == n], dtype=float)
        moy.append(v.mean())
        ecart.append(v.std(ddof=1) if len(v) > 1 else 0.0)
        effectif.append(len(v))
    return np.array(ns), np.array(moy), np.array(ecart), np.array(effectif)


def courbe(ax, runs, label, couleur):
    ns, moy, ecart, _ = agreger(runs)
    ax.errorbar(ns, moy, yerr=ecart, color=couleur, linewidth=1.8, marker="o",
                markersize=5, capsize=4, elinewidth=1.2, zorder=3, label=label)
    ax.plot([r["n"] for r in runs], [r["r_best"] for r in runs], "o",
            color=couleur, markersize=4, alpha=0.35, markeredgewidth=0, zorder=2)
    return ns, moy, ecart


def max_attendu(scores, ns, tirages, graine):
    """Bootstrap : max de n tirages avec remise dans la distribution libre."""
    rng = np.random.default_rng(graine)
    moy, bas, haut = [], [], []
    for n in ns:
        m = rng.choice(scores, size=(tirages, int(n)), replace=True).max(axis=1)
        moy.append(m.mean())
        bas.append(np.percentile(m, 10))
        haut.append(np.percentile(m, 90))
    return np.array(moy), np.array(bas), np.array(haut)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", nargs="+", default=[str(RACINE / "results" / "bestofn.json")])
    p.add_argument("--ref", default=str(RACINE / "samples" / "free_seed12345.pt"))
    p.add_argument("--tirages", type=int, default=20000)
    p.add_argument("--graine-bootstrap", type=int, default=0)
    p.add_argument("--out", default=str(RACINE / "figures" / "fig1_best_of_n.png"))
    args = p.parse_args()

    runs = charger(args.json)

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ns, moy, ecart = courbe(ax, runs, "best-of-n (moyenne ± σ sur les seeds)", BEST_OF_N)

    libre = None
    if args.ref and Path(args.ref).exists():
        libre = reward(torch.load(args.ref, map_location="cpu")).numpy()
        ref_moy, ref_bas, ref_haut = max_attendu(libre, ns, args.tirages, args.graine_bootstrap)
        ax.plot(ns, ref_moy, color=REFERENCE, linewidth=1.6, linestyle="--", zorder=1,
                label=f"E[max de n tirages] — BORNE BASSE : le bootstrap ne peut\n"
                      f"pas dépasser le max des {len(libre)} rewards libres ({libre.max():+.2f})")
        ax.fill_between(ns, ref_bas, ref_haut, color=REFERENCE, alpha=0.12, linewidth=0,
                        zorder=0, label="décile 10–90 du max")
        ax.axhline(libre.mean(), color=ENCRE_FAIBLE, linewidth=1, linestyle=":", zorder=0)
        ax.annotate("moyenne libre", xy=(ns[0], libre.mean()), xytext=(0, -12),
                    textcoords="offset points", fontsize=8, color=ENCRE_FAIBLE)

    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(int(n)) for n in ns])
    ax.set_xlabel("n  (échantillons tirés, coût = n × T appels réseau)", color=ENCRE, fontsize=10)
    ax.set_ylabel("r_best", color=ENCRE, fontsize=10)
    ax.grid(color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    for bord in ("top", "right"):
        ax.spines[bord].set_visible(False)
    for bord in ("left", "bottom"):
        ax.spines[bord].set_color("#cccccc")
    ax.tick_params(colors=ENCRE_FAIBLE, labelsize=9)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")

    seeds = sorted({r["seed"] for r in runs})
    appels = {r["n"]: r["n_model_calls"] for r in runs}
    fig.suptitle("Figure 1 — best-of-n : la reward du meilleur échantillon croît avec n",
                 fontsize=12.5, color=ENCRE, x=0.012, ha="left", y=0.98)
    fig.text(0.012, 0.005,
             f"seeds {seeds}  ·  n ∈ {[int(n) for n in ns]}  ·  "
             f"appels réseau {min(appels.values())}–{max(appels.values())}",
             fontsize=8.5, color=ENCRE_FAIBLE, family="monospace")

    sortie = Path(args.out)
    sortie.parent.mkdir(exist_ok=True)
    fig.savefig(sortie, dpi=170, bbox_inches="tight", facecolor="white")
    for n, m, e in zip(ns, moy, ecart):
        print(f"n={int(n):3d}  r_best={m:+.4f} ± {e:.4f}")
    print(sortie)


if __name__ == "__main__":
    main()
