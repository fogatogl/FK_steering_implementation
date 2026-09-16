"""Figure 0 : images du modèle libre, leur score, la distribution du score."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.patheffects
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from smc.rewards import reward

ENCRE = "#2b2b2b"
ENCRE_FAIBLE = "#8a8a8a"
BARRE = "#4a6fa5"
REPERE = "#c0563a"


def grille(ax, images, scores):
    n = len(images)
    cote = int(np.ceil(np.sqrt(n)))
    img = (images.permute(0, 2, 3, 1).numpy() + 1.0) / 2.0
    img = np.clip(img, 0.0, 1.0)

    h, w = img.shape[1], img.shape[2]
    marge = 2
    planche = np.ones((cote * (h + marge) + marge, cote * (w + marge) + marge, 3))
    for i in range(n):
        l, c = divmod(i, cote)
        y, x = marge + l * (h + marge), marge + c * (w + marge)
        planche[y:y + h, x:x + w] = img[i]

    ax.imshow(planche, interpolation="nearest")
    for i in range(n):
        l, c = divmod(i, cote)
        ax.text(marge + c * (w + marge) + 1, marge + l * (h + marge) + 4,
                f"{scores[i]:+.2f}", fontsize=5, color="white", family="monospace",
                path_effects=[matplotlib.patheffects.withStroke(linewidth=1.4,
                                                               foreground="black")])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def histogramme(ax, scores, bins):
    mu, sigma = scores.mean(), scores.std(ddof=1)

    ax.hist(scores, bins=bins, color=BARRE, edgecolor="white", linewidth=0.8)
    ax.axvline(mu, color=REPERE, linewidth=2, zorder=3)
    ax.axvspan(mu - sigma, mu + sigma, color=REPERE, alpha=0.10, zorder=0)

    haut = ax.get_ylim()[1]
    ax.plot(scores, np.full_like(scores, -haut * 0.045), "|",
            color=ENCRE_FAIBLE, markersize=7, markeredgewidth=0.9, clip_on=False)
    ax.set_ylim(-haut * 0.09, haut)

    ax.annotate(f"moyenne {mu:+.3f}", xy=(mu, haut * 0.97),
                xytext=(6, 0), textcoords="offset points",
                color=REPERE, fontsize=9, va="top")
    ax.annotate(f"±1σ = ±{sigma:.3f}", xy=(mu + sigma, haut * 0.86),
                xytext=(6, 0), textcoords="offset points",
                color=ENCRE_FAIBLE, fontsize=9, va="top")

    ax.set_xlabel("reward  r(x) = [moy(R) − ½·(moy(G) + moy(B))] / σ_ref", color=ENCRE, fontsize=10)
    ax.set_ylabel("images", color=ENCRE, fontsize=10)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    for bord in ("top", "right"):
        ax.spines[bord].set_visible(False)
    for bord in ("left", "bottom"):
        ax.spines[bord].set_color("#cccccc")
    ax.tick_params(colors=ENCRE_FAIBLE, labelsize=9)
    return mu, sigma


def verdict_sigma(scores, tirages=4000, graine=0):
    """σ_ref est à jour ssi σ vaut 1 ; on compare 1 à l'IC bootstrap, pas à σ."""
    if len(scores) < 128:
        return (f"σ_ref : indécidable sur n={len(scores)} (σ={scores.std(ddof=1):.3f}) — "
                f"le bootstrap ne retrouve pas des queues absentes de l'échantillon.")
    rng = np.random.default_rng(graine)
    bs = np.std(rng.choice(scores, (tirages, len(scores))), axis=1, ddof=1)
    bas, haut = np.percentile(bs, [5, 95])
    if bas <= 1.0 <= haut:
        return (f"σ_ref de smc/rewards.py : à jour — σ={scores.std(ddof=1):.3f}, "
                f"IC90 [{bas:.3f}, {haut:.3f}] contient 1")
    return (f"σ_ref de smc/rewards.py : PÉRIMÉE — σ={scores.std(ddof=1):.3f}, "
            f"IC90 [{bas:.3f}, {haut:.3f}] exclut 1 → multiplier σ_ref par {scores.std(ddof=1):.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", default=None)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--tag", default="free")
    p.add_argument("--bins", type=int, default=32)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    chemin = Path(args.json) if args.json else \
        RACINE / "results" / f"{args.tag}_samples_seed{args.seed}.json"
    donnees = json.loads(Path(chemin).read_text())
    images = torch.load(RACINE / donnees["images"], map_location="cpu")
    # Rescorées et non relues du JSON : la figure suit `reward`, donc l'unité de λ.
    scores = reward(images).numpy()

    ordre = np.argsort(scores)
    images, scores_tries = images[ordre], scores[ordre]

    fig = plt.figure(figsize=(12.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.16)
    ax_grille = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1])

    grille(ax_grille, images, scores_tries)
    mu, sigma = histogramme(ax_hist, scores, args.bins)

    ax_grille.set_title(f"{donnees['n']} images libres (seed {donnees['seed']}), "
                        "triées par reward",
                        fontsize=9.5, color=ENCRE, loc="left", pad=8)
    ax_hist.set_title("Distribution de la reward", fontsize=10, color=ENCRE,
                      loc="left", pad=8)
    fig.suptitle("Figure 0 — la reward « rouge » sur l'échantillonnage non guidé",
                 fontsize=13, color=ENCRE, x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.015,
             f"n={donnees['n']}  moyenne={mu:+.3f}  σ={sigma:.3f}  "
             f"min={scores.min():+.3f}  max={scores.max():+.3f}  "
             f"étendue={scores.max() - scores.min():.3f}\n"
             + verdict_sigma(scores),
             fontsize=9, color=ENCRE_FAIBLE, family="monospace")

    sortie = Path(args.out) if args.out else RACINE / "figures" / f"fig0_{args.tag}_reward.png"
    sortie.parent.mkdir(exist_ok=True)
    fig.savefig(sortie, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"{sortie}  mean={mu:+.4f} std={sigma:.4f} "
          f"min={scores.min():+.4f} max={scores.max():+.4f}")


if __name__ == "__main__":
    main()
