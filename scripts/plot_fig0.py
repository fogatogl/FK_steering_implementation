"""Figure 0: free-model images, their score, the score distribution."""
import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.patheffects
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from figstyle import BLUE, INK, INK_LIGHT, RED, dress
from smc.rewards import reward

ROOT = Path(__file__).resolve().parent.parent


def grid(ax, images, scores):
    n = len(images)
    side = int(np.ceil(np.sqrt(n)))
    img = (images.permute(0, 2, 3, 1).numpy() + 1.0) / 2.0
    img = np.clip(img, 0.0, 1.0)

    h, w = img.shape[1], img.shape[2]
    gap = 2
    sheet = np.ones((side * (h + gap) + gap, side * (w + gap) + gap, 3))
    for i in range(n):
        r, c = divmod(i, side)
        y, x = gap + r * (h + gap), gap + c * (w + gap)
        sheet[y:y + h, x:x + w] = img[i]

    ax.imshow(sheet, interpolation="nearest")
    for i in range(n):
        r, c = divmod(i, side)
        ax.text(gap + c * (w + gap) + 1, gap + r * (h + gap) + 4,
                f"{scores[i]:+.2f}", fontsize=5, color="white", family="monospace",
                path_effects=[matplotlib.patheffects.withStroke(linewidth=1.4,
                                                               foreground="black")])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def histogram(ax, scores, bins):
    mu, sigma = scores.mean(), scores.std(ddof=1)

    ax.hist(scores, bins=bins, color=BLUE, edgecolor="white", linewidth=0.8)
    ax.axvline(mu, color=RED, linewidth=2, zorder=3)
    ax.axvspan(mu - sigma, mu + sigma, color=RED, alpha=0.10, zorder=0)

    top = ax.get_ylim()[1]
    ax.plot(scores, np.full_like(scores, -top * 0.045), "|",
            color=INK_LIGHT, markersize=7, markeredgewidth=0.9, clip_on=False)
    ax.set_ylim(-top * 0.09, top)

    ax.annotate(f"mean {mu:+.3f}", xy=(mu, top * 0.97),
                xytext=(6, 0), textcoords="offset points",
                color=RED, fontsize=9, va="top")
    ax.annotate(f"±1σ = ±{sigma:.3f}", xy=(mu + sigma, top * 0.86),
                xytext=(6, 0), textcoords="offset points",
                color=INK_LIGHT, fontsize=9, va="top")

    ax.set_xlabel("reward  r(x) = [mean(R) − ½·(mean(G) + mean(B))] / σ_ref", color=INK, fontsize=10)
    ax.set_ylabel("images", color=INK, fontsize=10)
    dress(ax, grid="y")
    return mu, sigma


def sigma_verdict(scores, draws=4000, seed=0):
    """sigma_ref is up to date iff sigma is 1; compare 1 to the bootstrap CI, not to sigma."""
    if len(scores) < 128:
        return (f"σ_ref: undecidable on n={len(scores)} (σ={scores.std(ddof=1):.3f}) — "
                f"the bootstrap cannot recover tails absent from the sample.")
    rng = np.random.default_rng(seed)
    bs = np.std(rng.choice(scores, (draws, len(scores))), axis=1, ddof=1)
    lo, hi = np.percentile(bs, [5, 95])
    if lo <= 1.0 <= hi:
        return (f"σ_ref in smc/rewards.py: up to date — σ={scores.std(ddof=1):.3f}, "
                f"CI90 [{lo:.3f}, {hi:.3f}] contains 1")
    return (f"σ_ref in smc/rewards.py: STALE — σ={scores.std(ddof=1):.3f}, "
            f"CI90 [{lo:.3f}, {hi:.3f}] excludes 1 → multiply σ_ref by {scores.std(ddof=1):.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", default=None)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--tag", default="free")
    p.add_argument("--bins", type=int, default=32)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    path = Path(args.json) if args.json else \
        ROOT / "results" / f"{args.tag}_samples_seed{args.seed}.json"
    data = json.loads(Path(path).read_text())
    images = torch.load(ROOT / data["images"], map_location="cpu")
    # Rescored rather than read from the JSON: the figure follows `reward`, hence the unit of lambda.
    scores = reward(images).numpy()

    order = np.argsort(scores)
    images, sorted_scores = images[order], scores[order]

    fig = plt.figure(figsize=(12.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.16)
    ax_grid = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1])

    grid(ax_grid, images, sorted_scores)
    mu, sigma = histogram(ax_hist, scores, args.bins)

    ax_grid.set_title(f"{data['n']} free samples (seed {data['seed']}), sorted by reward",
                      fontsize=9.5, color=INK, loc="left", pad=8)
    ax_hist.set_title("Reward distribution", fontsize=10, color=INK, loc="left", pad=8)
    fig.suptitle("Figure 0 — the \"red\" reward on unguided sampling",
                 fontsize=13, color=INK, x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.015,
             f"n={data['n']}  mean={mu:+.3f}  σ={sigma:.3f}  "
             f"min={scores.min():+.3f}  max={scores.max():+.3f}  "
             f"range={scores.max() - scores.min():.3f}\n"
             + sigma_verdict(scores),
             fontsize=9, color=INK_LIGHT, family="monospace")

    out = Path(args.out) if args.out else ROOT / "figures" / f"fig0_{args.tag}_reward.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"{out}  mean={mu:+.4f} std={sigma:.4f} "
          f"min={scores.min():+.4f} max={scores.max():+.4f}")


if __name__ == "__main__":
    main()
