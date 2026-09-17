"""Figure 1: r_best against n (best-of-n), 3 seeds, error bars.

Reference = E[max of n draws], bootstrapped from the free samples, rescored
here to stay in the sweep's unit.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from figstyle import BLUE, INK, INK_LIGHT, RED, aggregate, dress, load_runs
from smc.rewards import reward

ROOT = Path(__file__).resolve().parent.parent


def curve(ax, runs, label, color, key_x="n", key_y="r_best"):
    xs, mean, std, _ = aggregate(runs, key_x, key_y)
    ax.errorbar(xs, mean, yerr=std, color=color, linewidth=1.8, marker="o",
                markersize=5, capsize=4, elinewidth=1.2, zorder=3, label=label)
    ax.plot([r[key_x] for r in runs], [r[key_y] for r in runs], "o",
            color=color, markersize=4, alpha=0.35, markeredgewidth=0, zorder=2)
    return xs, mean, std


def expected_max(scores, ns, draws, seed):
    """Bootstrap: max of n draws with replacement from the free distribution."""
    rng = np.random.default_rng(seed)
    mean, lo, hi = [], [], []
    for n in ns:
        m = rng.choice(scores, size=(draws, int(n)), replace=True).max(axis=1)
        mean.append(m.mean())
        lo.append(np.percentile(m, 10))
        hi.append(np.percentile(m, 90))
    return np.array(mean), np.array(lo), np.array(hi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", nargs="+", default=[str(ROOT / "results" / "bestofn.json")])
    p.add_argument("--ref", default=str(ROOT / "samples" / "free_seed12345.pt"))
    p.add_argument("--draws", type=int, default=20000)
    p.add_argument("--bootstrap-seed", type=int, default=0)
    p.add_argument("--out", default=str(ROOT / "figures" / "fig1_best_of_n.png"))
    args = p.parse_args()

    runs = load_runs(args.json)

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ns, mean, std = curve(ax, runs, "best-of-n (mean ± σ over seeds)", BLUE)

    free = None
    if args.ref and Path(args.ref).exists():
        free = reward(torch.load(args.ref, map_location="cpu")).numpy()
        ref_mean, ref_lo, ref_hi = expected_max(free, ns, args.draws, args.bootstrap_seed)
        ax.plot(ns, ref_mean, color=RED, linewidth=1.6, linestyle="--", zorder=1,
                label=f"E[max of n draws] — LOWER BOUND: the bootstrap cannot\n"
                      f"exceed the max of the {len(free)} free rewards ({free.max():+.2f})")
        ax.fill_between(ns, ref_lo, ref_hi, color=RED, alpha=0.12, linewidth=0,
                        zorder=0, label="10–90th percentile of the max")
        ax.axhline(free.mean(), color=INK_LIGHT, linewidth=1, linestyle=":", zorder=0)
        ax.annotate("free mean", xy=(ns[0], free.mean()), xytext=(0, -12),
                    textcoords="offset points", fontsize=8, color=INK_LIGHT)

    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(int(n)) for n in ns])
    ax.set_xlabel("n  (samples drawn, cost = n × T network calls)", color=INK, fontsize=10)
    ax.set_ylabel("r_best", color=INK, fontsize=10)
    dress(ax)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")

    seeds = sorted({r["seed"] for r in runs})
    calls = {r["n"]: r["n_model_calls"] for r in runs}
    fig.suptitle("Figure 1 — best-of-n: the best sample's reward grows with n",
                 fontsize=12.5, color=INK, x=0.012, ha="left", y=0.98)
    fig.text(0.012, 0.005,
             f"seeds {seeds}  ·  n ∈ {[int(n) for n in ns]}  ·  "
             f"network calls {min(calls.values())}–{max(calls.values())}",
             fontsize=8.5, color=INK_LIGHT, family="monospace")

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    for n, m, e in zip(ns, mean, std):
        print(f"n={int(n):3d}  r_best={m:+.4f} ± {e:.4f}")
    print(out)


if __name__ == "__main__":
    main()
