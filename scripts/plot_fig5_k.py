"""Figure 5: what growing k buys, at a budget matched on network calls.

Same two panels as figure 2. r_sample is the particle drawn from the final
weights, what the method returns; r_max is the best of the k. Grey is
best-of-k, which spends the same k*T network calls.

k is on a log2 axis, lambda is the line style. The two potentials do not cover
the same k: `difference` was swept on 2-16, `max` on 4-16.

best-of-k appears in both result files, run from the same effective seed, so
the same (k, seed) is dropped to one record: concatenating would show six
seeds where three were run.

From the root: `python scripts/plot_fig5_k.py`.
"""
import argparse
from pathlib import Path

import matplotlib
import matplotlib.lines
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figstyle import BLUE, INK, INK_LIGHT, RED, aggregate, dress, load_runs

ROOT = Path(__file__).resolve().parent.parent
COLOR = {"difference": BLUE, "max": RED}
DASH = {0.5: ":", 1.0: "--", 2.0: "-"}


def panel(ax, fk, baseline, key):
    for pot in sorted({r["potential"] for r in fk}):
        for lam in sorted({r["lam"] for r in fk if r["potential"] == pot}):
            sub = [r for r in fk if r["potential"] == pot and r["lam"] == lam]
            ks, mean, std, _ = aggregate(sub, "k", key)
            ax.errorbar(np.log2(ks), mean, yerr=std, color=COLOR[pot],
                        linestyle=DASH.get(lam, "-"), linewidth=1.6,
                        marker="o", markersize=4.5, capsize=3, elinewidth=1.0,
                        zorder=3, label=f"{pot} λ={lam:g}")

    ks, mean, std, _ = aggregate(baseline, "k", key)
    ax.errorbar(np.log2(ks), mean, yerr=std, color=INK_LIGHT, linewidth=1.6,
                marker="s", markersize=4.5, capsize=3, elinewidth=1.0,
                zorder=2, label="best-of-k")

    allk = sorted({r["k"] for r in fk} | {r["k"] for r in baseline})
    ax.set_xticks(np.log2(allk))
    ax.set_xticklabels([str(k) for k in allk])
    ax.set_xlabel("k  (budget = k·T network calls)", color=INK, fontsize=10)
    ax.set_ylabel(key, color=INK, fontsize=10)
    dress(ax)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", nargs="+",
                   default=[str(ROOT / "results" / "sweep_k_classifier.json"),
                            str(ROOT / "results" / "sweep_k_max_classifier.json")])
    p.add_argument("--resampler", default="systematic")
    p.add_argument("--out", default=str(ROOT / "figures" / "fig5_sweep_k.png"))
    args = p.parse_args()

    all_runs = load_runs(args.json)
    fk = [r for r in all_runs
          if r["method"] == "fk" and r["resampler"] == args.resampler]
    baseline, seen = [], set()
    for r in all_runs:
        if r["method"] != "best_of_n" or (r["k"], r["seed"]) in seen:
            continue
        seen.add((r["k"], r["seed"]))
        baseline.append(r)
    if not fk:
        raise SystemExit(f"no fk run with resampler={args.resampler}")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))
    for ax, key in zip(axes, ("r_sample", "r_max")):
        panel(ax, fk, baseline, key)
    axes[0].set_title("r_sample — the particle drawn from the final weights",
                      fontsize=10, color=INK, loc="left", pad=8)
    axes[1].set_title("r_max — the best of the k particles",
                      fontsize=10, color=INK, loc="left", pad=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, frameon=False, ncol=4,
               loc="upper left", bbox_to_anchor=(0.012, 0.945))

    seeds = sorted({r["seed"] for r in fk})
    fig.suptitle("Figure 5 — k at matched budget: the weights carry the gain, the max does not",
                 fontsize=12.5, color=INK, x=0.012, ha="left", y=0.98)
    fig.text(0.012, 0.005,
             f"CIFAR DDPM, classifier reward on class 3, T=1000  ·  seeds {seeds}  ·  "
             f"resampler {args.resampler}, ESS < k/2  ·  error bars = σ over seeds  ·  "
             "k=2 never resamples: ESS ≥ 1 = k/2",
             fontsize=8.5, color=INK_LIGHT, family="monospace")
    fig.tight_layout(rect=(0, 0.03, 1, 0.91))

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")

    for pot in sorted({r["potential"] for r in fk}):
        for lam in sorted({r["lam"] for r in fk if r["potential"] == pot}):
            sub = [r for r in fk if r["potential"] == pot and r["lam"] == lam]
            ks, s_mean, s_std, n = aggregate(sub, "k", "r_sample")
            _, m_mean, m_std, _ = aggregate(sub, "k", "r_max")
            _, e_mean, _, _ = aggregate(sub, "k", "ess_min")
            for i, k in enumerate(ks):
                print(f"{pot:10s} λ={lam:<4g} k={k:<3g} r_sample={s_mean[i]:+.4f} ± {s_std[i]:.4f}  "
                      f"r_max={m_mean[i]:+.4f} ± {m_std[i]:.4f}  ess_min={e_mean[i]:5.2f}  ({n[i]} seeds)")
    ks, s_mean, s_std, n = aggregate(baseline, "k", "r_sample")
    _, m_mean, m_std, _ = aggregate(baseline, "k", "r_max")
    for i, k in enumerate(ks):
        print(f"best-of-k         k={k:<3g} r_sample={s_mean[i]:+.4f} ± {s_std[i]:.4f}  "
              f"r_max={m_mean[i]:+.4f} ± {m_std[i]:.4f}  ({n[i]} seeds)")
    print(out)


if __name__ == "__main__":
    main()
