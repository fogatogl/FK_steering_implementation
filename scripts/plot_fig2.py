"""Figure 2: lambda sweep for the three potentials, minimum ESS on a second axis.

Two panels. r_sample = the particle drawn from the final weights, what the
method actually returns; r_max = the best of the k, what it found. The grey
line is best-of-k, which lambda=0 must recover exactly.

lambda is categorical on the axis: 0 has no place on a log scale and the six
values are unreadable on a linear one. The three potentials are offset by
±0.07 so the error bars do not overlap.

From the root: `python scripts/plot_fig2.py`.
"""
import argparse
from pathlib import Path

import matplotlib
import matplotlib.lines
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figstyle import BLUE, GREEN, INK, INK_LIGHT, RED, aggregate, dress, load_runs

ROOT = Path(__file__).resolve().parent.parent
COLOR = {"difference": BLUE, "max": RED, "sum": GREEN}


def panel(ax, runs, baseline, potentials, lams, key, k):
    """One panel: the reward `key` per potential, min ESS on the right axis."""
    ax_ess = ax.twinx()
    offsets = np.linspace(-0.07, 0.07, len(potentials))

    for pot, dx in zip(potentials, offsets):
        sub = [r for r in runs if r["potential"] == pot]
        if not sub:
            continue
        xs, mean, std, _ = aggregate(sub, "lam", key)
        pos = np.array([lams.index(x) for x in xs], dtype=float) + dx
        ax.errorbar(pos, mean, yerr=std, color=COLOR[pot], linewidth=1.8,
                    marker="o", markersize=5, capsize=4, elinewidth=1.2,
                    zorder=3, label=pot)

        _, ess_mean, _, _ = aggregate(sub, "lam", "ess_min")
        ax_ess.plot(pos, ess_mean, color=COLOR[pot], linewidth=1.1,
                    linestyle=":", marker="s", markersize=3, alpha=0.75, zorder=2)

    if baseline:
        v = np.array([r[key] for r in baseline], dtype=float)
        ax.axhline(v.mean(), color=INK_LIGHT, linewidth=1.2, zorder=1)
        ax.axhspan(v.mean() - v.std(ddof=1), v.mean() + v.std(ddof=1),
                   color=INK_LIGHT, alpha=0.10, linewidth=0, zorder=0)

    ax.set_xticks(range(len(lams)))
    ax.set_xticklabels([f"{l:g}" for l in lams])
    ax.set_xlim(-0.45, len(lams) - 0.55)
    ax.set_xlabel("λ", color=INK, fontsize=10)
    ax.set_ylabel(key, color=INK, fontsize=10)
    dress(ax)

    ax_ess.set_yscale("log", base=2)
    ax_ess.set_ylim(0.85, k * 1.25)
    ticks = [2 ** i for i in range(int(np.log2(k)) + 1)]
    ax_ess.set_yticks(ticks)
    ax_ess.set_yticklabels([str(t) for t in ticks])
    ax_ess.set_ylabel(f"min ESS  (dotted, k={k})", color=INK_LIGHT, fontsize=9)
    ax_ess.tick_params(colors=INK_LIGHT, labelsize=8)
    for side in ("top", "left"):
        ax_ess.spines[side].set_visible(False)
    ax_ess.spines["right"].set_color("#dddddd")
    return ax_ess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", nargs="+",
                   default=[str(ROOT / "results" / "sweep_lambda.json")])
    p.add_argument("--resampler", default="systematic")
    p.add_argument("--potentials", nargs="+", default=["difference", "max", "sum"])
    p.add_argument("--out", default=str(ROOT / "figures" / "fig2_sweep_lambda.png"))
    args = p.parse_args()

    all_runs = load_runs(args.json)
    runs = [r for r in all_runs
            if r["method"] == "fk" and r["resampler"] == args.resampler]
    baseline = [r for r in all_runs if r["method"] == "best_of_n"]
    if not runs:
        raise SystemExit(f"no fk run with resampler={args.resampler}")

    lams = sorted({r["lam"] for r in runs})
    k = runs[0]["k"]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))
    for ax, key in zip(axes, ("r_sample", "r_max")):
        panel(ax, runs, baseline, args.potentials, lams, key, k)

    axes[0].set_title("r_sample — the particle drawn from the final weights",
                      fontsize=10, color=INK, loc="left", pad=8)
    axes[1].set_title("r_max — the best of the k particles",
                      fontsize=10, color=INK, loc="left", pad=8)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(matplotlib.lines.Line2D([], [], color=INK_LIGHT, linestyle=":",
                                           marker="s", markersize=3, linewidth=1.1))
    labels.append("min ESS (right axis)")
    fig.legend(handles, labels, fontsize=9, frameon=False, ncol=4,
               loc="upper left", bbox_to_anchor=(0.012, 0.945))

    seeds = sorted({r["seed"] for r in runs})
    fig.suptitle("Figure 2 — FK steering: reward rises with λ, ESS collapses",
                 fontsize=12.5, color=INK, x=0.012, ha="left", y=0.98)
    fig.text(0.012, 0.005,
             f"k={k}  ·  seeds {seeds}  ·  resampler {args.resampler}  ·  "
             f"{runs[0]['n_model_calls']} network calls per run  ·  "
             f"grey = best-of-{k} (mean ± σ)",
             fontsize=8.5, color=INK_LIGHT, family="monospace")
    fig.tight_layout(rect=(0, 0.03, 1, 0.91))

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")

    for pot in args.potentials:
        sub = [r for r in runs if r["potential"] == pot]
        if not sub:
            continue
        xs, s_mean, s_std, n = aggregate(sub, "lam", "r_sample")
        _, m_mean, m_std, _ = aggregate(sub, "lam", "r_max")
        _, e_mean, _, _ = aggregate(sub, "lam", "ess_min")
        for i, l in enumerate(xs):
            print(f"{pot:10s} λ={l:<5g} r_sample={s_mean[i]:+.4f} ± {s_std[i]:.4f}  "
                  f"r_max={m_mean[i]:+.4f} ± {m_std[i]:.4f}  "
                  f"ess_min={e_mean[i]:5.2f}  ({n[i]} seeds)")
    print(out)


if __name__ == "__main__":
    main()
