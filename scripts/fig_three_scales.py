"""F8: at every scale, from CIFAR 32 px to Stable Diffusion 512 px, raising lambda drives the minimum ESS over the run towards one particle carrying all the weight.

Three aligned panels, lambda on the x axis, one quantity: the minimum over the run of the
effective sample size divided by k (1 = no weight degeneracy, 1/k = one particle carries
everything), grey dotted reference at 1. Sources, nothing hard-coded:
  CIFAR 32 px   results/sweep_lambda_classifier.json, `fk` runs, ess_min / k, mean and
                standard error over the 3 seeds per lambda (k = 16, T = 1000);
  CelebA 256 px results/sweep_lambda_hub_classifier.json, same fields (k = 16, DDIM 50);
  SD 512 px     collapse_lab/out/probe.json, arms lam0 / lam2 / ctl (lambda 0 / 2 / 10),
                min(ess_at_schedule) / k on the prompts the three arms share, mean and
                standard error over those prompts (k = 4, DDIM 100).
Pixel diversity and the reweighting target live in scripts/fig_three_scales_appendix.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle

ROOT = HERE.parent
SCALES = (("CIFAR 32 px", "sweep_lambda_classifier.json"),
          ("CelebA 256 px", "sweep_lambda_hub_classifier.json"),
          ("SD 512 px", None))
SD_ARMS = ("lam0", "lam2", "ctl")


def sem(v):
    return v.std(ddof=1) / len(v) ** .5 if len(v) > 1 else 0.0


def scale_sweep(json_name):
    """{lambda: (ESS_min / k mean, standard error, n)} from a lambda sweep file."""
    d = json.loads((ROOT / "results" / json_name).read_text())
    runs = [r for r in d["runs"] if r.get("method") != "best_of_n"]
    k = d["k"]
    out = {}
    for lam in sorted({r["lam"] for r in runs}):
        ess = np.array([r["ess_min"] / k for r in runs if r["lam"] == lam])
        out[lam] = (ess.mean(), sem(ess), len(ess))
    return out, k


def scale_sd():
    """{lambda: (ESS_min / k mean, standard error, n)} from the collapse lab probe."""
    runs = json.loads((ROOT / "collapse_lab" / "out" / "probe.json").read_text())["runs"]
    by_arm = {}
    for r in runs:
        by_arm.setdefault(r["arm"], {})[r["prompt_id"]] = r
    common = sorted(set.intersection(*(set(by_arm[a]) for a in SD_ARMS)))
    out = {}
    for arm in SD_ARMS:
        d = by_arm[arm]
        ess = np.array([min(d[c]["ess_at_schedule"]) / d[c]["k"] for c in common])
        out[d[common[0]]["lam"]] = (ess.mean(), sem(ess), len(common))
    return out, d[common[0]]["k"]


def load():
    data = []
    for name, json_name in SCALES:
        e, k = scale_sweep(json_name) if json_name else scale_sd()
        data.append((name, e, k))
    return data


def main():
    data = load()
    for name, e, k in data:
        print(f"{name}, k = {k}")
        for lam, (m, s, n) in sorted(e.items()):
            print(f"  lambda {lam:4g}  ESS_min / k = {m:.3f} +/- {s:.3f}  (n = {n}), ESS_min = {m * k:.2f} of {k}")
        print(f"  floor of the ratio (one particle carries everything): 1/k = {1 / k:.3f}")
    print("error bars: standard error over the 3 seeds (CIFAR, CelebA) or over the prompts the three SD arms share")

    fig, axes = figstyle.figure(2.9, ncols=3, sharey=True)
    for ax, (name, e, k) in zip(axes, data):
        lams = sorted(e)
        ax.axhline(1, color=figstyle.GREY_LIGHT, ls=":", lw=1)
        # the floor of the ratio: one particle carries all the weight
        ax.axhline(1 / k, color=figstyle.GREY, ls="--", lw=0.9)
        ax.annotate("one particle, 1/k", (min(e), 1 / k), xytext=(6, 4), textcoords="offset points",
                    ha="left", va="bottom", color=figstyle.GREY, fontsize=9)
        ax.errorbar(lams, [e[l][0] for l in lams], yerr=[e[l][1] for l in lams],
                    color=figstyle.GREY_DARK, marker="o", ms=4, lw=1.4, capsize=2)
        # the scales do not share their lambdas: symlog, linear under 0.5
        ax.set_xscale("symlog", linthresh=0.5)
        ax.set_xticks(lams)
        ax.set_xticklabels([f"{l:g}" for l in lams])
        ax.minorticks_off()
        ax.set_ylim(0, 1.08)
        ax.set_xlabel(f"λ, {name}, k = {k}")
        figstyle.dress(ax)
    axes[0].set_ylabel("min ESS over the run / k")
    png, svg = figstyle.save(fig, ROOT / "figures" / "f8_three_scales")
    print(png.relative_to(ROOT), svg.relative_to(ROOT))


if __name__ == "__main__":
    main()
