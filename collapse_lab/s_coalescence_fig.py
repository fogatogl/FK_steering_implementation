"""F6: the number of roots an arm keeps is predicted from its weights and its resampler alone, one point per arm on the diagonal.

Reuses the model of n_coalescence.py (imported, not recomputed): for each arm, E[final roots]
under the resampler the arm used (comb integrated over u, or multinomial at every step),
averaged over the prompts, against the observed mean of n_lineages. Error bars = standard
error over the prompts, both axes. Diagonal = perfect prediction. Three points are labelled
and coloured (ctl, adapt, floor2, figstyle.arm_color / arm_label); the others are light grey,
grouped in the legend under figstyle.OTHER_ARMS.

Appendix (figures/f6_coalescence_appendix): the same points, plus, for the arms that used the
comb (lam0 aside, it never resampled), a hollow marker at the multinomial prediction on the
same weights, joined to the filled one: what the authors' resampler would have left.

Data: collapse_lab/out/probe.json through n_coalescence.par (the runs that carry
logG_at_schedule; fifteen arms, seed 2024), and the reruns of ctl, floor2 and lam0 in sessions C
and D (100 prompts each, one process), drawn as squares in the colour of their arm.
Output: figures/f6_coalescence and figures/f6_coalescence_appendix, PNG and SVG, in one run.
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

LAB = Path(__file__).resolve().parent
ROOT = LAB.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(LAB))
import figstyle as fs

with contextlib.redirect_stdout(io.StringIO()):      # n_coalescence prints its tables on import
    import n_coalescence as nc

LABELLED = ("ctl", "adapt", "floor2")
OFFSET = {"ctl": (8, -9), "adapt": (8, -3), "floor2": (-18, 26)}   # label offsets in points

stats = {}
for a in nc.ORDRE:
    d = nc.par[a]; ps = sorted(d)
    multinomial = d[ps[0]].get("resampler", "systematic") == "multinomial"
    obs = np.array([d[p]["n_lineages"] for p in ps], float)
    own = np.array([nc.predire(d[p], "multinomial" if multinomial else "systematic")[0] for p in ps])
    mult = own if multinomial else np.array([nc.predire(d[p], "multinomial")[0] for p in ps])
    se = lambda x: x.std(ddof=1) / len(x) ** 0.5
    stats[a] = dict(n=len(ps), obs=obs.mean(), obs_se=se(obs), pred=own.mean(), pred_se=se(own),
                    mult=mult.mean(), multinomial=multinomial, k=d[ps[0]]["k"])
k = max(s["k"] for s in stats.values())
lo, hi = 1 - 0.05 * (k - 1), k + 0.05 * (k - 1)

print(__doc__.splitlines()[0], "\n")
print(f"  {'arm':7s} {'n':>3s} | {'observed':>8s} {'predicted':>9s} {'resampler':>11s} | {'multinomial on the same weights':>31s}")
for a, s in stats.items():
    print(f"  {a:7s} {s['n']:3d} | {s['obs']:8.2f} {s['pred']:9.2f} {'multinomial' if s['multinomial'] else 'systematic':>11s} | "
          f"{s['mult']:31.2f}")
print("  source: collapse_lab/out/probe.json via n_coalescence.par\n")


def panel(ax, appendix):
    ax.plot([lo, hi], [lo, hi], color=fs.GREY, lw=1, ls="--", zorder=1)
    for a, s in stats.items():
        base, rerun = a.split("_")[0], a.endswith(("_C", "_D"))
        col = fs.arm_color(base) if base in LABELLED else fs.GREY_LIGHT
        ax.errorbar(s["pred"], s["obs"], xerr=s["pred_se"], yerr=s["obs_se"], fmt="s" if rerun else "o", color=col,
                    ms=5 if rerun else 6, capsize=2, lw=1, zorder=3 if base in LABELLED else 2)
        if a in LABELLED:
            ax.annotate(fs.arm_label(a), (s["pred"], s["obs"]), textcoords="offset points",
                        xytext=OFFSET[a], color=col, fontsize=9, va="center")
        if appendix and not s["multinomial"] and a != "lam0":
            ax.plot([s["pred"], s["mult"]], [s["obs"], s["obs"]], color=fs.GREY_LIGHT, lw=0.8, zorder=1)
            ax.plot(s["mult"], s["obs"], "o", mfc="none", mec=col, ms=6, zorder=2)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("final roots predicted from the weights and the resampler")
    ax.set_ylabel("final roots observed (mean over prompts)")
    fs.dress(ax, grid="both")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", color=fs.GREY_LIGHT, ls="", ms=6, label=fs.OTHER_ARMS),
               Line2D([], [], marker="s", color=fs.GREY_DARK, ls="", ms=5, label="reruns, sessions C and D")]
    if appendix:
        handles.append(Line2D([], [], marker="o", mfc="none", mec=fs.GREY_DARK, ls="", ms=6,
                              label="multinomial on the same weights"))
    ax.legend(handles=handles, loc="upper left")


for appendix, name in ((False, "f6_coalescence"), (True, "f6_coalescence_appendix")):
    fig, ax = fs.figure(5.6)
    panel(ax, appendix)
    png, svg = fs.save(fig, ROOT / "figures" / name)
    print("wrote", png, svg)
