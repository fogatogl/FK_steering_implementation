"""F7: what keeping more roots does to the best image's reward and to the mean of the four.

Two panels, arms on the y axis sorted by the mean number of roots kept (most at the top):
left the paired difference against ctl on ir_max (the reward of the best image, what table 1
measures), right on the mean of the four ir (what four clones hide). Point = mean over the
prompts paired with ctl, line = 95 % confidence interval (1.96 standard errors over the
prompts), vertical line at zero. In colour the arms the text discusses (floor2, adapt, late,
figstyle.arm_color), the others grey. The console table prints mean +/- standard error and
the mean roots per arm, plus bon4 - ctl as a reference row.

Data: collapse_lab/out/probe.json (fifteen arms, seed 2024, each paired by prompt to the ctl
run of its own machine group: session C for the fast card, session A then D for the A2,
finding 19), except the two arms with a 100-prompt run, drawn from it: floor2 from session C
(collapse_lab/out/probe_C.json, T4) and late from results/sd_s60_full.json at seed 2024 (A2); and
results/sd_baseline.json (bon4, console row only). Nothing hard-coded: everything comes from
the JSON files.
Output: figures/f7_two_rewards.png and .svg.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

LAB = Path(__file__).resolve().parent
ROOT = LAB.parent
sys.path.insert(0, str(ROOT / "scripts"))
import figstyle as fs
from commun import bon4, groupe, references

COLOURED = ("floor2", "adapt", "late")
Z = 1.96   # 95 % CI, normal approximation

runs = json.loads((LAB / "out" / "probe.json").read_text())["runs"]
par = {}
for r in runs:
    par.setdefault(r["arm"], {})[r["prompt_id"]] = r
par["floor2"] = {r["prompt_id"]: r for r in json.loads((LAB / "out" / "probe_C.json").read_text())["runs"]
                 if r["arm"] == "floor2"}
par["late"] = {r["prompt_id"]: r for r in json.loads((ROOT / "results" / "sd_s60_full.json").read_text())["runs"]
               if r["seed"] == 2024}
bon = bon4()
ctl = par["ctl"]
ORDRE = [a for a in ("late", "adapt", "floor", "lam2", "fadapt", "floor2", "thr05", "rise", "stat0", "multi", "vae", "idx", "R1", "lam0") if a in par]


REFS = references()


def diff(d, cle):
    com = [c for c in sorted(d) if c in REFS[groupe(d[c])]]
    x = np.array([cle(d[c]) - cle(REFS[groupe(d[c])][c]) for c in com])
    return x.mean(), x.std(ddof=1) / len(x) ** 0.5, len(com)


irmax = lambda r: r["ir_max"]
irmoy = lambda r: float(np.mean(r["ir"]))
lignees = lambda r: r["n_lineages"]

print(__doc__.splitlines()[0], "\n")
print(f"  {'arm':7s} {'n':>3s} | {'ir_max - ctl':>16s} | {'mean ir - ctl':>16s} | {'roots':>7s}")
rows = []
for a in ORDRE:
    m1, s1, n = diff(par[a], irmax)
    m2, s2, _ = diff(par[a], irmoy)
    lg = np.mean([lignees(par[a][c]) for c in par[a]])
    rows.append((a, n, m1, s1, m2, s2, lg))
    print(f"  {a:7s} {n:3d} | {m1:+.3f} +/- {s1:.3f} | {m2:+.3f} +/- {s2:.3f} | {lg:7.2f}")
ctl = REFS["fast"]   # bon4 ran on the fast card, 20/09
com = sorted(set(bon) & set(ctl))
b1 = np.array([bon[c]["ir_max"] - ctl[c]["ir_max"] for c in com])
b2 = np.array([np.mean(bon[c]["ir"]) - np.mean(ctl[c]["ir"]) for c in com])
print(f"  {'bon4':7s} {len(com):3d} | {b1.mean():+.3f} +/- {b1.std(ddof=1)/len(b1)**.5:.3f} | "
      f"{b2.mean():+.3f} +/- {b2.std(ddof=1)/len(b2)**.5:.3f} | {'4.00':>7s}   (reference, not drawn)")
print("  (mean +/- standard error over the prompts paired with ctl; the figure's lines are 1.96 standard errors)")

rows.sort(key=lambda r: r[6])            # fewest roots at the bottom, most at the top
y = np.arange(len(rows))
fig, axes = fs.figure(4.2, ncols=2, sharey=True)
for ax, (im, ise), xlabel in zip(axes, ((2, 3), (4, 5)), ("best image, minus FK (IR)", "mean of the four, minus FK (IR)")):
    ax.axvline(0, color=fs.GREY_DARK, lw=0.8, zorder=1)
    for yi, r in zip(y, rows):
        col = fs.arm_color(r[0]) if r[0] in COLOURED else fs.GREY
        ax.plot([r[im] - Z * r[ise], r[im] + Z * r[ise]], [yi, yi], color=col, lw=1.4, zorder=2, solid_capstyle="butt")
        ax.plot(r[im], yi, "o", color=col, ms=5, zorder=3)
    ax.set_xlabel(xlabel)
    fs.dress(ax, grid="x")
axes[0].set_yticks(y)
labels = [fs.arm_label(r[0]) for r in rows]
# two arms can share a figstyle label (adapt and fadapt): the arm key then disambiguates
labels = [f"{l} ({r[0]})" if labels.count(l) > 1 else l for l, r in zip(labels, rows)]
axes[0].set_yticklabels(labels)
axes[0].set_ylabel("arms, sorted by mean roots kept")
axes[0].set_ylim(-0.6, len(rows) - 0.4)
png, svg = fs.save(fig, ROOT / "figures" / "f7_two_rewards")
print("\nwrote", png.relative_to(ROOT), svg.relative_to(ROOT))
