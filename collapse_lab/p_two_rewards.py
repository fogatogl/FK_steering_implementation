"""P. ir_max et ir moyen cote a cote, par bras, apparies contre ctl (figure F4).

Lit out/probe.json et results/sd_baseline.json (bon4). Pour chaque bras : la difference
appariee par prompt contre ctl sur ir_max (la reward de la meilleure image, ce que la table 1
mesure) et sur la moyenne des quatre ir (ce que quatre clones cachent), erreur-type sur les
prompts, n en legende ; bon4 - ctl en repere horizontal ; les lignees gardees en axe
secondaire. Rien de code en dur : tout vient des JSON.
Sortie : out/fig_two_rewards.png et .svg, et la table imprimee.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent / "scripts"))
try:
    import figstyle  # noqa: F401  (style du depot si present)
except ImportError:
    pass
from commun import bon4

runs = json.loads((LAB / "out" / "probe.json").read_text())["runs"]
par = {}
for r in runs:
    par.setdefault(r["arm"], {})[r["prompt_id"]] = r
bon = bon4()
ctl = par["ctl"]
ORDRE = [a for a in ("late", "adapt", "floor", "lam2", "fadapt", "floor2", "thr05", "rise", "stat0", "multi", "vae", "idx", "R1", "lam0") if a in par]


def diff(d, cle):
    com = sorted(set(d) & set(ctl))
    x = np.array([cle(d[c]) - cle(ctl[c]) for c in com])
    return x.mean(), x.std(ddof=1) / len(x) ** 0.5, len(com)


irmax = lambda r: r["ir_max"]
irmoy = lambda r: float(np.mean(r["ir"]))
lignees = lambda r: r["n_lineages"]

print(__doc__.splitlines()[0], "\n")
print(f"  {'bras':7s} {'n':>3s} | {'ir_max - ctl':>16s} | {'ir moyen - ctl':>16s} | {'lignees':>7s}")
rows = []
for a in ORDRE:
    m1, s1, n = diff(par[a], irmax)
    m2, s2, _ = diff(par[a], irmoy)
    lg = np.mean([lignees(par[a][c]) for c in sorted(set(par[a]) & set(ctl))])
    rows.append((a, n, m1, s1, m2, s2, lg))
    print(f"  {a:7s} {n:3d} | {m1:+.3f} +/- {s1:.3f} | {m2:+.3f} +/- {s2:.3f} | {lg:7.2f}")
com = sorted(set(bon) & set(ctl))
b1 = np.array([bon[c]["ir_max"] - ctl[c]["ir_max"] for c in com])
b2 = np.array([np.mean(bon[c]["ir"]) - np.mean(ctl[c]["ir"]) for c in com])
print(f"  {'bon4':7s} {len(com):3d} | {b1.mean():+.3f} +/- {b1.std(ddof=1)/len(b1)**.5:.3f} | "
      f"{b2.mean():+.3f} +/- {b2.std(ddof=1)/len(b2)**.5:.3f} | {'4.00':>7s}   (repere)")

fig, ax = plt.subplots(figsize=(8.5, 4.2))
x = np.arange(len(rows)); w = 0.38
ax.bar(x - w / 2, [r[2] for r in rows], w, yerr=[r[3] for r in rows], capsize=3, color="#4c72b0", label="ir_max (meilleure des 4)")
ax.bar(x + w / 2, [r[4] for r in rows], w, yerr=[r[5] for r in rows], capsize=3, color="#dd8452", label="ir moyen des 4")
ax.axhline(0, color="k", lw=0.8)
ax.axhline(b1.mean(), color="#4c72b0", ls="--", lw=1, label=f"bon4 - ctl, ir_max ({b1.mean():+.2f})")
ax.axhline(b2.mean(), color="#dd8452", ls="--", lw=1, label=f"bon4 - ctl, ir moyen ({b2.mean():+.2f})")
ax.set_xticks(x); ax.set_xticklabels([f"{r[0]}\nn={r[1]}" for r in rows], fontsize=8)
ax.set_ylabel("difference appariee contre ctl (ImageReward)")
ax2 = ax.twinx()
ax2.plot(x, [r[6] for r in rows], "k.-", lw=1, ms=7, label="lignees gardees")
ax2.set_ylim(0.8, 4.2); ax2.set_ylabel("racines x_T distinctes a la fin (sur 4)")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left")
ax.set_title("Le prix des lignees : plat sur ir_max, croissant sur ir moyen", fontsize=10)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(LAB / "out" / f"fig_two_rewards.{ext}", dpi=150)
print(f"\n-> {LAB / 'out' / 'fig_two_rewards.png'}")
