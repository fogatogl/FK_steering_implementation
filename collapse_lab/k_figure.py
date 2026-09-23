"""K. Deux panneaux : ou meurent les lignees, et pourquoi c'est au mauvais moment.

Gauche : nombre moyen de racines x_T survivantes apres chaque pas planifie, un
trait par bras de la sonde. Droite : l'inversion, la pression de selection
(etendue de lambda * r_phi, en nats) contre le niveau de la reward guide, le long
du debruitage. Sortie dans collapse_lab/out/, pas dans figures/ : ce n'est pas une
figure du billet.
"""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from figstyle import BLUE, GREEN, INK, INK_LIGHT, RED, dress

LAB = Path(__file__).resolve().parent
R = LAB.parent / "results"
runs = json.loads((LAB / "out" / "probe.json").read_text())["runs"]
par = {}
for r in runs:
    par.setdefault(r["arm"], {})[r["prompt_id"]] = r

ts = [80, 60, 40, 20, 0]
x = np.arange(len(ts))
fig, (g, d) = plt.subplots(1, 2, figsize=(10.2, 4.0))

style = {"ctl": (RED, "-", "lam=10, sans plancher  (la reference)"),
         "floor": (BLUE, "-", "lam=10 + plancher a 0  (le code publie)"),
         "lam2": (GREEN, "--", "lam=2, sans plancher"),
         "floor2": (INK_LIGHT, "--", "lam=2 + plancher"),
         "lam0": ("#b08b3e", ":", "lam=0  (controle : aucun pilotage)")}
for a, (c, ls, lab) in style.items():
    if a not in par:
        continue
    tr = np.array([par[a][p]["lineages_trace"] for p in sorted(par[a])], dtype=float)
    m, se = tr.mean(0), tr.std(0, ddof=1) / len(tr) ** .5
    g.plot(x, m, ls, color=c, linewidth=1.8, marker="o", markersize=4,
           label=f"{lab}  (n={len(tr)})")
    g.fill_between(x, m - se, m + se, color=c, alpha=0.12, linewidth=0)
g.axhline(4, color=INK_LIGHT, linewidth=0.8, linestyle=":")
g.set_xticks(x); g.set_xticklabels([f"t={t}" for t in ts])
g.set_ylim(0.8, 4.3)
g.set_ylabel("racines $x_T$ distinctes parmi les 4 cases", color=INK, fontsize=10)
g.set_title("Les lignees meurent au premier pas planifie", color=INK, fontsize=11, loc="left")
dress(g); g.legend(frameon=False, fontsize=8, labelcolor=INK_LIGHT, loc="lower left")

# droite : l'inversion pression / signal
fk = json.loads((R / "sd_variants" / "fk4_stat.json").read_text())["runs"]
rr = np.array([r["r_at_schedule"] for r in fk])
etendue = 10.0 * (rr.max(2) - rr.min(2))
niveau = rr.mean(2)
d.plot(x, np.median(etendue, 0), "-o", color=RED, linewidth=1.8, markersize=4,
       label="pression : etendue de $\\lambda\\,r_\\phi$ (nats)")
d.fill_between(x, np.percentile(etendue, 25, 0), np.percentile(etendue, 75, 0),
               color=RED, alpha=0.12, linewidth=0)
d2 = d.twinx()
d2.plot(x, np.median(niveau, 0), "-s", color=BLUE, linewidth=1.8, markersize=4,
        label="niveau de $r_\\phi$ (ImageReward)")
d2.axhline(0, color=BLUE, linewidth=0.8, linestyle=":")
d2.tick_params(colors=BLUE, labelsize=9)
d2.set_ylabel("niveau de $r_\\phi$", color=BLUE, fontsize=10)
for s in ("top",): d2.spines[s].set_visible(False)
d.set_xticks(x); d.set_xticklabels([f"t={t}" for t in ts])
d.set_ylabel("nats entre la meilleure et la pire case", color=RED, fontsize=10)
d.tick_params(axis="y", colors=RED)
d.set_title("La pression est maximale la ou la reward n'est pas encore une reward",
            color=INK, fontsize=11, loc="left")
dress(d, grid="x")
d.legend(frameon=False, fontsize=8, labelcolor=INK_LIGHT, loc="upper right")
d2.legend(frameon=False, fontsize=8, labelcolor=INK_LIGHT, loc="center right")

fig.tight_layout()
out = LAB / "out" / "collapse.png"
fig.savefig(out, dpi=160)
print(out)
