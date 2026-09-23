"""S. Figure F3 : racines finales predites depuis les poids contre observees, un point par bras.

Reprend le modele de n_coalescence.py (importe, pas recalcule autrement) : pour chaque bras,
E[racines finales] sous le resampler que le bras a utilise (peigne integre sur u, ou multinomial
a chaque pas), moyenne sur les prompts, contre la moyenne observee de n_lineages. Barres = erreur-
type sur les prompts. Diagonale = prediction parfaite. Le multinomial des auteurs applique aux
memes poids que le peigne est trace en creux pour les bras systematiques : ce que leur resampler
aurait laisse.
Sortie : out/fig_coalescence.png et .svg.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent / "scripts"))
try:
    import figstyle
    dress, INK, BLUE, RED = figstyle.dress, figstyle.INK_LIGHT, figstyle.BLUE, figstyle.RED
except ImportError:
    dress, INK, BLUE, RED = (lambda ax, grid="both": None), "#8a8a8a", "#4a6fa5", "#c0563a"

import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):      # n_coalescence imprime ses tables a l'import
    import n_coalescence as nc

fig, ax = plt.subplots(figsize=(5.2, 5))
for a in nc.ORDRE:
    d = nc.par[a]; ps = sorted(d)
    multinomial = d[ps[0]].get("resampler", "systematic") == "multinomial"
    obs = np.array([d[p]["n_lineages"] for p in ps], float)
    pred = np.array([nc.predire(d[p], "multinomial" if multinomial else "systematic")[0] for p in ps])
    ax.errorbar(pred.mean(), obs.mean(), xerr=pred.std(ddof=1) / len(ps) ** .5, yerr=obs.std(ddof=1) / len(ps) ** .5,
                fmt="s" if multinomial else "o", color=RED if multinomial else BLUE, ms=6, capsize=2, lw=1)
    ax.annotate(a, (pred.mean(), obs.mean()), textcoords="offset points", xytext=(6, -3), fontsize=8)
    if not multinomial and a not in ("lam0",):
        alt = np.mean([nc.predire(d[p], "multinomial")[0] for p in ps])
        ax.plot(alt, obs.mean(), marker="o", mfc="none", mec=INK, ms=5, lw=0)
ax.plot([0.9, 4.1], [0.9, 4.1], color=INK, lw=1, ls="--")
ax.set_xlim(0.9, 4.1); ax.set_ylim(0.9, 4.1)
ax.set_xlabel("racines finales predites depuis les poids et le resampler")
ax.set_ylabel("racines finales observees (moyenne sur les prompts)")
ax.set_title("La coalescence se lit sur les poids : quinze bras, un modele\n"
             "(carres : bras multinomiaux ; cercles vides : ce que le multinomial ferait des memes poids)", fontsize=8.5)
dress(ax)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(LAB / "out" / f"fig_coalescence.{ext}", dpi=150)
print(LAB / "out" / "fig_coalescence.png")
