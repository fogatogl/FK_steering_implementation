"""L. La cible elle-meme : que vaut l'ESS de exp(lam * ir) sur quatre tirages libres ?

Aucun echantillonneur n'est en cause ici. On prend les quatre images de bon4 (quatre
racines menees librement), on les reponderer par exp(lam * ImageReward), et on lit
l'ESS. C'est la concentration de la cible p(x0) exp(lam r(x0)) restreinte a quatre
candidats : le plafond de diversite que quatre particules peuvent porter a ce lambda,
quoi que fasse le noyau, le potentiel ou le calendrier.
"""
import json
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent.parent / "results"
base = json.loads((R / "sd_baseline.json").read_text())["runs"]
bon = [r for r in base if r["sampler"] == "bon4" and r["seed"] == 2024]
ref = json.loads((R / "sd_ref_fields100.json").read_text())["runs"]
fk = [r for r in ref if r["sampler"] == "fk4" and r["seed"] == 2024]


def ess(logw):
    w = np.exp(logw - logw.max())
    w /= w.sum()
    return 1 / (w ** 2).sum()


print(__doc__.splitlines()[0], f"\n  {len(bon)} prompts, etendue mediane de l'ir final libre :",
      f"{np.median([np.ptp(r['ir']) for r in bon]):.2f}\n")
print(f"  {'lambda':>6s} | {'ESS mediane':>11s} | {'Q1 - Q3':>13s} | part < 1.5")
for lam in (0.5, 1, 2, 5, 10):
    e = np.array([ess(lam * np.array(r["ir"])) for r in bon])
    print(f"  {lam:6.1f} | {np.median(e):11.2f} | {np.percentile(e, 25):5.2f} - {np.percentile(e, 75):5.2f} | {(e < 1.5).mean():.0%}")

e80 = np.median([r["ess_at_schedule"][0] for r in fk])
print(f"\n  a comparer : ESS mediane de fk4 au premier pas planifie (t = 80), lam = 10 : {e80:.2f}")
