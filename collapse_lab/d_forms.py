"""D. Le premier pas ne depend pas du potentiel : max, difference, statistic y coincident.

A gate vide (smc/fk.py, _potential_terms) les trois branches donnent logG = lam * r_t.
fk4_stat (max/statistic) et fk4_diff (difference) partagent prompts et seeds, et
aucun reechantillonnage n'a lieu avant le pas t=80 : leurs trajectoires y sont donc
la MEME. Si les lignes 0 coincident bit a bit, le choix du potentiel ne peut rien
changer a la lignee, puisque c'est la que la lignee meurt.
"""
import json
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent.parent / "results"
load = lambda p: json.loads((R / p).read_text())["runs"]

stat = {r["prompt_id"]: r for r in load("sd_variants/fk4_stat.json")}
diff = {r["prompt_id"]: r for r in load("sd_variants/fk4_diff.json")}
com = sorted(set(stat) & set(diff))
print(__doc__.splitlines()[0], "\n")

r0s = np.array([stat[c]["r_at_schedule"][0] for c in com])
r0d = np.array([diff[c]["r_at_schedule"][0] for c in com])
e0s = np.array([stat[c]["ess_at_schedule"][0] for c in com])
e0d = np.array([diff[c]["ess_at_schedule"][0] for c in com])
print(f"  {len(com)} prompts communs")
print(f"  ligne 0 de r_phi identique         : ecart max {np.abs(r0s - r0d).max():.2e}")
print(f"  ESS au pas 0 identique             : ecart max {np.abs(e0s - e0d).max():.2e}")
print(f"  racine gardee identique            : "
      f"{sum(set(stat[c]['root_slots']) == set(diff[c]['root_slots']) for c in com)}/{len(com)} prompts")

# et ce que les potentiels changent ensuite : rien de mesurable sur la reward
for tag, d in (("max/statistic", stat), ("difference", diff)):
    ir = np.array([d[c]["ir_max"] for c in com])
    lin = np.array([d[c]["n_lineages"] for c in com])
    nr = np.array([d[c]["n_resamplings"] for c in com])
    print(f"  {tag:14s} ir_max moyen {ir.mean():+.4f}  lignees {lin.mean():.2f}  "
          f"reechantillonnages {nr.mean():.2f}")
a = np.array([stat[c]["ir_max"] for c in com]) - np.array([diff[c]["ir_max"] for c in com])
print(f"  difference appariee max - difference : {a.mean():+.4f} +/- {a.std(ddof=1)/len(a)**.5:.4f}")

# les lignes suivantes, elles, divergent : les deux runs ne sont plus le meme run
for m, t in enumerate([80, 60, 40, 20, 0]):
    d = np.abs(np.array([stat[c]["r_at_schedule"][m] for c in com])
               - np.array([diff[c]["r_at_schedule"][m] for c in com])).max()
    print(f"    t={t:3d} : ecart max entre les deux fichiers sur r_phi = {d:.3f}"
          + ("   <- identiques" if d < 1e-6 else ""))
