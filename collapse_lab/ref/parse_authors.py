"""Depouillement de R0 : le code des auteurs contre bon4, ctl et la table 1 du papier.

Lit results/sd_authors_R0.json (run_authors.py), results/sd_baseline.json (bon4, k1, fk4 a
seed 2024), results/sd_ref_fields100.json (ctl = la ligne FK du code actuel) et
out/probe.json (R1 si present). Tout est apparie par prompt_id ; R0 n'a pas nos x_T, la
difference appariee est donc a lire comme deux echantillons sur les memes prompts, pas comme
un contrefactuel a bruit fixe. Regle de decision pre-enregistree (docs/protocol_sd.md, 22/09) :
ecart au papier "ferme" si R1 - bon4 >= +0.12 apparie, "borne" sinon.
"""
import json
from pathlib import Path
import numpy as np

LAB = Path(__file__).resolve().parents[1]
R = LAB.parent / "results"
CHECKOUT = Path("/home/onyxia/work/diffusion-models/results")   # la sortie GPU est ecrite la
PAPIER = {"k1": 0.187, "bon4": 0.737, "fk4": 0.898}              # experiments_new.tex:82-85, SD v1.5


def se(x):
    x = np.asarray(x, float)
    return f"{x.mean():+.3f} +/- {x.std(ddof=1) / len(x) ** .5:.3f} (n={len(x)})"


import sys
src = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    CHECKOUT / "sd_authors_R0.json" if (CHECKOUT / "sd_authors_R0.json").exists() else R / "sd_authors_R0.json")
R0 = json.loads(src.read_text())["runs"] if src.exists() else []
if not R0:
    print(f"(pas encore de R0 : {src} absent)\n")
base = json.loads((R / "sd_baseline.json").read_text())["runs"]
bon = {r["prompt_id"]: r for r in base if r["sampler"] == "bon4" and r["seed"] == 2024}
k1 = {r["prompt_id"]: r for r in base if r["sampler"] == "k1" and r["seed"] == 2024}
ctl = {r["prompt_id"]: r for r in json.loads((R / "sd_ref_fields100.json").read_text())["runs"]
       if r["sampler"] == "fk4" and r["seed"] == 2024}
probe = json.loads((LAB / "out" / "probe.json").read_text())["runs"]
R1 = {r["prompt_id"]: r for r in probe if r["arm"] == "R1"}

print(__doc__.splitlines()[0], "\n")
for cfg in sorted({r["sampler"] for r in R0}):
    d = {r["prompt_id"]: r for r in R0 if r["sampler"] == cfg}
    ps = sorted(d)
    print(f"{cfg} : {len(ps)} prompts, seed {d[ps[0]]['seed']}, {d[ps[0]]['config']}")
    print(f"   ir_max moyen {np.mean([d[p]['ir_max'] for p in ps]):.3f}   ir moyen des 4 {np.mean([d[p]['ir_mean'] for p in ps]):.3f}"
          f"   images distinctes {np.mean([d[p]['n_distinct_images'] for p in ps]):.2f}/4"
          f" (moins de 4 dans {np.mean([d[p]['n_distinct_images'] < 4 for p in ps]):.0%} des runs)"
          f"   div_pix {np.mean([d[p]['div_pix'] for p in ps]):.3f}   {np.mean([d[p]['seconds'] for p in ps]):.0f} s/run")
    print(f"   papier, SD v1.5 : k1 {PAPIER['k1']}, bon4 {PAPIER['bon4']}, FK {PAPIER['fk4']} (FK - bon4 = +0.161)")
    for nom, ref in (("bon4", bon), ("k1", k1), ("ctl", ctl)):
        com = [p for p in ps if p in ref]
        print(f"   {cfg} - {nom:4s} sur ir_max : {se([d[p]['ir_max'] - ref[p]['ir_max'] for p in com])}"
              f"   ({sum(d[p]['ir_max'] > ref[p]['ir_max'] for p in com)}/{len(com)} gagnes)")
    if R1:
        com = [p for p in ps if p in R1]
        if com:
            print(f"   {cfg} - R1   sur ir_max : {se([d[p]['ir_max'] - R1[p]['ir_max'] for p in com])}   (memes choix d'implementation, codes differents)")
    print()

if R1:
    ps = sorted(set(R1) & set(bon))
    x = np.array([R1[p]["ir_max"] - bon[p]["ir_max"] for p in ps])
    print(f"R1 - bon4 apparie, {len(ps)} prompts : {se(x)} ; contre ctl : {se([R1[p]['ir_max'] - ctl[p]['ir_max'] for p in ps if p in ctl])}")
    print(f"   regle : ecart au papier {'FERME' if x.mean() >= 0.12 else 'BORNE'} (seuil +0.12 ; papier +0.161 ; ctl - bon4 = "
          f"{np.mean([ctl[p]['ir_max'] - bon[p]['ir_max'] for p in ps if p in ctl]):+.3f} sur ces prompts)")
    print(f"   R1 : lignees {np.mean([R1[p]['n_lineages'] for p in ps]):.2f}, une racine dans {np.mean([R1[p]['n_lineages'] == 1 for p in ps]):.0%},"
          f" reech {np.mean([R1[p]['n_resamplings'] for p in ps]):.2f}, t=80 inerte dans "
          f"{np.mean([(np.ptp(R1[p]['logG_at_schedule'][0]) < 1e-9) for p in ps]):.0%}")
