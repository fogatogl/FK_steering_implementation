"""J. D'ou vient le gain de fk4 sur bon4, une fois la racine perdue ?

Trois quantites par prompt, toutes sur les memes quatre x_T :

  A = bon4.ir_max                   la meilleure des 4 racines, menee librement
  B = bon4.ir[racine gardee par fk] ce que la racine que fk garde aurait donne seule
  C = fk4.ir_max                    ce que fk en tire reellement

C - A est le gain publie. B - A est ce que coute l'effondrement au niveau de la
racine. C - B est ce que le pilotage ajoute le long de la trajectoire, a racine
fixee : la selection entre freres aux pas tardifs, la ou r_phi predit enfin.
"""
import json
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent.parent / "results"
load = lambda p: json.loads((R / p).read_text())["runs"]
bon = {r["prompt_id"]: r for r in load("sd_baseline.json")
       if r["sampler"] == "bon4" and r["seed"] == 2024}
fk = [r for r in load("sd_ref_fields100.json") if r["prompt_id"] in bon]
print(__doc__.splitlines()[0], "\n")

A = np.array([max(bon[r["prompt_id"]]["ir"]) for r in fk])
B = np.array([np.mean([bon[r["prompt_id"]]["ir"][j] for j in set(r["root_slots"])]) for r in fk])
C = np.array([r["ir_max"] for r in fk])
M = np.array([np.mean(bon[r["prompt_id"]]["ir"]) for r in fk])
se = lambda v: v.std(ddof=1) / len(v) ** .5
print(f"  {len(fk)} prompts, seed 2024\n")
print(f"  A  meilleure des 4 racines, libre        : {A.mean():+.4f}")
print(f"  M  racine moyenne, libre                 : {M.mean():+.4f}")
print(f"  B  racine gardee par fk, libre           : {B.mean():+.4f}")
print(f"  C  ce que fk en tire                     : {C.mean():+.4f}")
print(f"\n  B - M  la racine gardee vaut mieux que le hasard : {(B-M).mean():+.4f} +/- {se(B-M):.4f}")
print(f"  B - A  ce que l'effondrement coute sur la racine : {(B-A).mean():+.4f} +/- {se(B-A):.4f}")
print(f"  C - B  ce que le pilotage ajoute a racine fixee  : {(C-B).mean():+.4f} +/- {se(C-B):.4f}")
print(f"  C - A  le gain publie de fk4 sur bon4           : {(C-A).mean():+.4f} +/- {se(C-A):.4f}")
print(f"\n  Verification que la decomposition ferme : (B-A) + (C-B) = {((B-A)+(C-B)).mean():+.4f}, "
      f"C - A = {(C-A).mean():+.4f}")
print(f"\n  Lecture : fk4 part d'une racine {abs((B-A).mean()):.3f} moins bonne que celle que")
print(f"  best-of-4 choisit, et remonte {(C-B).mean():.3f} en pilotant. Le gain net de")
print(f"  {(C-A).mean():+.3f} n'est donc pas 'fk choisit une meilleure racine' : c'est")
print(f"  'fk choisit moins bien mais pilote', et l'effondrement est le prix du pilotage,")
print(f"  pas son moyen.")
