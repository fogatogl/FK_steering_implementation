"""J. Where does the gain of fk4 over bon4 come from, once the root is lost?

Three quantities per prompt, all on the same four x_T:

  A = bon4.ir_max                   the best of the 4 roots, run freely
  B = bon4.ir[root kept by fk]      what the root that fk keeps would have given alone
  C = fk4.ir_max                    what fk actually gets out of it

C - A is the published gain. B - A is what the collapse costs at the level of the
root. C - B is what the steering adds along the trajectory, with the root
fixed: the selection among siblings at the late steps, where r_phi finally predicts.
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
print(f"  A  best of the 4 roots, free             : {A.mean():+.4f}")
print(f"  M  mean root, free                       : {M.mean():+.4f}")
print(f"  B  root kept by fk, free                 : {B.mean():+.4f}")
print(f"  C  what fk gets out of it                : {C.mean():+.4f}")
print(f"\n  B - M  the kept root is better than chance        : {(B-M).mean():+.4f} +/- {se(B-M):.4f}")
print(f"  B - A  what the collapse costs on the root        : {(B-A).mean():+.4f} +/- {se(B-A):.4f}")
print(f"  C - B  what the steering adds with the root fixed : {(C-B).mean():+.4f} +/- {se(C-B):.4f}")
print(f"  C - A  the published gain of fk4 over bon4        : {(C-A).mean():+.4f} +/- {se(C-A):.4f}")
print(f"\n  Check that the decomposition closes: (B-A) + (C-B) = {((B-A)+(C-B)).mean():+.4f}, "
      f"C - A = {(C-A).mean():+.4f}")
print(f"\n  Reading: fk4 starts from a root {abs((B-A).mean()):.3f} worse than the one that")
print(f"  best-of-4 picks, and climbs back {(C-B).mean():.3f} by steering. The net gain of")
print(f"  {(C-A).mean():+.3f} is therefore not 'fk picks a better root': it is")
print(f"  'fk picks worse but steers', and the collapse is the price of the steering,")
print(f"  not its means.")
