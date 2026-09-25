"""D. The first step does not depend on the potential: max, difference and statistic coincide there.

With an empty gate (smc/fk.py, _potential_terms) the three branches give logG = lam * r_t.
fk4_stat (max/statistic) and fk4_diff (difference) share prompts and seeds, and
no resampling takes place before step t=80: their trajectories there are therefore
the SAME. If the rows 0 coincide bit for bit, the choice of potential cannot change
anything in the lineage, since that is where the lineage dies.
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
print(f"  {len(com)} common prompts")
print(f"  row 0 of r_phi identical           : max gap {np.abs(r0s - r0d).max():.2e}")
print(f"  ESS at step 0 identical            : max gap {np.abs(e0s - e0d).max():.2e}")
print(f"  root kept identical                : "
      f"{sum(set(stat[c]['root_slots']) == set(diff[c]['root_slots']) for c in com)}/{len(com)} prompts")

# and what the potentials change afterwards: nothing measurable on the reward
for tag, d in (("max/statistic", stat), ("difference", diff)):
    ir = np.array([d[c]["ir_max"] for c in com])
    lin = np.array([d[c]["n_lineages"] for c in com])
    nr = np.array([d[c]["n_resamplings"] for c in com])
    print(f"  {tag:14s} mean ir_max {ir.mean():+.4f}  lineages {lin.mean():.2f}  "
          f"resamplings {nr.mean():.2f}")
a = np.array([stat[c]["ir_max"] for c in com]) - np.array([diff[c]["ir_max"] for c in com])
print(f"  paired difference max - difference: {a.mean():+.4f} +/- {a.std(ddof=1)/len(a)**.5:.4f}")

# the following rows do diverge: the two runs are no longer the same run
for m, t in enumerate([80, 60, 40, 20, 0]):
    d = np.abs(np.array([stat[c]["r_at_schedule"][m] for c in com])
               - np.array([diff[c]["r_at_schedule"][m] for c in com])).max()
    print(f"    t={t:3d}: max gap between the two files on r_phi = {d:.3f}"
          + ("   <- identical" if d < 1e-6 else ""))

# the post's section 2: the increment form this repository uses against the paper's statistic
# form, on one machine (fk4_stat ran on the A2 at 89 s, like the 21/09 reference in increment form)
ref21 = {r["prompt_id"]: r for r in json.loads((Path(__file__).resolve().parent.parent / "results" / "sd_ref_fields100.json").read_text())["runs"]
         if r["sampler"] == "fk4" and r["seed"] == 2024}
d = np.array([r["ir_max"] - ref21[p]["ir_max"] for p, r in stat.items()])
print(f"\n  statistic form (fk4_stat) - increment form (21/09 reference), same machine: "
      f"{d.mean():+.4f} +/- {d.std(ddof=1) / len(d) ** .5:.4f} (n={len(d)})")

