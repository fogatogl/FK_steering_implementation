"""H. What can change the ESS, and what cannot.

The weight is exp(lambda * r), normalised. Any transformation of r that adds the
SAME amount to every particle vanishes in the normalisation: centring the
reward, subtracting its mean, subtracting the shared running max, subtracting the
reward of the previous step when the particles are clones; none of this
moves the ESS in the slightest. Only the operations that compress the GAPS
between particles count: lowering lambda, or creating ties as the floor does.

This is checked in three lines, and it tells in advance which leads are dead.
"""
import numpy as np
from smc.weights import normalize_logw, ess
import torch, json
from pathlib import Path

r = torch.tensor([-2.2531, -0.4344, -0.6896, -0.8958])
LAM = 10.0
base = ess(normalize_logw(LAM * r)[0])
print(__doc__.splitlines()[0], "\n")
print(f"  r = {r.tolist()}, lambda = {LAM}")
print(f"  ESS as is                                 : {base:.4f}")
for nom, rr in (("centred (r - mean)", r - r.mean()),
                ("shifted by +100", r + 100),
                ("minus the shared running max (0.5)", r - 0.5),
                ("normalised by the standard deviation", (r - r.mean()) / r.std()),
                ("floored at 0 (reward_min_value)", torch.clamp(r, min=0.0)),
                ("lambda divided by 5", r * 0.2)):
    print(f"  {nom:42s}: {ess(normalize_logw(LAM * rr)[0]):.4f}")
print("\n  The first three transformed lines equal the first line bit for bit: a")
print("  shared shift does nothing. Normalising by the standard deviation does change")
print("  the scale and hence the ESS, but it also changes the target exp(lambda * r(x_0)).")

# --- the lambda that would hold the ESS at k/2, step by step, from the actual runs ---
from smc.fk import bisect_lambda
R = Path(__file__).resolve().parent.parent / "results"
runs = json.loads((R / "sd_variants" / "fk4_stat.json").read_text())["runs"]
print(f"\n  The lambda_t that would hold the ESS at k/2 = 2, computed by smc.fk.bisect_lambda on")
print(f"  the r_phi actually recorded ({len(runs)} runs):")
for m, t in enumerate([80, 60, 40, 20, 0]):
    lams = []
    for run in runs:
        rt = torch.tensor(run["r_at_schedule"][m])
        lams.append(bisect_lambda(torch.zeros(4), rt, torch.zeros(4), 2.0, 100.0, 10.0))
    lams = np.array(lams)
    print(f"    t = {t:3d}: median {np.median(lams):6.2f}   Q1 {np.percentile(lams,25):6.2f}  "
          f"Q3 {np.percentile(lams,75):6.2f}   (the run uses 10.0 throughout)")
print("  Reading: the lambda that would keep two particles alive is about 2 at the first")
print("  step and rises afterwards, because the range of r_phi narrows. The run does")
print("  the opposite: lambda constant at 10, hence maximal pressure where the range")
print("  is widest and the signal weakest.")
