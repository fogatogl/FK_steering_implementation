"""L. The target itself: what is the ESS of exp(lam * ir) over four free draws?

No sampler is involved here. We take the four images of bon4 (four
roots run freely), reweight them by exp(lam * ImageReward), and read
the ESS. This is the concentration of the target p(x0) exp(lam r(x0)) restricted to four
candidates: the diversity ceiling that four particles can carry at this lambda,
whatever the kernel, the potential or the schedule do.
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


print(__doc__.splitlines()[0], f"\n  {len(bon)} prompts, median range of the free final ir:",
      f"{np.median([np.ptp(r['ir']) for r in bon]):.2f}\n")
print(f"  {'lambda':>6s} | {'median ESS':>11s} | {'Q1 - Q3':>13s} | share < 1.5")
for lam in (0.5, 1, 2, 5, 10):
    e = np.array([ess(lam * np.array(r["ir"])) for r in bon])
    print(f"  {lam:6.1f} | {np.median(e):11.2f} | {np.percentile(e, 25):5.2f} - {np.percentile(e, 75):5.2f} | {(e < 1.5).mean():.0%}")

e80 = np.median([r["ess_at_schedule"][0] for r in fk])
print(f"\n  for comparison: median ESS of fk4 at the first scheduled step (t = 80), lam = 10: {e80:.2f}")
