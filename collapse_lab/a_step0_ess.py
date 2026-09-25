"""A. Is the logged ESS exactly the one that lam * r_phi implies?

Bug detector: if the ESS recomputed from r_at_schedule[0] matches
ess_at_schedule[0], the collapse is arithmetic and not a defect of
weights.py / resampling.py. No import from smc/ is needed here: with an empty
gate the three potentials give logG = lam * r_t (smc/fk.py, _potential_terms).
"""
import json, math
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent.parent / "results"
load = lambda p: json.loads((R / p).read_text())["runs"]


def poids(logw):
    lw = np.asarray(logw, float)
    lw = lw - lw.max(-1, keepdims=True)
    w = np.exp(lw)
    return w / w.sum(-1, keepdims=True)


ess = lambda w: 1.0 / (w ** 2).sum(-1)

print(__doc__.splitlines()[0], "\n")

for tag, f in (("fk4_stat", "sd_variants/fk4_stat.json"),
               ("fk4_diff", "sd_variants/fk4_diff.json"),
               ("ref100", "sd_ref_fields100.json")):
    runs = load(f)
    avec = [r for r in runs if "r_at_schedule" in r]
    if not avec:
        print(f"{tag:9s} {len(runs):3d} runs, no r_at_schedule (file older than 21/09)")
        continue
    r0 = np.array([r["r_at_schedule"][0] for r in avec])
    lam = np.array([[r["lam"]] for r in avec])
    e = ess(poids(lam * r0))
    loggee = np.array([r["ess_at_schedule"][0] for r in avec])
    d = np.abs(e - loggee)
    print(f"{tag:9s} {len(avec):3d} runs with r_at_schedule: "
          f"|ESS_recalc - ESS_logged| median {np.median(d):.2e}  max {d.max():.2e}  "
          f"({(d < 1e-3).sum()}/{len(d)} under 1e-3)")

# --- scale, on the 40 runs that carry r_at_schedule ---
# One file only: fk4_diff carries the same 20 prompts at the same x_T (see commun.py).
avec = load("sd_variants/fk4_stat.json")
r = np.array([x["r_at_schedule"] for x in avec])          # (n, 5 rows, 4 slots)
ts = [80, 60, 40, 20, 0]
q = lambda v: f"{np.percentile(v, 25):6.3f} |{np.median(v):6.3f} |{np.percentile(v, 75):6.3f}"

print("\nScale of the guide reward r_phi(predict_x0), 20 runs, k=4   (Q1 | med | Q3)")
print(f"{'':14s}{'level':>22s}{'range max-min':>24s}{'gap 1st-2nd':>24s}"
      f"{'ESS at lam=10':>22s}")
for m, t in enumerate(ts):
    x = r[:, m, :]
    srt = np.sort(x, 1)
    e = ess(poids(10.0 * x))
    print(f"  t = {t:<9d}{q(x.ravel())}{q(x.max(1) - x.min(1)):>24s}"
          f"{q(srt[:, -1] - srt[:, -2]):>24s}{q(e):>22s}")

print("\n  logged ESS (median) per row:",
      "  ".join(f"t={t}:{np.median([x['ess_at_schedule'][m] for x in avec]):.2f}"
                for m, t in enumerate(ts)))
print("  max weight at lam=10 (median):",
      "  ".join(f"t={t}:{np.median(poids(10.0 * r[:, m, :]).max(1)):.3f}" for m, t in enumerate(ts)))
