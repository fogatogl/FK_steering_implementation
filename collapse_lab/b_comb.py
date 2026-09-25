"""B. How many distinct ancestors does the systematic comb leave alive?

Reproduces smc/resampling.py:resample_systematic exactly (regular comb,
u ~ U(0, 1/k)) on the weights recomputed from r_at_schedule, without importing smc/.
The number of copies of particle j is the floor or ceil of k*w_j: for a max
weight of 0.85 at k=4, the head takes 3 or 4 slots and at most 2 lineages remain.
"""
import json
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent.parent / "results"
load = lambda p: json.loads((R / p).read_text())["runs"]
poids = lambda lw: np.exp(lw - lw.max(-1, keepdims=True)) / np.exp(lw - lw.max(-1, keepdims=True)).sum(-1, keepdims=True)


def survivants_exact(w, k=4):
    """Expectation and distribution of the number of distinct ancestors under the comb, u integrated out.

    Point i of the comb is u + i/k, u ~ U(0,1/k); particle j takes
    point i iff cum_{j-1} <= u + i/k < cum_j. j survives iff at least one point
    falls on it. The integral is exact, by splitting [0,1/k] at the breakpoints.
    """
    cum = np.cumsum(w)
    bornes = sorted({0.0, 1.0 / k} | {c - i / k for c in cum[:-1] for i in range(k)
                                      if 0.0 < c - i / k < 1.0 / k})
    loi = {}
    for a, b in zip(bornes[:-1], bornes[1:]):
        u = 0.5 * (a + b)
        idx = np.searchsorted(cum, u + np.arange(k) / k, side="left")
        loi[len(set(idx.tolist()))] = loi.get(len(set(idx.tolist())), 0.0) + (b - a) * k
    return loi


runs = load("sd_variants/fk4_stat.json")   # one file only: see commun.py
r0 = np.array([r["r_at_schedule"][0] for r in runs])
w0 = poids(10.0 * r0)

lois = [survivants_exact(w) for w in w0]
esp = np.array([sum(n * p for n, p in l.items()) for l in lois])
p1 = np.array([l.get(1, 0.0) for l in lois])
print(__doc__.splitlines()[0], "\n")
print(f"At the first scheduled step ALONE (t=80), over {len(runs)} runs, lam=10:")
print(f"  expected number of distinct ancestors: {esp.mean():.3f}  (median {np.median(esp):.3f})")
print(f"  P(a single lineage from this step on): {p1.mean():.3f}")
for n in (1, 2, 3, 4):
    print(f"    P({n} ancestors) mean: {np.mean([l.get(n, 0.0) for l in lois]):.3f}")

# chain: a bound on the final number of lineages.
obs = np.array([r["n_lineages"] for r in runs])
print(f"\n  n_lineages observed at the end of the run: mean {obs.mean():.3f}, "
      f"{(obs == 1).sum()}/{len(obs)} with a single lineage")
print(f"  (the number of lineages can only decrease: 4 resamplings, "
      f"mean n_resamplings {np.mean([r['n_resamplings'] for r in runs]):.2f})")

# S80: a single selection point -> tests the prediction for the first step in isolation
s80 = load("sd_variants/S80.json")
o80 = np.array([r["n_lineages"] for r in s80])
print(f"\nS80 control (schedule [0, 80]: ONE resampling only, at t=80):")
print(f"  n_lineages observed: mean {o80.mean():.3f}, distribution "
      f"{ {n: int((o80 == n).sum()) for n in sorted(set(o80.tolist()))} } over {len(o80)} runs")
print(f"  predicted by the comb at step t=80: {esp.mean():.3f} expected ancestors")
print(f"  median ESS at t=80 on S80: {np.median([r['ess_at_schedule'][0] for r in s80]):.2f}")

# --- static counterfactual: the same comb at other lambdas, and under the floor ---
print("\nThe same first step, at other lambdas (and under the authors' floor):")
print(f"  {'regime':22s} {'median ESS':>12s} {'expected ancestors':>19s} {'P(1 lineage)':>12s} "
      f"{'inert step':>11s}")
for nom, lam, plancher in (("lam=10 (reference)", 10.0, False), ("lam=5", 5.0, False),
                           ("lam=2", 2.0, False), ("lam=1", 1.0, False),
                           ("lam=10 + floor 0", 10.0, True), ("lam=2 + floor 0", 2.0, True)):
    rr = np.maximum(r0, 0.0) if plancher else r0
    w = poids(lam * rr)
    e = 1.0 / (w ** 2).sum(-1)
    l = [survivants_exact(x) for x in w]
    inerte = (rr.max(1) - rr.min(1)) < 1e-12
    print(f"  {nom:22s} {np.median(e):12.2f} {np.mean([sum(n*p for n,p in x.items()) for x in l]):19.3f} "
          f"{np.mean([x.get(1,0.0) for x in l]):12.3f} {inerte.mean():11.0%}")
print("  (inert step = the four logG are equal, so ESS = k exactly: at threshold 1.0")
print("   smc/weights.py:should_resample tests ESS < threshold*k strictly, so it does not")
print("   resample and the four lineages survive intact)")
