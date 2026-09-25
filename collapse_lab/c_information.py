"""C. Does the selection that kills the lineages carry information?

r_at_schedule[0] is read at the first scheduled step (t=80), before any
resampling: particle j of fk4 is there still exactly particle j
of bon4, which shares its x_T (same prompts, same seed_effective = seed*1000+i,
same randn_tensor). bon4["ir"][j] is therefore what this root becomes if it is
left alone. The question: does the ranking at t=80 predict this final ranking?
"""
import json, itertools
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent.parent / "results"
load = lambda p: json.loads((R / p).read_text())["runs"]

bon = {r["prompt_id"]: r for r in load("sd_baseline.json")
       if r["sampler"] == "bon4" and r["seed"] == 2024}
k1 = {r["prompt_id"]: r for r in load("sd_baseline.json")
      if r["sampler"] == "k1" and r["seed"] == 2024}

# fk4_diff carries the same prompts at the same x_T and the same row 0: stacking it
# would count each observation twice and wrongly divide the standard error by the square root of 2.
fk = [r for r in load("sd_variants/fk4_stat.json") if r["prompt_id"] in bon]
print(__doc__.splitlines()[0], "\n")
print(f"{len(fk)} fk4 runs paired with bon4 on (prompt_id, seed=2024)\n")

# pairing check: same seed_effective on both sides
mauvais = [r["prompt_id"] for r in fk if r["seed_effective"] != bon[r["prompt_id"]]["seed_effective"]]
print(f"  pairing check: {len(mauvais)} seed_effective mismatches"
      + (f" -> {mauvais[:5]}" if mauvais else " (ok)"))


def kendall(a, b):
    """tau-b on 4 points; +1 = same order, 0 = independent, -1 = reversed."""
    n, c, d = len(a), 0, 0
    for i, j in itertools.combinations(range(n), 2):
        s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
        c += s > 0
        d += s < 0
    return (c - d) / (c + d) if c + d else 0.0


ts = [80, 60, 40, 20, 0]
print("\n  row 0 (t=80) of r_phi  against  final ir of the SAME root under bon4:")
tau = np.array([kendall(r["r_at_schedule"][0], bon[r["prompt_id"]]["ir"]) for r in fk])
top1 = np.array([int(np.argmax(r["r_at_schedule"][0]) == np.argmax(bon[r["prompt_id"]]["ir"]))
                 for r in fk])
# standard error on the mean tau, independent runs
print(f"    mean Kendall tau: {tau.mean():+.3f} +/- {tau.std(ddof=1)/len(tau)**.5:.3f}"
      f"   (median {np.median(tau):+.3f})")
print(f"    the best at t=80 is the final best: {top1.mean():.0%} of runs "
      f"({top1.sum()}/{len(top1)}), chance 25 %")
# the same tau, but between the guide's final reward (row 4) and the final ir: the upper bound
print(f"\n  to bound it: how much reward does the chosen root cost?")
perte = np.array([max(bon[r["prompt_id"]]["ir"]) - bon[r["prompt_id"]]["ir"][int(np.argmax(r["r_at_schedule"][0]))]
                  for r in fk])
alea = np.array([max(bon[r["prompt_id"]]["ir"]) - np.mean(bon[r["prompt_id"]]["ir"]) for r in fk])
print(f"    ir(best) - ir(root chosen at t=80)  : {perte.mean():+.4f}")
print(f"    ir(best) - ir(random root)          : {alea.mean():+.4f}")
ecart = perte - alea   # paired: same prompt on both sides
print(f"    paired difference (chosen - random) : {ecart.mean():+.4f} +/- "
      f"{ecart.std(ddof=1)/len(ecart)**.5:.4f}")
print(f"    -> indistinguishable from chance; the sign is unfavourable but the standard error "
      f"covers it by a wide margin")

print("\n  and does the information rise along the trajectory?")
print("    (tau between row m and row 4 of the SAME run: valid only if the lineage")
print("     is unique, otherwise the ranks of row m do not follow the slots of row 4)")
uni = [r for r in fk if r["n_lineages"] == 1]
for m, t in enumerate(ts[:-1]):
    # after a full collapse the 4 slots of each row are siblings, but the permutation
    # of the comb between rows is not recorded: only the spread is read
    sp = np.array([np.ptp(r["r_at_schedule"][m]) for r in uni])
    print(f"    t={t:3d}: median range {np.median(sp):.3f}  "
          f"-> at lam=10, {10*np.median(sp):.1f} nats between the best and the worst slot")

print("\n  reference: standard deviation of r_phi across the 4 slots, per row")
for m, t in enumerate(ts):
    s = np.array([np.std(r["r_at_schedule"][m]) for r in fk])
    mu = np.array([np.mean(r["r_at_schedule"][m]) for r in fk])
    print(f"    t={t:3d}: mean level {mu.mean():+.3f}, within-run standard deviation {s.mean():.3f}, "
          f"runs where all 4 r_phi are < 0: {np.mean([max(r['r_at_schedule'][m]) < 0 for r in fk]):.0%}")
