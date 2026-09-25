"""I. The surviving root, on 100 prompts instead of 20.

sd_ref_fields100.json does not carry r_at_schedule, but it carries root_slots: the
starting slot the four final images descend from. Paired with the per-slot ir
of bon4 (same prompt, same seed_effective, same x_T), it answers directly
"is the kept root better than a root drawn at random", with five times
more runs than finding 3.

docs/max_potential.md had already measured a wrong-root rate at the level of
chance on S60. What is new here: the RANK of the kept root and the cost in
reward, not only a binary rate, and on the reference file with 100
prompts.
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
mauvais = [r["prompt_id"] for r in fk if r["seed_effective"] != bon[r["prompt_id"]]["seed_effective"]]
print(f"  {len(fk)} paired runs, {len(mauvais)} seed_effective mismatches"
      + (" (ok)" if not mauvais else f" -> {mauvais[:3]}"))

rangs, pertes, alea, meilleur = [], [], [], []
for r in fk:
    ir = np.array(bon[r["prompt_id"]]["ir"])
    ordre = np.argsort(np.argsort(-ir))          # 0 = the best root
    for j in set(r["root_slots"]):               # one row per surviving root
        rangs.append(int(ordre[j]))
        pertes.append(float(ir.max() - ir[j]))
    alea.append(float(ir.max() - ir.mean()))
    meilleur.append(int(np.argmax(ir) in set(r["root_slots"])))

rangs, pertes, alea = np.array(rangs), np.array(pertes), np.array(alea)
n = len(fk)
print(f"\n  rank of the kept root among the 4 (0 = the best), {len(rangs)} roots:")
print(f"    mean {rangs.mean():.3f} +/- {rangs.std(ddof=1)/len(rangs)**.5:.3f}   "
      f"(1.500 by chance)")
print(f"    distribution: " + "  ".join(f"rank {k}: {(rangs == k).mean():.0%}" for k in range(4))
      + "   (25 % each by chance)")
lig = np.mean([len(set(r["root_slots"])) for r in fk])
print(f"\n  the best root survives in {np.mean(meilleur):.0%} of runs")
print(f"    chance level, given the number of surviving roots ({lig:.2f} per run): "
      f"{lig/4:.0%}")
print(f"\n  cost in ImageReward:")
print(f"    ir(best) - ir(kept root)   : {pertes.mean():+.4f} "
      f"+/- {pertes.std(ddof=1)/len(pertes)**.5:.4f}")
print(f"    ir(best) - ir(random root) : {alea.mean():+.4f} "
      f"+/- {alea.std(ddof=1)/len(alea)**.5:.4f}")
d = np.array([np.mean([bon[r["prompt_id"]]["ir"].__getitem__(j) for j in set(r["root_slots"])])
              - np.mean(bon[r["prompt_id"]]["ir"]) for r in fk])
print(f"    paired: ir(kept root) - ir(mean of the 4) = {d.mean():+.4f} "
      f"+/- {d.std(ddof=1)/len(d)**.5:.4f} over {n} prompts")
print(f"\n  (reading after the check below)")

# --- check: one observation per run, and a bootstrap over the prompts ---
# The runs with two surviving roots gave two rows above. One run = one
# observation, otherwise the least collapsed runs count twice.
print("\n  check, one observation per run:")
rang_run = np.array([np.mean([int(np.argsort(np.argsort(-np.array(bon[r["prompt_id"]]["ir"])))[j])
                              for j in set(r["root_slots"])]) for r in fk])
rng = np.random.default_rng(2024)
def boot(v, tirages=20000):
    t = v[rng.integers(0, len(v), size=(tirages, len(v)))].mean(1)
    return np.percentile(t, [2.5, 97.5])
lo, hi = boot(rang_run)
print(f"    mean rank per run: {rang_run.mean():.3f}  bootstrap 95 % CI [{lo:.3f}, {hi:.3f}]  "
      f"(1.500 by chance)")
lo, hi = boot(d)
print(f"    gain over the mean root: {d.mean():+.4f}  bootstrap 95 % CI [{lo:+.4f}, {hi:+.4f}]")
print(f"    share of the mean-root / best-root gap recovered: "
      f"{100*d.mean()/alea.mean():.0f} %")

# and the same calculation with the runs split by the number of surviving roots
for nl in sorted({len(set(r["root_slots"])) for r in fk}):
    sub = [r for r in fk if len(set(r["root_slots"])) == nl]
    dd = np.array([np.mean([bon[r["prompt_id"]]["ir"][j] for j in set(r["root_slots"])])
                   - np.mean(bon[r["prompt_id"]]["ir"]) for r in sub])
    print(f"    {len(sub):3d} runs with {nl} root(s): gain {dd.mean():+.4f} "
          f"+/- {dd.std(ddof=1)/max(len(dd),1)**.5:.4f}")

print("\n  Takeaway: the kept root is NOT drawn at random at n = 100. It recovers")
print("  about a quarter of the available gap, where best-of-4 recovers all of it")
print("  by construction. But root_slots is the product of ALL the resamplings,")
print("  not of the step t = 80 alone: in the runs where two roots survive at t = 80, it is")
print("  t = 60 that decides, with a guide reward already less noisy. Finding 3, which")
print("  isolates the step t = 80 alone, measures something else and stands as it is.")
