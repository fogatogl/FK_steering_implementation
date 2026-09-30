"""Readout of session H. Pre-registration: docs/protocol_sd.md, session H and its amendment.

H1: appendix D's arms (probe_H.json) against session D's ctl and lam0, same A2 group, same x_T.
The primary readout is div_pix in the runs that end on one root, paired by prompt with ctl.
H2: the released code before its fix under the launcher's own seeding (sd_authors_once.json).
"""
import json
from pathlib import Path
import numpy as np
from commun import wilson

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "collapse_lab" / "out"
rng = np.random.default_rng(0)


def arms(p):
    d = {}
    for r in json.loads(p.read_text())["runs"] if p.exists() else []:
        d.setdefault(r["arm"], {})[r["prompt_id"]] = r
    return d


def se(x):
    x = np.asarray(x, float)
    return f"{x.mean():+.3f} +/- {x.std(ddof=1) / len(x) ** .5:.3f} (n={len(x)}, {int((x > 0).sum())} won)"


def boot(x, n=10000):
    x = np.asarray(x, float)
    m = rng.choice(x, (n, len(x))).mean(1)
    return f"[{np.quantile(m, .025):+.3f}, {np.quantile(m, .975):+.3f}]"


H, D = arms(OUT / "session_H" / "probe_H.json"), arms(OUT / "session_D" / "probe_D.json")
if H:
    ids = [pid for pid in D["ctl"] if all(pid in H.get(a, {}) for a in H)]
    A = {"ctl": D["ctl"], "lam0": D["lam0"], **H}
    print(f"H1, {len(ids)} prompts done in every arm ({', '.join(H)})\n")
    for a, runs in A.items():
        R = [runs[pid] for pid in ids]
        one = [r["n_lineages"] == 1 for r in R]
        dp1 = [r["div_pix"] for r, o in zip(R, one) if o]
        print(f"{a:8s} roots {np.mean([r['n_lineages'] for r in R]):.2f}, one root {wilson(sum(one), len(R))}, "
              f"div_pix {np.mean([r['div_pix'] for r in R]):.3f}, "
              f"one-root {np.mean(dp1) if dp1 else float('nan'):.3f}, first ESS {np.median([r['ess_at_schedule'][0] for r in R]):.2f}, "
              f"ir_max {np.mean([r['ir_max'] for r in R]):.3f}, mean of four {np.mean([np.mean(r['ir']) for r in R]):.3f}, "
              f"{np.median([r['seconds'] for r in R]):.0f} s")
    print("\ndiv_pix in one-root runs, arm minus ctl, on the prompts where both end on one root:")
    for a in H:
        if a == "free200":
            continue
        d = [A[a][p]["div_pix"] - A["ctl"][p]["div_pix"] for p in ids
             if A[a][p]["n_lineages"] == 1 and A["ctl"][p]["n_lineages"] == 1]
        if len(d) > 1:
            print(f"  {a:8s} {se(d)} {boot(d)}")
    base = {"d200": "free200", "st200": "free200", "pos100": "lam0", "ctl": "lam0", "recipeD": "free200"}
    g = {a: np.array([A[a][p]["ir_max"] - A[b][p]["ir_max"] for p in ids]) for a, b in base.items() if a in A and b in A}
    print("\nir_max over the free sampler at the same step count:")
    for a, x in g.items():
        print(f"  {a:8s} - {base[a]:8s} {se(x)}")
    if "d200" in g:
        x = g["d200"] - g["ctl"]
        m, s = x.mean(), x.std(ddof=1) / len(x) ** .5
        d = [A["d200"][p]["div_pix"] - A["ctl"][p]["div_pix"] for p in ids
             if A["d200"][p]["n_lineages"] == 1 and A["ctl"][p]["n_lineages"] == 1]
        dp1 = np.mean([A["d200"][p]["div_pix"] for p in ids if A["d200"][p]["n_lineages"] == 1])
        bar = np.mean([A["ctl"][p]["div_pix"] for p in ids if A["ctl"][p]["n_lineages"] == 1]) + 0.05
        lo = np.quantile(rng.choice(np.asarray(d), (10000, len(d))).mean(1), .025) if len(d) > 1 else float("nan")
        print(f"\ndecision (amendment of 07h00): d200 one-root div_pix {dp1:.3f} (>= ctl + 0.05 = {bar:.3f}), "
              f"lower bound of d200 - ctl {lo:+.3f} (> 0), gain of d200 minus gain of ctl {m:+.3f} +/- {s:.3f} "
              f"(>= -1 se): {'holds' if dp1 >= bar and lo > 0 and m >= -s else 'does not hold'}")

# H1-bis (amendment of 17h50): the same four arms on prompts 41 to 100, read with H1's 40 by the rule of
# 07h00; printed only once the 240 runs are on disk
B = arms(OUT / "session_H" / "probe_H1bis.json")
if H and sum(len(v) for v in B.values()) == 240:
    four = ("ctl", "lam0", "free200", "d200")
    A = {a: {**H[a], **B[a]} for a in four}
    ids = [pid for pid in A["ctl"] if all(pid in A[a] for a in four)]
    print(f"\nH1-bis with H1, {len(ids)} prompts in every arm ({', '.join(four)})")
    for a in four:
        R = [A[a][pid] for pid in ids]
        one = [r["n_lineages"] == 1 for r in R]
        dp1 = [r["div_pix"] for r, o in zip(R, one) if o]
        print(f"  {a:8s} one root {wilson(sum(one), len(R))}, div_pix {np.mean([r['div_pix'] for r in R]):.3f}, "
              f"one-root {np.mean(dp1) if dp1 else float('nan'):.3f}, first ESS {np.median([r['ess_at_schedule'][0] for r in R]):.2f}, "
              f"ir_max {np.mean([r['ir_max'] for r in R]):.3f}, mean of four {np.mean([np.mean(r['ir']) for r in R]):.3f}")
    d = [A["d200"][p]["div_pix"] - A["ctl"][p]["div_pix"] for p in ids
         if A["d200"][p]["n_lineages"] == 1 and A["ctl"][p]["n_lineages"] == 1]
    print(f"  div_pix in one-root runs, d200 minus ctl: {se(d)} {boot(d)}")
    g1 = np.array([A["d200"][p]["ir_max"] - A["free200"][p]["ir_max"] for p in ids])
    g0 = np.array([A["ctl"][p]["ir_max"] - A["lam0"][p]["ir_max"] for p in ids])
    print(f"  ir_max, d200 minus free200 {se(g1)}; ctl minus lam0 {se(g0)}")
    x = g1 - g0
    m, s = x.mean(), x.std(ddof=1) / len(x) ** .5
    dp1 = np.mean([A["d200"][p]["div_pix"] for p in ids if A["d200"][p]["n_lineages"] == 1])
    bar = np.mean([A["ctl"][p]["div_pix"] for p in ids if A["ctl"][p]["n_lineages"] == 1]) + 0.05
    lo = np.quantile(rng.choice(np.asarray(d), (10000, len(d))).mean(1), .025)
    print(f"  decision on {len(ids)} prompts: d200 one-root div_pix {dp1:.3f} (>= ctl + 0.05 = {bar:.3f}), lower bound "
          f"of d200 - ctl {lo:+.3f} (> 0), gain of d200 minus gain of ctl {m:+.3f} +/- {s:.3f} (>= -1 se): "
          f"{'holds' if dp1 >= bar and lo > 0 and m >= -s else 'does not hold'}")
    print(f"  mean of the four, d200 minus ctl, paired: {se([np.mean(A['d200'][p]['ir']) - np.mean(A['ctl'][p]['ir']) for p in ids])}")
    # the 100 include the screen's 40: the same rule on the 60 new prompts alone
    new = [pid for pid in ids if pid in B["ctl"]]
    d = [A["d200"][p]["div_pix"] - A["ctl"][p]["div_pix"] for p in new
         if A["d200"][p]["n_lineages"] == 1 and A["ctl"][p]["n_lineages"] == 1]
    x = np.array([(A["d200"][p]["ir_max"] - A["free200"][p]["ir_max"]) - (A["ctl"][p]["ir_max"] - A["lam0"][p]["ir_max"]) for p in new])
    m, s = x.mean(), x.std(ddof=1) / len(x) ** .5
    dp1 = np.mean([A["d200"][p]["div_pix"] for p in new if A["d200"][p]["n_lineages"] == 1])
    bar = np.mean([A["ctl"][p]["div_pix"] for p in new if A["ctl"][p]["n_lineages"] == 1]) + 0.05
    lo = np.quantile(rng.choice(np.asarray(d), (10000, len(d))).mean(1), .025)
    print(f"  the {len(new)} new prompts alone: d200 one-root div_pix {dp1:.3f} (ctl + 0.05 = {bar:.3f}), d200 - ctl {se(d)}, "
          f"lower bound {lo:+.3f}, gain difference {m:+.3f} +/- {s:.3f}: "
          f"{'holds' if dp1 >= bar and lo > 0 and m >= -s else 'does not hold'}")

once = ROOT / "results" / "sd_authors_once.json"
if once.exists():
    runs = json.loads(once.read_text())["runs"]
    P = {}
    for r in runs:
        P.setdefault((r["sampler"], r["seed"]), {})[r["prompt_id"]] = r
    print("\nH2, the released code before its fix, the launcher's seeding")
    for (s, seed), d in sorted(P.items()):
        print(f"  {s:28s} seed {seed}: n={len(d)}, ir_max {np.mean([r['ir_max'] for r in d.values()]):.3f}, "
              f"{np.median([r['seconds'] for r in d.values()]):.0f} s, {d[next(iter(d))]['versions']['diffusers']}")
    fk, bo = "authors_paper_once_prefix", "authors_free_once_prefix"
    seeds = [s for s in (42, 43, 44) if len(P.get((fk, s), {})) == 100 and len(P.get((bo, s), {})) == 100]
    if seeds:
        ids = list(P[(fk, seeds[0])])
        x = [np.mean([P[(fk, s)][p]["ir_max"] - P[(bo, s)][p]["ir_max"] for s in seeds]) for p in ids]
        print(f"  FK minus best-of-4 on seeds {seeds}, each prompt averaged over the passes: {se(x)}, paper +0.161")
        print(f"  FK pass means {[round(float(np.mean([r['ir_max'] for r in P[(fk, s)].values()])), 3) for s in seeds]}")
        for s in seeds:
            print(f"  seed {s}, FK minus best-of-4: {se([P[(fk, s)][p]['ir_max'] - P[(bo, s)][p]['ir_max'] for p in ids])}")
        m, s = np.mean(x), np.std(x, ddof=1) / len(x) ** .5
        print(f"  three-pass means: best-of-4 {np.mean([r['ir_max'] for t in seeds for r in P[(bo, t)].values()]):.3f}, "
              f"FK {np.mean([r['ir_max'] for t in seeds for r in P[(fk, t)].values()]):.3f}; "
              f"the paper's +0.161 is {(0.161 - m) / s:.1f} standard errors above")
        for name, smp in (("FK", fk), ("best-of-4", bo)):
            R = [r for t in seeds for r in P[(smp, t)].values()]
            print(f"  {name}: div_pix {np.mean([r['div_pix'] for r in R]):.3f}, fewer than four distinct images in "
                  f"{sum(r['n_distinct_images'] < 4 for r in R)} of {len(R)} runs, mean of the four {np.mean([r['ir_mean'] for r in R]):.3f}")
