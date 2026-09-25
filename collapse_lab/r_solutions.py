"""R. Do the corrections solve the collapse? The three criteria of ASSESSMENT.md.

Reads out/probe.json (every arm, unequal n, a cut queue is tolerated) and results/. Everything
is paired by prompt on the intersection of prompt_id. The criteria:
  1. the mechanism: share of prompts with a single root, lineages per step (aligned on the
     value of t, the `late` arm having no t = 80), root-weighted ESS at the end;
  2. the ceiling of the target: that ESS against the ESS of exp(lam * ir) over the four free
     draws of bon4 on the same prompts (l_target_ess.py, restricted to the arm's prompts);
  3. the price: ir_max, mean ir, weighted ir against ctl and against bon4, bootstrap 95 % CI.
"""
import json
from pathlib import Path
import numpy as np

from commun import bon4, groupe, load, references, wilson

LAB = Path(__file__).resolve().parent
runs = json.loads((LAB / "out" / "probe.json").read_text())["runs"]
par = {}
for r in runs:
    par.setdefault(r["arm"], {})[r["prompt_id"]] = r
bon = bon4()
fk4 = {r["prompt_id"]: r for r in load("sd_ref_fields100.json")
       if r["sampler"] == "fk4" and r["seed"] == 2024}
ORDRE = [a for a in ("ctl", "late", "adapt", "floor", "lam2", "fadapt", "floor2", "thr05", "rise",
                     "stat0", "multi", "vae", "idx", "R1", "lam0") if a in par]
rng = np.random.default_rng(0)
# six ctl records are restored from sd_ref_fields100.json without logG or lineages_trace
# (ASSESSMENT.md, incident of 23/09): everything that reads the weights is computed on the complete records
complet = lambda r: "logG_at_schedule" in r


def ess(logw):
    w = np.exp(logw - logw.max())
    w /= w.sum()
    return 1 / (w ** 2).sum()


def poids_finaux(r):
    """logW at the end, rebuilt: at threshold 1.0 it restarts from zero at every non-terminal
    step where logG has any spread (ESS < k in the strict sense); below 1.0, where ESS < threshold * k."""
    lg = np.array(r["logG_at_schedule"])
    e = np.array(r["ess_at_schedule"])
    k = r["k"]
    logW = np.zeros(k)
    for m in range(len(lg)):
        logW = logW + lg[m]
        if m == len(lg) - 1:
            break
        if r["threshold"] >= 1.0:
            resampled = (lg[m].max() - lg[m].min()) > 1e-9
        else:
            resampled = e[m] < r["threshold"] * k
        if resampled:
            logW = np.zeros(k)
    w = np.exp(logW - logW.max())
    return w / w.sum()


def ess_racines(r):
    w = poids_finaux(r)
    roots = np.array(r["root_slots"])
    W = np.array([w[roots == u].sum() for u in np.unique(roots)])
    return 1 / (W ** 2).sum()


def par_t(r):
    """Lineages after each scheduled step, indexed by the value of t."""
    ts = sorted(r["schedule_t"], reverse=True)
    return dict(zip(ts, r["lineages_trace"]))


def inertes(r):
    lg = np.array(r["logG_at_schedule"])
    ts = sorted(r["schedule_t"], reverse=True)
    return dict(zip(ts, (lg.max(1) - lg.min(1)) < 1e-9))


def boot(d, n=4000):
    d = np.asarray(d, dtype=float)
    if len(d) < 2:
        return d.mean(), np.nan, np.nan
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def fmt(m, lo, hi):
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"


print(__doc__.splitlines()[0])
print("  arms present: " + ", ".join(f"{a} ({len(par[a])})" for a in ORDRE), "\n")

# --- 0. checks ---------------------------------------------------------------
print("0. checks")
if "ctl" in par:
    com = sorted(set(par["ctl"]) & set(fk4))
    d = np.abs([par["ctl"][c]["ir_max"] - fk4[c]["ir_max"] for c in com])
    print(f"   ctl against the fk4 of sd_ref_fields100.json, {len(com)} prompts: max gap {d.max():.4f}")
pire_e, pire_w = 0.0, 0.0
for a in ORDRE:
    for r in filter(complet, par[a].values()):
        lg = np.array(r["logG_at_schedule"])
        e_rec = np.array([ess(row) for row in lg])
        pire_e = max(pire_e, float(np.abs(e_rec - np.array(r["ess_at_schedule"])).max())) if r["threshold"] >= 1 else pire_e
        pire_w = max(pire_w, abs(1 / (poids_finaux(r) ** 2).sum() - r["ess_at_schedule"][-1]))
print(f"   ESS per step recomputed from logG (threshold 1.0): max gap {pire_e:.1e} over {len(runs)} runs")
print(f"   final weights rebuilt, final ESS against ess_at_schedule[-1]: max gap {pire_w:.1e}")
if "late" in par:
    r = next(iter(par["late"].values()))
    print(f"   late: schedule {sorted(r['schedule_t'])}, {len(r['lineages_trace'])} scheduled steps, "
          f"{r['n_resamplings']} resamplings on the first run")

# --- 1. the mechanism --------------------------------------------------------
TS = [80, 60, 40, 20, 0]
print(f"\n1. the mechanism: where the lineages die, and what remains once weighted")
print(f"   {'arm':7s} {'n':>3s} {'1 root':>9s} {'>=3':>5s} | " + " ".join(f"t={t:<4d}" for t in TS)
      + f" | {'wtd ESS':>9s} {'inert t=80/60/40':>19s} {'div_pix':>8s}")
stats = {}
for a in ORDRE:
    d = par[a]
    ps = sorted(d)
    pc = [c for c in ps if complet(d[c])]
    tr = [par_t(d[c]) for c in pc]
    ine = [inertes(d[c]) for c in pc]
    un = np.mean([d[c]["n_lineages"] == 1 for c in ps])
    trois = np.mean([d[c]["n_lineages"] >= 3 for c in ps])
    er = np.array([ess_racines(d[c]) for c in pc])
    stats[a] = dict(un=un, er=er, lin=np.array([d[c]["n_lineages"] for c in ps]))
    cols = " ".join(f"{np.mean([x[t] for x in tr]):<6.2f}" if t in tr[0] else f"{'-':<6s}" for t in TS)
    ins = "/".join(f"{np.mean([x[t] for x in ine]):.0%}" if t in ine[0] else "-" for t in (80, 60, 40))
    print(f"   {a:7s} {len(ps):3d} {un:9.0%} {trois:5.0%} | {cols} | {er.mean():9.2f} {ins:>19s} "
          f"{np.mean([d[c]['div_pix'] for c in ps]):8.3f}")
print("   `1 root`: share of prompts whose four final images descend from the same x_T.")
print("   `wtd ESS`: 1 / sum_r W_r^2, W_r = total final weight of the slots that descend from root r.")

# --- 2. the ceiling of the target -------------------------------------------
print(f"\n2. the ceiling of the target: ESS of exp(lam * ir) over the four free draws (bon4) of the"
      f" same prompts,\n   against the root-weighted ESS the arm returns, and its raw lineages")
print(f"   {'arm':7s} {'lam':>5s} {'n':>3s} | {'ceiling mean':>12s} {'ceiling med':>11s} | {'wtd ESS':>9s} {'lineages':>8s} | ratio wtd/ceiling")
for a in ORDRE:
    d = par[a]
    ps = sorted(set(d) & set(bon))
    lam = d[ps[0]]["lam"]
    pl = np.array([ess(lam * np.array(bon[c]["ir"])) for c in ps])
    er = np.array([ess_racines(d[c]) for c in ps if complet(d[c])])
    lin = np.array([d[c]["n_lineages"] for c in ps])
    print(f"   {a:7s} {lam:5.1f} {len(ps):3d} | {pl.mean():12.2f} {np.median(pl):11.2f} | {er.mean():9.2f} {lin.mean():8.2f} | {er.mean() / pl.mean():.2f}")
print("   ratio ~ 1: the arm returns the diversity its target carries, no more, no less. Above 1 at lam = 10:")
print("   it returns roots the target crushes (visible if the weights are ignored, which ir_max and div_pix do).")

# --- 3. the price ------------------------------------------------------------
print(f"\n3. the price: paired differences per prompt, mean and bootstrap 95 % CI")


def ir_moy(r):
    return float(np.mean(r["ir"]))


def ir_pond(r):
    return float((poids_finaux(r) * np.array(r["ir"])).sum())


# every arm is paired with the ctl run of its own machine group, prompt by prompt (finding 19:
# across two GPU models the same x_T gives other images, so a slot or root comparison across
# groups measures the machine). REFS[g][c] is the reference of prompt c in group g.
REFS = references()
ref = {c: r for c, r in par.get("ctl", {}).items()}


def ref_de(r):
    return REFS[groupe(r)].get(r["prompt_id"])


print(f"   against ctl of the same machine group (fast: session C; A2: session A, then D)")
print(f"   {'arm':7s} {'n':>3s} {'A2/fast':>7s} | {'ir_max':>26s} | {'mean ir':>26s} | {'weighted ir':>26s} | wins | same roots")
for a in ORDRE:
    if a == "ctl":
        continue
    com = sorted(c for c in par[a] if ref_de(par[a][c]) is not None)
    g = [groupe(par[a][c]) for c in com]
    dm = [par[a][c]["ir_max"] - ref_de(par[a][c])["ir_max"] for c in com]
    da = [ir_moy(par[a][c]) - ir_moy(ref_de(par[a][c])) for c in com]
    dw = [ir_pond(par[a][c]) - ir_pond(ref_de(par[a][c])) for c in com
          if complet(par[a][c]) and complet(ref_de(par[a][c]))]
    same = np.mean([set(par[a][c]["root_slots"]) == set(ref_de(par[a][c])["root_slots"]) for c in com
                    if "root_slots" in par[a][c] and "root_slots" in ref_de(par[a][c])])
    print(f"   {a:7s} {len(com):3d} {g.count('A2'):3d}/{g.count('fast'):<3d} | {fmt(*boot(dm)):>26s} | {fmt(*boot(da)):>26s} | "
          f"{fmt(*boot(dw)):>26s} | {int(np.sum(np.array(dm) > 0))}/{len(com)} | {same:.0%}")
print("   same roots: share of prompts where the arm keeps the same set of x_T roots as its reference.")
# the proportions the post quotes from these tables, with their interval
for a in ("vae",):
    if a in par:
        pairs = [c for c in par[a] if ref_de(par[a][c]) is not None and "root_slots" in par[a][c]]
        k = sum(set(par[a][c]["root_slots"]) == set(ref_de(par[a][c])["root_slots"]) for c in pairs)
        print(f"   {a}: same roots as its reference, 95 % Wilson interval: {wilson(k, len(pairs))}")
if "floor" in par:
    runs_f = [r for r in par["floor"].values() if complet(r)]
    k = sum(bool(inertes(r)[80]) for r in runs_f)
    print(f"   floor: first step (t = 80) inert, 95 % Wilson interval: {wilson(k, len(runs_f))}")
for a in ("adapt", "thr05"):
    if a in par:
        k = sum(r["n_lineages"] == 1 for r in par[a].values())
        print(f"   {a}: one root, 95 % Wilson interval: {wilson(k, len(par[a]))}")
print(f"\n   against bon4 (four free draws, ir_max = best-of-4)")
print(f"   {'arm':7s} {'n':>3s} | {'ir_max':>26s} | {'mean ir':>26s} | wins")
for a in ORDRE:
    com = sorted(set(par[a]) & set(bon))
    dm = [par[a][c]["ir_max"] - bon[c]["ir_max"] for c in com]
    da = [ir_moy(par[a][c]) - ir_moy(bon[c]) for c in com]
    print(f"   {a:7s} {len(com):3d} | {fmt(*boot(dm)):>26s} | {fmt(*boot(da)):>26s} | {int(np.sum(np.array(dm) > 0))}/{len(com)}")

# is the price flat? a single number: for each prompt, the mean of ir_max over every correction
# arm present, against ctl. This is the "flat price" test of finding 13, which has more power
# than six separate CIs that graze zero on the same side.
# corrections in the sense of finding 14, not the bisection arms (stat0, multi, vae, idx, R1,
# which imitate the published code) nor late (which repairs nothing); floor2 is at n = 34 (ASSESSMENT.md)
corr = [a for a in ("adapt", "floor", "lam2", "fadapt", "floor2", "thr05", "rise") if a in par]
if corr:
    com = sorted({c for a in corr for c in par[a] if ref_de(par[a][c]) is not None})
    dm = [np.mean([par[a][c]["ir_max"] - ref_de(par[a][c])["ir_max"] for a in corr
                   if c in par[a] and ref_de(par[a][c]) is not None]) for c in com]
    print(f"\n   flat price: mean of ir_max over the correction arms ({', '.join(corr)}) - ctl, "
          f"n={len(com)}: {fmt(*boot(dm))}")
    dispersion = [np.std([par[a][c]["ir_max"] - ref_de(par[a][c])["ir_max"] for a in corr
                          if c in par[a] and ref_de(par[a][c]) is not None]) for c in com]
    print(f"   standard deviation of the price across arms, per prompt, median: {np.median(dispersion):.3f}")

# between correction arms, the pairs that settle the ranking of finding 14
paires = [("floor2", "lam2"), ("floor2", "fadapt"), ("late", "floor"), ("late", "ctl"), ("late", "floor2"), ("fadapt", "floor")]
print(f"\n   between arms, paired: lineages, weighted ESS, ir_max")
for a, b in paires:
    if a not in par or b not in par:
        continue
    com = sorted(set(par[a]) & set(par[b]))
    if len(com) < 2:
        continue
    dl = [par[a][c]["n_lineages"] - par[b][c]["n_lineages"] for c in com]
    de = [ess_racines(par[a][c]) - ess_racines(par[b][c]) for c in com if complet(par[a][c]) and complet(par[b][c])]
    dm = [par[a][c]["ir_max"] - par[b][c]["ir_max"] for c in com]
    print(f"   {a} - {b:7s} n={len(com):2d} | lineages {fmt(*boot(dl))} | wtd ESS {fmt(*boot(de))} | ir_max {fmt(*boot(dm))}")

# --- 4. verdicts, one per criterion -----------------------------------------
print(f"\n4. verdicts (criteria of ASSESSMENT.md, fixed before the data)")
for a in ORDRE:
    if a in ("ctl", "lam0"):
        continue
    d = par[a]
    ps = sorted(set(d) & set(bon))
    lam = d[ps[0]]["lam"]
    un = stats[a]["un"]
    pl = np.mean([ess(lam * np.array(bon[c]["ir"])) for c in ps])
    er = np.mean([ess_racines(d[c]) for c in ps if complet(d[c])])
    c1 = "YES" if un < 0.25 else "no"
    c2 = er / pl
    com = [c for c in sorted(d) if ref_de(d[c]) is not None]
    m_ctl, lo_ctl, hi_ctl = boot([d[c]["ir_max"] - ref_de(d[c])["ir_max"] for c in com]) if com else (np.nan,) * 3
    m_bon, lo_bon, hi_bon = boot([d[c]["ir_max"] - bon[c]["ir_max"] for c in ps])
    c3_ctl = "CI covers 0" if lo_ctl <= 0 <= hi_ctl else "CI excludes 0"
    c3_bon = "CI covers 0" if lo_bon <= 0 <= hi_bon else ("above" if lo_bon > 0 else "below")
    print(f"   {a:7s} (1) under 25 % with one root: {c1} ({un:.0%})   "
          f"(2) wtd ESS / ceiling of its target: {c2:.2f} ({er:.2f} / {pl:.2f})   "
          f"(3) ir_max - ctl {m_ctl:+.3f} ({c3_ctl}), ir_max - bon4 {m_bon:+.3f} ({c3_bon})")
print("   (3) is read with the `flat price` line above: six CIs that graze zero on the same side are not six zeros.")
