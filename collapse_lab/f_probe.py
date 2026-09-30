"""F. Readout of the probe: when the lineages die, and what the floor changes.

Five arms on the same 20 prompts and the same x_T (seed 2024, seed_effective =
2024000 + i, identical to scripts/run_sd_baseline.py). Everything is paired by prompt.
"""
import json, itertools
from pathlib import Path
import numpy as np

LAB = Path(__file__).resolve().parent
R = LAB.parent / "results"
runs = json.loads((LAB / "out" / "probe.json").read_text())["runs"]
par = {}
for r in runs:
    if "logG_at_schedule" not in r:    # six ctl restored without weights from ref100 (collapse_lab/ASSESSMENT.md, 23/09)
        continue
    par.setdefault(r["arm"], {})[r["prompt_id"]] = r

base = json.loads((R / "sd_baseline.json").read_text())["runs"]
bon = {r["prompt_id"]: r for r in base if r["sampler"] == "bon4" and r["seed"] == 2024}
# The fk4 of sd_baseline.json dates from 20/09 and predates 5025180: it no longer describes the
# code the probe runs. The reference FK row is sd_ref_fields100.json.
ref100 = json.loads((R / "sd_ref_fields100.json").read_text())["runs"]
fk4 = {r["prompt_id"]: r for r in ref100 if r["sampler"] == "fk4" and r["seed"] == 2024}
ts = [80, 60, 40, 20, 0]
print(__doc__.splitlines()[0], "\n")
print(f"  arms present: " + ", ".join(f"{a} ({len(d)} prompts)" for a, d in par.items()))

# --- 0. pilot checks -------------------------------------------------------
if "ctl" in par:
    com = sorted(set(par["ctl"]) & set(fk4))
    d = np.array([par["ctl"][c]["ir_max"] - fk4[c]["ir_max"] for c in com])
    print(f"\n0a. pilot check: ctl against the fk4 of sd_ref_fields100.json, {len(com)} prompts")
    print(f"    ir_max, paired difference {d.mean():+.4f} +/- {d.std(ddof=1)/max(len(d),1)**.5:.4f} "
          f"(identical to within {np.abs(d).max():.4f} at worst)")
if "lam0" in par:
    com = sorted(set(par["lam0"]) & set(bon))
    # slot by slot: that is the hypothesis (slot j of fk = slot j of bon4), and a sort
    # would only test that the sets are equal.
    e = np.array([np.abs(np.array(par["lam0"][c]["ir"]) - np.array(bon[c]["ir"])).max() for c in com])
    print(f"\n0b. pairing check: lam=0 must give back the 4 free draws of bon4")
    print(f"    max gap over the 4 ir, per prompt: median {np.median(e):.4f}, max {e.max():.4f}")
    print(f"    -> if it is zero, slot j of fk4 and slot j of bon4 do share x_T,")
    print(f"       which is the hypothesis of the information measurement (script c_information.py)")

# --- 0c. the bug detector, extended to the five rows -----------------------
# At threshold 1.0 logW restarts from zero at each resampling, and is zero at the
# unscheduled steps: the ESS read at step m must therefore be exactly that of logG_m alone.
print("\n0c. ESS recomputed from logG against the recorded ESS, the five rows")
for a in sorted(par):
    d, pire = par[a], 0.0
    for c in d:
        lg = np.array(d[c]["logG_at_schedule"])
        w = np.exp(lg - lg.max(1, keepdims=True))
        w /= w.sum(1, keepdims=True)
        pire = max(pire, float(np.abs(1 / (w ** 2).sum(1) - np.array(d[c]["ess_at_schedule"])).max()))
    print(f"    {a:8s} max gap over {len(d)} runs x 5 rows: {pire:.2e}"
          + ("   <- exact arithmetic" if pire < 1e-3 else "   <- expected: this arm does not resample at every step, the check assumes it does"
             if a in ("thr05",) else "   <- TO INVESTIGATE"))

# --- 1. where the lineages die ---------------------------------------------
print(f"\n1. number of distinct x_T roots after each scheduled step (mean over the prompts)")
print(f"   {'arm':8s} {'lam':>5s} {'floor':>9s} | " + " ".join(f"t={t:<5d}" for t in ts)
      + f" | {'resampl':>8s} {'div_pix':>8s} {'ir_max':>8s}")
for a in ("ctl", "floor", "adapt", "fadapt", "lam2", "floor2", "lam0"):
    if a not in par:
        continue
    d = par[a]
    tr = np.array([d[c]["lineages_trace"] for c in sorted(d)])
    print(f"   {a:8s} {d[sorted(d)[0]]['lam']:5.1f} {str(d[sorted(d)[0]]['plancher']):>9s} | "
          + " ".join(f"{v:<7.2f}" for v in tr.mean(0))
          + f" | {np.mean([d[c]['n_resamplings'] for c in d]):8.2f} "
          f"{np.mean([d[c]['div_pix'] for c in d]):8.4f} "
          f"{np.mean([d[c]['ir_max'] for c in d]):+8.4f}")
    if d[sorted(d)[0]].get("adaptatif"):
        lam_m = np.median([d[c]["lam_at_schedule"] for c in sorted(d)], axis=0)
        print(f"   {'':8s} {'lam_t':>5s} {'median':>9s} | " + " ".join(f"{v:<7.2f}" for v in lam_m))

# --- 2. the floor: how many steps does it make inert ------------------------
print(f"\n2. scheduled steps where the weights come out uniform (no selection)")
print(f"   {'arm':8s} | " + " ".join(f"t={t:<5d}" for t in ts))
for a in ("ctl", "floor", "adapt", "fadapt", "lam2", "floor2"):
    if a not in par:
        continue
    d = par[a]
    lg = np.array([d[c]["logG_at_schedule"] for c in sorted(d)])      # (n, 5, k)
    unif = (lg.max(2) - lg.min(2)) < 1e-9
    print(f"   {a:8s} | " + " ".join(f"{v:<7.0%}" for v in unif.mean(0)))

# --- 3. is the increment a level? -------------------------------------------
print(f"\n3. is logG - lam * r_phi CONSTANT across the slots? (then the weight is exp(lam*r),")
print(f"   a ranking on the level, and subtracting the potential protects against nothing)")
print(f"   {'arm':8s} | " + " ".join(f"t={t:<5d}" for t in ts) + "   (share of runs)")
for a in ("ctl", "floor", "adapt", "fadapt", "lam2", "floor2"):
    if a not in par:
        continue
    d = par[a]
    cst = []
    for c in sorted(d):
        lg = np.array(d[c]["logG_at_schedule"])
        r = np.array(d[c]["r_at_schedule"])
        lam_m = np.array(d[c].get("lam_at_schedule", [d[c]["lam"]] * len(lg)))[:, None]
        ecart = lg - lam_m * r
        cst.append((ecart.max(1) - ecart.min(1)) < 1e-3)
    print(f"   {a:8s} | " + " ".join(f"{v:<7.0%}" for v in np.array(cst).mean(0)))

# --- 4. the reward, paired --------------------------------------------------
print(f"\n4. final reward, paired differences per prompt (ImageReward, max over the 4)")
ref = par.get("ctl", {})
for a in ("floor", "adapt", "fadapt", "lam2", "floor2", "lam0"):
    if a not in par or not ref:
        continue
    com = sorted(set(par[a]) & set(ref))
    d = np.array([par[a][c]["ir_max"] - ref[c]["ir_max"] for c in com])
    print(f"   {a:8s} - ctl : {d.mean():+.4f} +/- {d.std(ddof=1)/len(d)**.5:.4f}  "
          f"({(d > 0).sum()}/{len(d)} prompts won)")
for a in ("ctl", "floor", "adapt", "fadapt", "lam2", "floor2"):
    if a not in par:
        continue
    com = sorted(set(par[a]) & set(bon))
    d = np.array([par[a][c]["ir_max"] - bon[c]["ir_max"] for c in com])
    print(f"   {a:8s} - bon4: {d.mean():+.4f} +/- {d.std(ddof=1)/len(d)**.5:.4f}  "
          f"({(d > 0).sum()}/{len(d)} prompts won)")

# --- 5. the information in the step that kills ------------------------------
def kendall(a, b):
    c = dd = 0
    for i, j in itertools.combinations(range(len(a)), 2):
        s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
        c += s > 0; dd += s < 0
    return (c - dd) / (c + dd) if c + dd else 0.0

if "ctl" in par:
    com = sorted(set(par["ctl"]) & set(bon))
    print(f"\n5. information in the first step: r_phi(t=80) against the final ir of the same root")
    tau = np.array([kendall(par["ctl"][c]["r_at_schedule"][0], bon[c]["ir"]) for c in com])
    top = np.array([int(np.argmax(par["ctl"][c]["r_at_schedule"][0]) == np.argmax(bon[c]["ir"]))
                    for c in com])
    print(f"   Kendall tau {tau.mean():+.3f} +/- {tau.std(ddof=1)/len(tau)**.5:.3f}, "
          f"top-1 {top.mean():.0%} against 25 % by chance, n={len(com)}")
