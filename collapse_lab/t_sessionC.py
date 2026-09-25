"""T. Session C: ctl, lam0 and floor2 on 100 prompts in one process (out/probe_C.json).

The one file where slot pairing holds: same x_T, same process, so lam0[j] is the free
continuation of the root j that ctl and floor2 see. Confronts the predictions of
docs/protocol_sd.md ("Pre-registration of session C") and recomputes findings 3 and 5 bis
of FINDINGS.md on that pairing:
  - finding 3: does ctl's ranking of r_phi at t = 80 predict the free final ir (lam0)?
    Kendall tau per prompt, top-1;
  - finding 5 bis: A = best free root (max of lam0), M = mean root, B = the root ctl keeps,
    read in lam0, C = what ctl makes of it (ir_max). A - B is what the collapse costs in
    root, C - B what the steering and the best-of-four read-out return.
Controls: ctl_C against sd_ref_fields100.json (another machine) by prompt, slot correlation.
"""
import itertools, json
from pathlib import Path
import numpy as np

from commun import bon4, load, wilson

LAB = Path(__file__).resolve().parent
P = LAB / "out" / "probe_C.json"
runs = json.loads(P.read_text())["runs"] if P.exists() else json.loads(
    Path("/home/onyxia/work/diffusion-models/collapse_lab/out/probe_C.json").read_text())["runs"]
par = {}
for r in runs:
    par.setdefault(r["arm"], {})[r["prompt_id"]] = r
ref = {r["prompt_id"]: r for r in load("sd_ref_fields100.json") if r["sampler"] == "fk4" and r["seed"] == 2024}
bon = bon4()
rng = np.random.default_rng(0)


def se(x):
    x = np.asarray(x, float)
    return f"{x.mean():+.3f} +/- {x.std(ddof=1) / len(x) ** .5:.3f} (n={len(x)})"


def kendall(a, b):
    c = d = 0
    for i, j in itertools.combinations(range(len(a)), 2):
        s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
        c += s > 0; d += s < 0
    return (c - d) / (c + d) if c + d else 0.0


ctl, lam0, fl2 = par.get("ctl", {}), par.get("lam0", {}), par.get("floor2", {})
print(__doc__.splitlines()[0])
print(f"  arms: ctl {len(ctl)}, lam0 {len(lam0)}, floor2 {len(fl2)}\n")

# --- controls ------------------------------------------------------------------
com = sorted(set(ctl) & set(ref))
if com:
    a = np.array([ctl[p]["ir"] for p in com]).ravel(); b = np.array([ref[p]["ir"] for p in com]).ravel()
    print(f"0. ctl_C against ref100 (21/09), {len(com)} prompts: ir_max {se([ctl[p]['ir_max'] - ref[p]['ir_max'] for p in com])},"
          f" slot correlation {np.corrcoef(a, b)[0, 1]:.2f}, same root kept {np.mean([set(ctl[p]['root_slots']) == set(ref[p]['root_slots']) for p in com]):.0%}")
com = sorted(set(lam0) & set(bon))
if com:
    a = np.array([lam0[p]["ir"] for p in com]).ravel(); b = np.array([bon[p]["ir"] for p in com]).ravel()
    print(f"   lam0_C against bon4 (20/09), {len(com)} prompts: ir_max {se([lam0[p]['ir_max'] - bon[p]['ir_max'] for p in com])},"
          f" slot correlation {np.corrcoef(a, b)[0, 1]:.2f} (predicted < 0.7)")

# --- floor2 against ctl, paired by x_T inside the session ------------------------------
com = sorted(set(fl2) & set(ctl))
if com:
    d = np.array([fl2[p]["ir_max"] - ctl[p]["ir_max"] for p in com])
    idx = rng.integers(0, len(d), (4000, len(d))); m = d[idx].mean(1)
    print(f"\n1. floor2 - ctl (x_T paired), {len(com)} prompts: ir_max {d.mean():+.3f} [{np.percentile(m, 2.5):+.3f}, {np.percentile(m, 97.5):+.3f}]"
          f" (predicted [-0.14, -0.02]); mean ir {se([np.mean(fl2[p]['ir']) - np.mean(ctl[p]['ir']) for p in com])}")
    print(f"   floor2: {np.mean([fl2[p]['n_lineages'] for p in com]):.2f} roots (predicted 2.8-3.1), {np.mean([fl2[p]['n_lineages'] == 1 for p in com]):.0%} on one root (predicted 5-10 %),"
          f" div_pix {np.mean([fl2[p]['div_pix'] for p in com]):.3f}; ctl: {np.mean([ctl[p]['n_lineages'] for p in com]):.2f} roots, {np.mean([ctl[p]['n_lineages'] == 1 for p in com]):.0%}")
    print(f"   floor2 - bon4 ir_max {se([fl2[p]['ir_max'] - bon[p]['ir_max'] for p in com if p in bon])}; ctl - bon4 {se([ctl[p]['ir_max'] - bon[p]['ir_max'] for p in com if p in bon])}"
          f"   (bon4 = the 20/09 file, same machine)")
    own = sorted(set(com) & set(lam0))
    print(f"   against the best of the four free draws of the same process (lam0): floor2 {se([fl2[p]['ir_max'] - lam0[p]['ir_max'] for p in own])},"
          f" ctl {se([ctl[p]['ir_max'] - lam0[p]['ir_max'] for p in own])}")
    print(f"   one root, 95 % Wilson interval: ctl {wilson(sum(ctl[p]['n_lineages'] == 1 for p in com), len(com))},"
          f" floor2 {wilson(sum(fl2[p]['n_lineages'] == 1 for p in com), len(com))}")

# --- finding 3: the information of the first step, on the valid pairing ----------------
com = sorted(set(ctl) & set(lam0))
if com:
    tau = np.array([kendall(ctl[p]["r_at_schedule"][0], lam0[p]["ir"]) for p in com])
    top = np.array([int(np.argmax(ctl[p]["r_at_schedule"][0]) == np.argmax(lam0[p]["ir"])) for p in com])
    print(f"\n2. finding 3 recomputed: Kendall tau between ctl's r_phi(t=80) and the free ir (lam0) of the same root:"
          f" {tau.mean():+.3f} +/- {tau.std(ddof=1) / len(tau) ** .5:.3f} (predicted < 0.15); top-1 {top.mean():.0%} against 25 % by chance, n={len(com)}")
    print(f"   top-1, 95 % Wilson interval: {wilson(int(top.sum()), len(top))}")

    # --- finding 5 bis: the A / M / B / C decomposition on lam0 ---------------------------
    A = np.array([max(lam0[p]["ir"]) for p in com])
    M = np.array([np.mean(lam0[p]["ir"]) for p in com])
    B = np.array([np.mean([lam0[p]["ir"][j] for j in set(ctl[p]["root_slots"])]) for p in com])   # the kept root(s), read free
    C = np.array([ctl[p]["ir_max"] for p in com])
    print(f"\n3. finding 5 bis recomputed (n={len(com)}): A best free root {A.mean():+.3f}, M mean root {M.mean():+.3f},"
          f" B root(s) kept by ctl, read free {B.mean():+.3f}, C what ctl makes of it {C.mean():+.3f}")
    print(f"   B - M {se(B - M)} (is the kept root better than chance); A - B {se(A - B)} (predicted [0.25, 0.50]);"
          f" C - B {se(C - B)} (what follows the choice); C - A {se(C - A)} (= ctl - lam0 on ir_max)")
    Cm = np.array([np.mean(ctl[p]["ir"]) for p in com])
    print(f"   C - B split: the mean of ctl's four - B {se(Cm - B)} (the particles move up), C - that mean {se(C - Cm)} "
          f"(ir_max picks the best of four near-copies)")
    rang = np.array([1 + sum(v > lam0[p]["ir"][j] for v in lam0[p]["ir"]) for p in com for j in set(ctl[p]["root_slots"])])
    print(f"   mean rank of the kept root among the 4 free ones: {rang.mean():.2f} (2.5 at random, for a rank from 1 to 4)")

# what section 5 of the post quotes about the first steps of ctl in session C
e = np.array([ctl[p]["ess_at_schedule"] for p in ctl])
span = np.array([np.ptp(ctl[p]["r_at_schedule"][0]) for p in ctl])
print(f"\n4. ctl in session C, n={len(ctl)}: median ESS at t = 80, 60, 40, 20, 0: "
      + ", ".join(f"{v:.2f}" for v in np.median(e, axis=0))
      + f"; span of the four guide rewards at t = 80: mean {span.mean():.3f}, median {np.median(span):.3f}"
      + f"; resamplings per run {np.mean([ctl[p]['n_resamplings'] for p in ctl]):.2f}")
print(f"   after the first resampling (t = 80): ctl {np.mean([ctl[p]['lineages_trace'][0] for p in ctl]):.2f} roots, one root in "
      f"{sum(ctl[p]['lineages_trace'][0] == 1 for p in ctl)}/{len(ctl)}; at the end {np.mean([ctl[p]['n_lineages'] for p in ctl]):.2f};"
      f" resamplings per run at most {max(ctl[p]['n_resamplings'] for p in ctl)}")
print(f"   div_pix: ctl {np.mean([ctl[p]['div_pix'] for p in ctl]):.3f} +/- {np.std([ctl[p]['div_pix'] for p in ctl], ddof=1) / len(ctl) ** .5:.3f}, "
      f"floor2 {np.mean([fl2[p]['div_pix'] for p in fl2]):.3f} +/- {np.std([fl2[p]['div_pix'] for p in fl2], ddof=1) / len(fl2) ** .5:.3f}, "
      f"lam0 {np.mean([lam0[p]['div_pix'] for p in lam0]):.3f} +/- {np.std([lam0[p]['div_pix'] for p in lam0], ddof=1) / len(lam0) ** .5:.3f}")
