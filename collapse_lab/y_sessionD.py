"""Y. Session D (NVIDIA A2, 23/09 evening): which machine group, and what it measures there.

Reads out/session_D/probe_D.json (ctl, lam0, floor2, one process, the device in every
record), its images/ and hps_finals.json when they exist, and the files of both groups:
the A2 group (results/sd_ref_fields100.json of 21/09, session A in out/probe.json, runs at
85-92 s) and the fast group (out/probe_C.json, results/sd_baseline.json of 20/09). Confronts
the predictions of docs/protocol_sd.md, "Pre-registration of session D". Tolerates a
session still running: every count says its n.

    /home/onyxia/work/.venvs/ddpm/bin/python collapse_lab/y_sessionD.py
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from commun import bon4, groupe, load, wilson

LAB = Path(__file__).resolve().parent
D_PATH = LAB / "out" / "session_D" / "probe_D.json"
D = {}
for r in json.loads(D_PATH.read_text())["runs"]:
    D.setdefault(r["arm"], {})[r["prompt_id"]] = r
C = {}
for r in json.loads((LAB / "out" / "probe_C.json").read_text())["runs"]:
    C.setdefault(r["arm"], {})[r["prompt_id"]] = r
A = {}   # session A: the A2-group records of probe.json
for r in json.loads((LAB / "out" / "probe.json").read_text())["runs"]:
    if groupe(r) == "A2":
        A.setdefault(r["arm"], {})[r["prompt_id"]] = r
ref21 = {r["prompt_id"]: r for r in load("sd_ref_fields100.json") if r["sampler"] == "fk4" and r["seed"] == 2024}
bon = bon4()


def se(x):
    x = np.asarray(x, float)
    return f"{x.mean():+.3f} +/- {x.std(ddof=1) / len(x) ** .5:.3f} (n={len(x)})"


def slot_gap(a, b):
    return max(abs(x - y) for x, y in zip(a["ir"], b["ir"]))


print(__doc__.splitlines()[0])
devices = sorted({r["session"]["device"] for d in D.values() for r in d.values()})
secs = [r["seconds"] for d in D.values() for r in d.values()]
print(f"  records: " + ", ".join(f"{a} {len(d)}" for a, d in D.items())
      + f"; device {devices}; {np.median(secs):.1f} s per run (median), sessions "
      + f"{len({r['session_id'] for d in D.values() for r in d.values()})}")

print("\n0. which group: slot rewards equal to 0.001")
for name, other in (("21/09 reference (A2)", {"ctl": ref21}), ("session A (A2)", A), ("session C (fast)", C)):
    for a in ("ctl", "lam0", "floor2"):
        com = sorted(set(D.get(a, {})) & set(other.get(a, {})))
        if com:
            eq = sum(slot_gap(D[a][p], other[a][p]) < 1e-3 for p in com)
            print(f"   {a:6s} against {name:22s}: {eq}/{len(com)} prompts equal, max slot gap "
                  f"{max(slot_gap(D[a][p], other[a][p]) for p in com):.4f}")
if "lam0" in D and "lam0" in C:
    com = sorted(set(D["lam0"]) & set(C["lam0"]))
    x = [v for p in com for v in D["lam0"][p]["ir"]]
    y = [v for p in com for v in C["lam0"][p]["ir"]]
    print(f"   lam0 D against lam0 C: slot correlation {np.corrcoef(x, y)[0, 1]:.2f}, mean |ir_max gap| "
          f"{np.mean([abs(D['lam0'][p]['ir_max'] - C['lam0'][p]['ir_max']) for p in com]):.3f} (n={len(com)}; "
          f"predicted >= 0.99 and < 0.02)")

print("\n1. the same measurements on the A2, paired by x_T inside session D")
if "ctl" in D:
    print(f"   ctl: ir_max {np.mean([r['ir_max'] for r in D['ctl'].values()]):.3f}, roots "
          f"{np.mean([r['n_lineages'] for r in D['ctl'].values()]):.2f}, one root in "
          f"{sum(r['n_lineages'] == 1 for r in D['ctl'].values())}/{len(D['ctl'])}, div_pix "
          f"{np.mean([r['div_pix'] for r in D['ctl'].values()]):.3f}  (predicted within 0.08 of 0.799, 1.0-1.15 roots)")
    print(f"   ctl after the first resampling (t = 80): {np.mean([r['lineages_trace'][0] for r in D['ctl'].values()]):.2f} roots, "
          f"one root in {sum(r['lineages_trace'][0] == 1 for r in D['ctl'].values())}/{len(D['ctl'])};"
          f" resamplings per run at most {max(r['n_resamplings'] for r in D['ctl'].values())}")
if "floor2" in D:
    print(f"   floor2: roots {np.mean([r['n_lineages'] for r in D['floor2'].values()]):.2f}, one root in "
          f"{sum(r['n_lineages'] == 1 for r in D['floor2'].values())}/{len(D['floor2'])}, div_pix "
          f"{np.mean([r['div_pix'] for r in D['floor2'].values()]):.3f}  (predicted 2.8-3.2 roots)")
if "lam0" in D:
    print(f"   lam0: div_pix {np.mean([r['div_pix'] for r in D['lam0'].values()]):.3f}")
both = sorted(set(D.get("ctl", {})) & set(D.get("floor2", {})))
if both:
    print(f"   floor2 - ctl, ir_max: {se([D['floor2'][p]['ir_max'] - D['ctl'][p]['ir_max'] for p in both])}"
          f"  (predicted in [-0.10, +0.08]); mean of the four: "
          f"{se([np.mean(D['floor2'][p]['ir']) - np.mean(D['ctl'][p]['ir']) for p in both])}")
both = sorted(set(D.get("ctl", {})) & set(D.get("lam0", {})))
if both:
    print(f"   ctl - best-of-4 (lam0's best slot, same x_T, same process): "
          f"{se([D['ctl'][p]['ir_max'] - D['lam0'][p]['ir_max'] for p in both])}, "
          f"won {sum(D['ctl'][p]['ir_max'] > D['lam0'][p]['ir_max'] for p in both)}/{len(both)}")
    com = sorted(set(C["ctl"]) & set(C["lam0"]) & set(both))
    print(f"   the same on session C, same prompts: {se([C['ctl'][p]['ir_max'] - C['lam0'][p]['ir_max'] for p in com])}")
    print(f"   ctl D - ctl C, same x_T, two machines: {se([D['ctl'][p]['ir_max'] - C['ctl'][p]['ir_max'] for p in com])}, "
          f"same roots kept in {np.mean([set(D['ctl'][p]['root_slots']) == set(C['ctl'][p]['root_slots']) for p in com]):.0%}")
    same = sum(set(D['ctl'][p]['root_slots']) == set(C['ctl'][p]['root_slots']) for p in com)
    print(f"   same roots kept across the two machines, 95 % Wilson interval: {wilson(same, len(com))}")
both = sorted(set(D.get("floor2", {})) & set(D.get("lam0", {})))
if both:
    print(f"   floor2 - best-of-4 (lam0's best slot, same x_T, same process): "
          f"{se([D['floor2'][p]['ir_max'] - D['lam0'][p]['ir_max'] for p in both])}")
    same = [p for p in both if D['floor2'][p]['n_resamplings'] == 0
            and np.allclose(D['floor2'][p]['ir'], D['lam0'][p]['ir'], atol=1e-4)]
    four = [p for p in both if D['floor2'][p]['n_lineages'] == 4]
    print(f"   floor2 never resamples and returns lam0's four images in {len(same)}/{len(both)} prompts;"
          f" {len(set(same) & set(four))} of its {len(four)} four-root runs are these")
if "ctl" in D and "floor2" in D:
    print(f"   one root, 95 % Wilson interval: ctl {wilson(sum(r['n_lineages'] == 1 for r in D['ctl'].values()), len(D['ctl']))},"
          f" floor2 {wilson(sum(r['n_lineages'] == 1 for r in D['floor2'].values()), len(D['floor2']))}")

print("\n2. thumbnails: t = 80 is before any resampling, so ctl, lam0 and floor2 share it")
img = D_PATH.parent / "images"
worst, n = 0, 0
for p in sorted(set(D.get("ctl", {})) & set(D.get("lam0", {})) & set(D.get("floor2", {}))):
    for j in range(4):
        a = np.asarray(Image.open(img / "ctl" / f"{p}_t80_{j}.webp"), dtype=int)
        for other in ("lam0", "floor2"):
            worst = max(worst, int(np.abs(a - np.asarray(Image.open(img / other / f"{p}_t80_{j}.webp"), dtype=int)).max()))
    n += 1
print(f"   max pixel gap at t = 80 over {n} prompts: {worst}/255 (predicted <= 1)")

hps_path = D_PATH.parent / "hps_finals.json"
if hps_path.exists():
    H = json.loads(hps_path.read_text())
    com = sorted(p for p in H.get("ctl", {}) if p in H.get("lam0", {}))
    pick = lambda arm, p: H[arm][p][int(np.argmax(D[arm][p]["ir"]))]
    print(f"\n3. HPS v2.1 at the image ImageReward selects: ctl - best-of-4 "
          f"{se([pick('ctl', p) - pick('lam0', p) for p in com])}  (predicted within +/- 0.01)")

# the schedule ablation (late, no step at t = 80, results/sd_s60_full.json) ran on the A2 at 87 s
# per run; its reference in the same group is the 21/09 fk4 at seed 2024
s60 = {r["prompt_id"]: r for r in load("sd_s60_full.json") if r["seed"] == 2024}
com = sorted(set(s60) & set(ref21))
print(f"\n4. late - ctl on the A2 (sd_s60_full against sd_ref_fields100, seed 2024): "
      f"{se([s60[p]['ir_max'] - ref21[p]['ir_max'] for p in com])}")
print(f"   late at seed 2024: one root in {np.mean([s60[p]['n_lineages'] == 1 for p in com]):.0%}, roots "
      f"{np.mean([s60[p]['n_lineages'] for p in com]):.2f}, div_pix {np.mean([s60[p]['div_pix'] for p in com]):.3f} (n={len(com)})")

# the seconds per run that separate the two machine groups, over every probe record and the 20/09 table
from commun import references  # noqa: E402
secs = {"A2": [], "fast": []}
for r in json.loads((LAB / "out" / "probe.json").read_text())["runs"] + list(C.get("ctl", {}).values()) \
        + [r for d in D.values() for r in d.values()] + load("sd_baseline.json"):
    if r.get("seconds") and r.get("k", r.get("n")) == 4:   # four particles or four draws, the same work
        secs["A2" if (r.get("session", {}).get("device", "").endswith("A2") or r["seconds"] > 75) else "fast"].append(r["seconds"])
for g, v in secs.items():
    v = np.array(v)
    print(f"5. seconds per run, {g}: median {np.median(v):.1f}, 5 to 95 % {np.percentile(v, 5):.0f} to {np.percentile(v, 95):.0f} (n={len(v)})")

