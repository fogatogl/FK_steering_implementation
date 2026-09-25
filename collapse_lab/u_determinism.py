"""Is the smc.models path replayable from one process to another? (finding 19, nuit4.sh)

Compares `ctl` on prompts 0 and 1 between: det_a and det_b (two processes, nothing changed),
det_c (cudnn.benchmark = False, use_deterministic_algorithms), probe_C.json (the morning's
session C, same pod) and sd_ref_fields100.json (21/09).

    python collapse_lab/u_determinism.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
ROOT = OUT.parent.parent


def charge(path, arm="ctl", sampler=None):
    runs = json.loads(path.read_text())["runs"]
    if sampler:
        return {r["prompt_id"]: r for r in runs if r["sampler"] == sampler}
    return {r["prompt_id"]: r for r in runs if r["arm"] == arm}


def arrondi(r):
    return [round(float(v), 4) for v in r["ir"]]


def ligne(nom, recs):
    for pid in sorted(recs):
        r = recs[pid]
        print(f"  {nom:10s} {pid:>6}  ir={r['ir']}  ir_max={r['ir_max']:+.4f}  "
              f"roots={r.get('root_slots')}  ESS={r.get('ess_at_schedule')}")


fichiers = {n: OUT / f"det_{n}.json" for n in "abc"}
present = {n: charge(p) for n, p in fichiers.items() if p.exists()}
pids = set.intersection(*(set(v) for v in present.values())) if present else set()
ref = charge(ROOT / "results" / "sd_ref_fields100.json", sampler="fk4")
sesC = charge(OUT / "probe_C.json") if (OUT / "probe_C.json").exists() else {}

print(f"processes that returned: {sorted(present)}; common prompts: {sorted(pids)}")
for pid in sorted(pids):
    print(f"\nprompt {pid}")
    ligne("21/09 ref", {pid: ref[pid]} if pid in ref else {})
    ligne("session C", {pid: sesC[pid]} if pid in sesC else {})
    for n in sorted(present):
        ligne(f"det_{n}", {pid: present[n][pid]})

if {"a", "b"} <= set(present):
    identiques = all(arrondi(present["a"][p]) == arrondi(present["b"][p]) for p in pids)
    print(f"\nA == B on the rounded scores (two processes, nothing changed): {identiques}")
if {"a", "c"} <= set(present):
    identiques = all(arrondi(present["a"][p]) == arrondi(present["c"][p]) for p in pids)
    print(f"A == C (C with the deterministic flags): {identiques}")
if sesC and present:
    n0 = sorted(present)[0]
    same = all(arrondi(present[n0][p]) == arrondi(sesC[p]) for p in pids if p in sesC)
    print(f"det_{n0} == the morning's session C (same pod, another process): {same}")
if ref and present:
    n0 = sorted(present)[0]
    same = all(arrondi(present[n0][p]) == arrondi(ref[p]) for p in pids if p in ref)
    print(f"det_{n0} == the 21/09 reference: {same}")
