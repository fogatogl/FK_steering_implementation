"""E. What the 19 variants already on disk say about the collapse.

n_lineages exists only since 21/09; div_pix, on the other hand, is in every
file and separates "four identical images" from "four different images".
bon4 gives the ceiling (four free draws), fk4 the observed floor.
"""
import json, statistics as st
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
load = lambda p: json.loads((R / p).read_text())["runs"]

base = load("sd_baseline.json")
div = load("sd_baseline_div20.json")   # div_pix exists only in this file
ref = {r["prompt_id"]: r for r in base if r["sampler"] == "bon4" and r["seed"] == 2024}
plancher = st.mean(r["div_pix"] for r in div if r["sampler"] == "fk4")
plafond = st.mean(r["div_pix"] for r in div if r["sampler"] == "bon4")
print(__doc__.splitlines()[0], "\n")
print(f"  bon4 (4 free draws)   div_pix {plafond:.4f}   <- ceiling")
print(f"  fk4  (reference, 20)  div_pix {plancher:.4f}   <- floor\n")
print(f"  {'variant':10s} {'lam':>5s} {'thresh':>6s} {'schedule':>22s} {'div_pix':>8s} "
      f"{'lineages':>8s} {'resampl':>8s} {'ir_max':>8s} {'gap/fk4':>10s}")

fkref = {r["prompt_id"]: r for r in base if r["sampler"] == "fk4" and r["seed"] == 2024}
for nom in ("S80", "S60", "S40", "D10", "L2", "L5", "L20", "A05", "T1", "T2", "T1A05",
            "T2A05", "T1t", "T2t", "T1tA05", "T2tA05", "fk4_stat", "fk4_diff"):
    try:
        runs = load(f"sd_variants/{nom}.json")
    except FileNotFoundError:
        continue
    r0 = runs[0]
    com = [r for r in runs if r["prompt_id"] in fkref]
    d = [r["ir_max"] - fkref[r["prompt_id"]]["ir_max"] for r in com]
    lin = [r["n_lineages"] for r in runs if r.get("n_lineages")]
    sched = str(r0.get("schedule_t"))
    print(f"  {nom:10s} {r0.get('lam', 0):5.1f} {r0.get('threshold', 0):6.2f} {sched:>22s} "
          f"{(f'{st.mean(dv):.4f}' if (dv := [r['div_pix'] for r in runs if r.get('div_pix') is not None]) else '   -'):>8s} "
          f"{(f'{st.mean(lin):.2f}' if lin else '  -'):>8s} "
          f"{st.mean(r['n_resamplings'] for r in runs):8.2f} "
          f"{st.mean(r['ir_max'] for r in runs):+8.4f} "
          f"{(f'{st.mean(d):+.4f}' if d else '-'):>10s}")

print("\n  reading: no variant brings div_pix back up toward the bon4 ceiling except those")
print("  that weaken the FIRST step (T2, lambda_1 = 0.4) or remove it (S80 has only")
print("  one). The schedule, the ramp, the form of the potential and the potential itself")
print("  leave div_pix at the floor.")
