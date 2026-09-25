"""Readout of R0: the authors' code against bon4, ctl and table 1 of the paper.

Reads results/sd_authors_R0.json (run_authors.py), results/sd_baseline.json (bon4, k1, fk4 at
seed 2024), results/sd_ref_fields100.json (ctl = the FK row of the current code) and
out/probe.json (R1 if present). Everything is paired by prompt_id; R0 does not have our x_T, so the
paired difference is to be read as two samples on the same prompts, not as
a counterfactual at fixed noise. Pre-registered decision rule (docs/protocol_sd.md, 22/09):
the gap to the paper is "closed" if the paired R1 - bon4 >= +0.12, "bounded" otherwise.
"""
import json
from pathlib import Path
import numpy as np

LAB = Path(__file__).resolve().parents[1]
R = LAB.parent / "results"
CHECKOUT = Path("/home/onyxia/work/diffusion-models/results")   # the GPU output is written there
PAPIER = {"k1": 0.187, "bon4": 0.737, "fk4": 0.898}              # experiments_new.tex:82-85, SD v1.5


def se(x):
    x = np.asarray(x, float)
    return f"{x.mean():+.3f} +/- {x.std(ddof=1) / len(x) ** .5:.3f} (n={len(x)})"


import sys
src = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    CHECKOUT / "sd_authors_R0.json" if (CHECKOUT / "sd_authors_R0.json").exists() else R / "sd_authors_R0.json")
R0 = json.loads(src.read_text())["runs"] if src.exists() else []
if not R0:
    print(f"(no R0 yet: {src} missing)\n")
base = json.loads((R / "sd_baseline.json").read_text())["runs"]
bon = {r["prompt_id"]: r for r in base if r["sampler"] == "bon4" and r["seed"] == 2024}
k1 = {r["prompt_id"]: r for r in base if r["sampler"] == "k1" and r["seed"] == 2024}
ctl = {r["prompt_id"]: r for r in json.loads((R / "sd_ref_fields100.json").read_text())["runs"]
       if r["sampler"] == "fk4" and r["seed"] == 2024}
probe = json.loads((LAB / "out" / "probe.json").read_text())["runs"]
R1 = {r["prompt_id"]: r for r in probe if r["arm"] == "R1"}

print(__doc__.splitlines()[0], "\n")
for cfg, seed in sorted({(r["sampler"], r["seed"]) for r in R0}):
    d = {r["prompt_id"]: r for r in R0 if r["sampler"] == cfg and r["seed"] == seed}
    ps = sorted(d)
    print(f"{cfg} (seed {seed}): {len(ps)} prompts, {d[ps[0]]['config']}"
          + ("  <- same x_T as bon4 / ctl / R1" if seed == 2024 and "_g" in cfg else ""))
    print(f"   mean ir_max {np.mean([d[p]['ir_max'] for p in ps]):.3f}   mean ir of the 4 {np.mean([d[p]['ir_mean'] for p in ps]):.3f}"
          f"   distinct images {np.mean([d[p]['n_distinct_images'] for p in ps]):.2f}/4"
          f" (fewer than 4 in {np.mean([d[p]['n_distinct_images'] < 4 for p in ps]):.0%} of runs)"
          f"   div_pix {np.mean([d[p]['div_pix'] for p in ps]):.3f}   {np.mean([d[p]['seconds'] for p in ps]):.0f} s/run")
    print(f"   paper, SD v1.5: k1 {PAPIER['k1']}, bon4 {PAPIER['bon4']}, FK {PAPIER['fk4']} (FK - bon4 = +0.161)")
    for nom, ref in (("bon4", bon), ("k1", k1), ("ctl", ctl)):
        com = [p for p in ps if p in ref]
        print(f"   {cfg} - {nom:4s} on ir_max: {se([d[p]['ir_max'] - ref[p]['ir_max'] for p in com])}"
              f"   ({sum(d[p]['ir_max'] > ref[p]['ir_max'] for p in com)}/{len(com)} won)")
    if R1:
        com = [p for p in ps if p in R1]
        if com:
            print(f"   {cfg} - R1   on ir_max: {se([d[p]['ir_max'] - R1[p]['ir_max'] for p in com])}   (same implementation choices, different code)")
    print()

if R1:
    ps = sorted(set(R1) & set(bon))
    x = np.array([R1[p]["ir_max"] - bon[p]["ir_max"] for p in ps])
    print(f"R1 - bon4 paired, {len(ps)} prompts: {se(x)}; against ctl: {se([R1[p]['ir_max'] - ctl[p]['ir_max'] for p in ps if p in ctl])}")
    print(f"   rule: gap to the paper {'CLOSED' if x.mean() >= 0.12 else 'BOUNDED'} (threshold +0.12; paper +0.161; ctl - bon4 = "
          f"{np.mean([ctl[p]['ir_max'] - bon[p]['ir_max'] for p in ps if p in ctl]):+.3f} on these prompts)")
    print(f"   R1: lineages {np.mean([R1[p]['n_lineages'] for p in ps]):.2f}, one root in {np.mean([R1[p]['n_lineages'] == 1 for p in ps]):.0%},"
          f" resamplings {np.mean([R1[p]['n_resamplings'] for p in ps]):.2f}, t=80 inert in "
          f"{np.mean([(np.ptp(R1[p]['logG_at_schedule'][0]) < 1e-9) for p in ps]):.0%}")

# the four runs of the released FK on their 40 common prompts, averaged per prompt first (a pooled
# row over 4 x 40 run-prompts would count each prompt four times and shrink the standard error)
fk_runs = [{r["prompt_id"]: r for r in R0 if r["sampler"] == c and r["seed"] == s}
           for c, s in sorted({(r["sampler"], r["seed"]) for r in R0 if r["sampler"].startswith("authors_paper")})]
common = sorted(set.intersection(*(set(d) for d in fk_runs)) & set(bon))
per_prompt = [np.mean([d[p]["ir_max"] for d in fk_runs]) - bon[p]["ir_max"] for p in common]
names = sorted({(r["sampler"], r["seed"]) for r in R0 if r["sampler"].startswith("authors_paper")})
for (c, sd), d in zip(names, fk_runs):
    x = np.array([d[p]["ir_max"] - bon[p]["ir_max"] for p in common])
    print(f"  {c} seed {sd} on the common prompts: {se(x)}, {(0.161 - x.mean()) / (x.std(ddof=1) / len(x) ** .5):.2f} "
          f"standard errors under the paper's +0.161")
print(f"\nThe {len(fk_runs)} runs of the released FK on their {len(common)} common prompts, averaged per prompt, "
      f"minus bon4: {se(per_prompt)}; ir_max {np.mean([np.mean([d[p]['ir_max'] for d in fk_runs]) for p in common]):.3f}")
pairs = {c: {r["prompt_id"]: r for r in R0 if r["sampler"] == c and r["seed"] == 2024} for c in ("authors_paper", "authors_paper_g")}
com = sorted(set(pairs["authors_paper"]) & set(pairs["authors_paper_g"]))
if com:
    g = np.array([pairs['authors_paper'][p]['ir_max'] - pairs['authors_paper_g'][p]['ir_max'] for p in com])
    print(f"Seed 2024, own seeding - through a generator, same x_T: "
          f"{se(g)} ({g.mean() / (g.std(ddof=1) / len(g) ** .5):.1f} standard errors); "
          f"R1 ir_max {np.mean([r['ir_max'] for r in R1.values()]):.3f} over {len(R1)} prompts")
# the two seedings at seed 42 share their x_T (their free sampler agrees slot by slot), as at 2024
fr = {c: {r["prompt_id"]: r for r in R0 if r["sampler"] == c and r["seed"] == 42} for c in ("authors_free", "authors_free_g")}
cf = sorted(set(fr["authors_free"]) & set(fr["authors_free_g"]))
print(f"Seed 42, free sampler under both seedings: {sum(fr['authors_free'][p]['ir'] == fr['authors_free_g'][p]['ir'] for p in cf)}/{len(cf)} prompts equal on every slot")
p42 = {c: {r["prompt_id"]: r for r in R0 if r["sampler"] == c and r["seed"] == 42} for c in ("authors_paper", "authors_paper_g")}
c42 = [p for p in common if p in p42["authors_paper"] and p in p42["authors_paper_g"]]
g42 = np.array([p42["authors_paper"][p]["ir_max"] - p42["authors_paper_g"][p]["ir_max"] for p in c42])
print(f"Seed 42, own seeding - through a generator, same x_T: {se(g42)} ({g42.mean() / (g42.std(ddof=1) / len(g42) ** .5):.1f} standard errors)")
means = np.array([np.mean([d[p]["ir_max"] for p in common]) for d in fk_runs])
within = np.sqrt(np.mean([np.var([d[p]["ir_max"] for d in fk_runs], ddof=1) for p in common]))
print(f"The four run means on the common prompts: {', '.join(f'{m:.3f}' for m in means)}; their standard deviation "
      f"{means.std(ddof=1):.2f}, against {within / len(common) ** .5:.2f} expected from the run-to-run spread within a prompt "
      f"(RMS {within:.2f}) if the runs were exchangeable")


# the released code ran on the benchmark's first 40 prompts only: the same-machine references there and on the rest
order = [p["id"] for p in json.loads((LAB.parent / "data" / "imagereward-benchmark-prompts.json").read_text())]
fk4 = {r["prompt_id"]: r for r in base if r["sampler"] == "fk4" and r["seed"] == 2024}
print(f"\nThe common prompts are benchmark positions {min(map(order.index, common))} to {max(map(order.index, common))}")
for nom, ps in (("these 40", common), ("the other 60", sorted(set(bon) - set(common))), ("all 100", sorted(bon))):
    print(f"   on {nom}: bon4 {np.mean([bon[p]['ir_max'] for p in ps]):.3f}, fk4 {np.mean([fk4[p]['ir_max'] for p in ps]):.3f}, "
          f"R1 {np.mean([R1[p]['ir_max'] for p in ps]):.3f}; fk4 - bon4 {se([fk4[p]['ir_max'] - bon[p]['ir_max'] for p in ps])}, "
          f"R1 - bon4 {se([R1[p]['ir_max'] - bon[p]['ir_max'] for p in ps])}")

# sessions E and F (25/09, the T4): the released code at seed 2024 completed to the 100 prompts, and
# its commit before "address max potential bug" (699c929); both paired by x_T with bon4 and R1
for label, path, sampler in (("E, global seed", R / "sd_authors_R0_100.json", "authors_paper"),
                             ("E, generator", R / "sd_authors_R0_100.json", "authors_paper_g"),
                             ("F, before the fix, global seed", R / "sd_authors_prefix.json", "authors_paper_prefix")):
    if not path.exists():
        continue
    d = {r["prompt_id"]: r for r in json.loads(path.read_text())["runs"] if r["sampler"] == sampler and r["seed"] == 2024}
    if not d:
        continue
    ps = [p for p in order if p in d]
    rest = [p for p in ps if p not in common]
    print(f"\nSession {label}: {len(ps)} prompts ({len(rest)} beyond the first 40), ir_max {np.mean([d[p]['ir_max'] for p in ps]):.3f}, "
          f"fewer than 4 distinct images in {np.mean([d[p]['n_distinct_images'] < 4 for p in ps]):.0%} of runs, "
          f"devices {sorted({d[p].get('device', 'not recorded') for p in ps})}")
    for nom, sub in (("all", ps), ("first 40", [p for p in ps if p in common]), ("other 60", rest)):
        if len(sub) > 1:
            print(f"   {nom}: - bon4 {se([d[p]['ir_max'] - bon[p]['ir_max'] for p in sub])}, - R1 {se([d[p]['ir_max'] - R1[p]['ir_max'] for p in sub])}, "
                  f"- fk4 {se([d[p]['ir_max'] - fk4[p]['ir_max'] for p in sub])}")
    if sampler == "authors_paper_prefix" and (R / "sd_authors_R0_100.json").exists():
        post = {r["prompt_id"]: r for r in json.loads((R / "sd_authors_R0_100.json").read_text())["runs"]
                if r["sampler"] == "authors_paper" and r["seed"] == 2024}
        both = [p for p in ps if p in post]
        x = np.array([d[p]["ir_max"] - bon[p]["ir_max"] for p in ps])
        print(f"   before the fix - after it (global seed, same x_T): {se([d[p]['ir_max'] - post[p]['ir_max'] for p in both])}")
        print(f"   rule of 22/09 applied: {'CLOSED' if x.mean() >= 0.12 else 'not closed'} (threshold +0.12)")
if (R / "sd_authors_R0_100.json").exists():
    E = json.loads((R / "sd_authors_R0_100.json").read_text())["runs"]
    g = {r["prompt_id"]: r for r in E if r["sampler"] == "authors_paper" and r["seed"] == 2024}
    h = {r["prompt_id"]: r for r in E if r["sampler"] == "authors_paper_g" and r["seed"] == 2024}
    both = [p for p in order if p in g and p in h]
    for nom, sub in (("all", both), ("first 40", [p for p in both if p in common]), ("other 60", [p for p in both if p not in common])):
        if len(sub) > 1:
            print(f"   global seed - generator, seed 2024, {nom}: {se([g[p]['ir_max'] - h[p]['ir_max'] for p in sub])}")
