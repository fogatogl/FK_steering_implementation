"""Readout of session G: the released code against best-of-4 and FK on seeds 2024, 2025 and 2026.

Seed 2024 comes from the T4 (sd_baseline.json, sessions E and F), seeds 2025 and 2026 from the A2
(sd_seeds_bon4.json, sd_seeds_authors.json). Each seed pairs by x_T inside one machine; the three-seed
row averages each prompt's paired difference over the seeds, then takes the standard error over the
prompts, as make_table_sd.py does for fk4 - bon4. Pre-registration: docs/protocol_sd.md, session G.
"""
import json
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[1] / "results"
PAPER = 0.161


def load(name, sampler, seed):
    p = R / name
    if not p.exists():
        return {}
    return {r["prompt_id"]: r for r in json.loads(p.read_text())["runs"] if r["sampler"] == sampler and r["seed"] == seed}


def se(x):
    x = np.asarray(x, float)
    return f"{x.mean():+.3f} +/- {x.std(ddof=1) / len(x) ** .5:.3f} (n={len(x)}, {int((x > 0).sum())} won)"


arms = {2024: {"bon4": load("sd_baseline.json", "bon4", 2024), "fk4": load("sd_baseline.json", "fk4", 2024),
               "post": load("sd_authors_R0_100.json", "authors_paper", 2024),
               "pre": load("sd_authors_prefix.json", "authors_paper_prefix", 2024)}}
arms[2024].update(bon4_T4=arms[2024]["bon4"], fk4_T4=arms[2024]["fk4"])
for s in (2025, 2026):
    arms[s] = {"bon4": load("sd_seeds_bon4.json", "bon4", s), "fk4": load("sd_seeds_fk4.json", "fk4", s),
               "post": load("sd_seeds_authors.json", "authors_paper", s),
               "pre": load("sd_seeds_authors.json", "authors_paper_prefix", s),
               "bon4_T4": load("sd_baseline.json", "bon4", s), "fk4_T4": load("sd_baseline.json", "fk4", s)}

print(__doc__.splitlines()[0], "\n")
for s, a in arms.items():
    for k, d in a.items():
        if d and not (s == 2024 and k.endswith("_T4")):
            v = list(d.values())
            print(f"seed {s} {k:8s} {len(d):3d} prompts, ir_max {np.mean([r['ir_max'] for r in v]):.3f}, "
                  f"{np.median([r['seconds'] for r in v]):.1f} s/run, devices {sorted({r.get('device', 'not recorded') for r in v})}"
                  + (f", fewer than 4 distinct images in {np.mean([r['n_distinct_images'] < 4 for r in v]):.0%}"
                     if "n_distinct_images" in v[0] else ""))

print("\nThe machine: best-of-4 on the A2 against the T4 at the same seeds")
xa, xt = [], []
for s in (2025, 2026):
    a, t = arms[s]["bon4"], arms[s]["bon4_T4"]
    ps = sorted(set(a) & set(t))
    if len(ps) > 1:
        print(f"   seed {s}: A2 - T4 {se([a[p]['ir_max'] - t[p]['ir_max'] for p in ps])}, "
              f"slots equal within 1e-3 in {np.mean([np.allclose(a[p]['ir'], t[p]['ir'], atol=1e-3) for p in ps]):.0%} of prompts")
        xa += [a[p]["ir_max"] for p in ps]
        xt += [t[p]["ir_max"] for p in ps]
if len(xa) > 1:
    print(f"   ir_max correlation over {len(xa)} (prompt, seed) pairs: {np.corrcoef(xa, xt)[0, 1]:.2f}")

print("\nPer seed, paired by x_T inside one machine")
diffs = {}
for nom, x, y in (("post - bon4", "post", "bon4"), ("pre - bon4", "pre", "bon4"), ("pre - post", "pre", "post"),
                  ("post - fk4", "post", "fk4"), ("pre - fk4", "pre", "fk4"), ("fk4 - bon4", "fk4", "bon4"),
                  ("fk4 - bon4 T4", "fk4_T4", "bon4_T4")):
    for s, a in arms.items():
        ps = sorted(set(a[x]) & set(a[y]))
        if len(ps) > 1:
            diffs.setdefault(nom, {})[s] = {p: a[x][p]["ir_max"] - a[y][p]["ir_max"] for p in ps}
            print(f"   seed {s} {nom:16s} {se(list(diffs[nom][s].values()))}")

print("\nOver the seeds, each prompt averaged first (seeds complete at 100 prompts only)")
for nom, per_seed in diffs.items():
    seeds = [s for s, d in per_seed.items() if len(d) == 100]
    if len(seeds) < 2:
        print(f"   {nom}: {len(seeds)} complete seed(s), nothing pooled")
        continue
    ps = sorted(set.intersection(*(set(per_seed[s]) for s in seeds)))
    x = np.array([np.mean([per_seed[s][p] for s in seeds]) for p in ps])
    m, e = x.mean(), x.std(ddof=1) / len(x) ** .5
    means = [np.mean(list(per_seed[s].values())) for s in seeds]
    one = np.mean([np.std(list(per_seed[s].values()), ddof=1) / 10 for s in seeds])
    print(f"   {nom:16s} seeds {seeds}: {se(x)}; seed means {', '.join(f'{v:+.3f}' for v in means)}, "
          f"their spread {np.std(means, ddof=1):.3f} against {one:.3f} for one seed's standard error"
          + (f"; the paper's +{PAPER} is {(PAPER - m) / e:.1f} standard errors above; gap rule (+0.12): "
             f"{'CLOSED' if m >= 0.12 else 'not closed'}, bootstrap 95 % "
             f"[{np.percentile(b := np.random.default_rng(0).choice(x, (20000, len(x))).mean(1), 2.5):+.3f}, "
             f"{np.percentile(b, 97.5):+.3f}]" if nom in ("post - bon4", "pre - bon4") else ""))
for k in ("bon4", "fk4", "post", "pre"):
    seeds = [s for s in arms if len(arms[s][k]) == 100]
    if len(seeds) > 1:
        print(f"   {k}: ir_max {np.mean([np.mean([r['ir_max'] for r in arms[s][k].values()]) for s in seeds]):.3f} over seeds {seeds}")
