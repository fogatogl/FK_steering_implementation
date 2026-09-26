"""Table 1 (SD v1.5 row): what we measure, next to the paper, as Markdown on stdout.

From the root: `python scripts/make_table_sd.py [--out docs/table_sd.md]`.
The paper's values come from results/paper_table1_sd15.json. `ir_mean` is the mean of the
`ir` list of a record (the k particles of one prompt), then averaged over records; it is
printed only when every record of a row carries an `ir` list.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from figstyle import aggregate, load_runs

ROOT = Path(__file__).resolve().parent.parent
PAPER_JSON = ROOT / "results" / "paper_table1_sd15.json"

COLS = ["sampler", "prompts", "ImageReward (ir_max)", "paper", "diff",
        "ir_mean", "HPS", "paper", "diff", "UNet rows", "UNet calls", "seconds"]


def paper_rows(path):
    """{sampler: (ir_max, hps_at_ir_max)} from the paper's table, exact sampler names."""
    d = json.loads(Path(path).read_text())
    return {r["sampler"]: (r["ir_max"], r["hps_at_ir_max"]) for r in d["rows"]}


def sigma_seeds(runs, key):
    """Sigma of the prompt mean from one seed to the next; None with a single seed."""
    seeds = sorted({r["seed"] for r in runs})
    if len(seeds) < 2:
        return None
    means = [np.mean([r[key] for r in runs if r["seed"] == s]) for s in seeds]
    return float(np.std(means, ddof=1))


def cell(value, sigma):
    return f"{value:.3f}" + (f" +/- {sigma:.3f}" if sigma is not None else "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", nargs="+", default=[str(ROOT / "results" / "sd_baseline.json")])
    p.add_argument("--paper", default=str(PAPER_JSON))
    p.add_argument("--out")
    args = p.parse_args()

    runs = load_runs(args.json)
    if not runs:
        raise SystemExit("no record")
    paper = paper_rows(args.paper)

    xs, ir, _, _ = aggregate(runs, "sampler", "ir_max")
    _, hps, _, _ = aggregate(runs, "sampler", "hps_at_ir_max")
    _, unet_rows, _, _ = aggregate(runs, "sampler", "n_unet_rows")
    _, unet_calls, _, _ = aggregate(runs, "sampler", "n_unet_calls")
    _, seconds, _, _ = aggregate(runs, "sampler", "seconds")
    # increasing budget rather than alphabetical: k1, bon4, then FK
    n_per_sampler = {r["sampler"]: r["n"] for r in runs}
    order = np.argsort([n_per_sampler[s] for s in xs], kind="stable")

    table = [f"| {' | '.join(COLS)} |", "|---|" + "---:|" * (len(COLS) - 1)]
    for i in order:
        sub = [r for r in runs if r["sampler"] == xs[i]]
        ref = paper.get(xs[i])
        n_prompts = len({r["prompt_id"] for r in sub})
        p_ir, p_hps = (f"{ref[0]:.3f}", f"{ref[1]:.3f}") if ref else ("-", "-")
        d_ir, d_hps = (f"{ir[i] - ref[0]:+.3f}", f"{hps[i] - ref[1]:+.3f}") if ref else ("-", "-")
        if all("ir" in r for r in sub):
            ir_mean = cell(float(np.mean([np.mean(r["ir"]) for r in sub])), None)
        else:
            ir_mean = "-"
        table.append(
            f"| {xs[i]} | {n_prompts} | {cell(ir[i], sigma_seeds(sub, 'ir_max'))} | {p_ir} | {d_ir} "
            f"| {ir_mean} | {cell(hps[i], sigma_seeds(sub, 'hps_at_ir_max'))} | {p_hps} | {d_hps} "
            f"| {unet_rows[i]:.0f} | {unet_calls[i]:.0f} | {seconds[i]:.1f} |")

    seeds = sorted({r["seed"] for r in runs})
    text = "\n".join(table) + (
        f"\n\nAggregate of the file as is: records {len(runs)}, distinct prompts "
        f"{len({r['prompt_id'] for r in runs})}, seeds {seeds}. "
        "diff = measured - paper; ImageReward and HPS are those of the particle that "
        "ImageReward picks; ir_mean averages the k particles of a prompt first. The matched "
        "budget reads on the UNet rows, not on the calls: the k particles go through one "
        "batched forward. +/- is the sigma of the prompt mean across seeds.\n")

    print(text)
    # per prompt, seeds averaged first: the standard error over prompts, and the paired FK gain
    per = {x: {} for x in xs}
    for r in runs:
        per[r["sampler"]].setdefault(r["prompt_id"], []).append(r["ir_max"])
    per = {x: {q: float(np.mean(v)) for q, v in d.items()} for x, d in per.items()}
    for x in xs:
        v = np.array(list(per[x].values()))
        print(f"{x}: ir_max {v.mean():.3f} +/- {v.std(ddof=1) / len(v) ** .5:.3f} (standard error over {len(v)} prompts)")
    if "fk4" in per and "bon4" in per:
        q = sorted(set(per["fk4"]) & set(per["bon4"]))
        d = np.array([per["fk4"][c] - per["bon4"][c] for c in q])
        b = np.random.default_rng(0).choice(d, (20000, len(d))).mean(1)
        worst = min(q, key=lambda c: per["fk4"][c] - per["bon4"][c])
        print(f"fk4 - bon4 paired: {d.mean():+.4f} +/- {d.std(ddof=1) / len(d) ** .5:.4f}, {int((d > 0).sum())}/{len(d)} won, "
              f"bootstrap 95 % [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}], worst prompt {worst} "
              f"{per['fk4'][worst] - per['bon4'][worst]:+.2f}")
    # the spread of a one-seed, 100-prompt mean, read from the spread across seeds within each prompt
    # pooled over the prompts (about 200 degrees of freedom, where the three seed means give 2)
    def pooled(groups):
        v = [np.var(g, ddof=1) for g in groups if len(g) > 1]
        return float(np.sqrt(np.mean(v) / len(v)))
    by = {x: {} for x in xs}
    for r in runs:
        by[r["sampler"]].setdefault(r["prompt_id"], {})[r["seed"]] = r["ir_max"]
    # measured (a mean over the seeds) minus paper, in units of its own standard deviation: the
    # paper's value counts as one seed, or as a mean over as many seeds as ours
    n_s = len(seeds)
    for i in order:
        ref = paper.get(xs[i])
        s1 = pooled([list(d.values()) for d in by[xs[i]].values()])
        if ref:
            d = ir[i] - ref[0]
            print(f"{xs[i]}: pooled seed spread of a one-seed mean {s1:.3f}; measured - paper {d:+.3f}, "
                  f"{d / (s1 * (1 / n_s + 1) ** .5):+.1f} of its standard deviation if Table 1 is one seed, "
                  f"{d / (s1 * (2 / n_s) ** .5):+.1f} if it averages {n_s}")
    if "fk4" in by and "bon4" in by and "fk4" in paper and "bon4" in paper:
        g = [[by["fk4"][q][s] - by["bon4"][q][s] for s in by["fk4"][q] if s in by["bon4"][q]]
             for q in by["fk4"] if q in by["bon4"]]
        s1, n_seeds = pooled(g), len(seeds)
        s3 = s1 / n_seeds ** .5
        gap = paper["fk4"][0] - paper["bon4"][0] - float(np.mean([np.mean(x) for x in g]))
        print(f"fk4 - bon4: pooled seed spread {s1:.3f} for one seed, {s3:.3f} for a {n_seeds}-seed mean; "
              f"the gap to the paper, {gap:.3f}, is {gap / np.hypot(s3, s1):.1f} of its standard deviation if Table 1 is one seed, "
              f"{gap / np.hypot(s3, s3):.1f} if it averages {n_seeds}")
    for sd in sorted({r["seed"] for r in runs}):
        f = {r["prompt_id"]: r["ir_max"] for r in runs if r["sampler"] == "fk4" and r["seed"] == sd}
        b = {r["prompt_id"]: r["ir_max"] for r in runs if r["sampler"] == "bon4" and r["seed"] == sd}
        d = np.array([f[c] - b[c] for c in sorted(set(f) & set(b))])
        if len(d) > 1:
            print(f"fk4 - bon4 at seed {sd}: {d.mean():+.3f} +/- {d.std(ddof=1) / len(d) ** .5:.3f} ({len(d)} prompts)")
    t = dict(zip(xs, seconds))
    if "fk4" in t and "bon4" in t:
        # the rows match, the wall clock does not: what best-of-N costs the time of one FK run
        print(f"At equal wall clock FK buys best-of-{4 * t['fk4'] / t['bon4']:.2f} "
              f"({t['fk4']:.2f} s against {t['bon4']:.2f} s per run).")
        # a few best-of-4 runs took 82 to 103 s (another job on the card): the medians resist them
        med = {x: float(np.median([r["seconds"] for r in runs if r["sampler"] == x])) for x in ("fk4", "bon4")}
        slow = sum(r["seconds"] > 75 for r in runs if r["sampler"] == "bon4")
        print(f"On medians: {med['fk4']:.1f} s against {med['bon4']:.1f} s per run, best-of-{4 * med['fk4'] / med['bon4']:.2f} "
              f"({slow} of the bon4 runs took over 75 s).")
    # the mean of the four, paired per prompt (seeds averaged first)
    mean4 = {x: {} for x in xs}
    for r in runs:
        mean4[r["sampler"]].setdefault(r["prompt_id"], []).append(float(np.mean(r["ir"])))
    mean4 = {x: {q: float(np.mean(v)) for q, v in d.items()} for x, d in mean4.items()}
    if "fk4" in mean4 and "bon4" in mean4:
        q = sorted(set(mean4["fk4"]) & set(mean4["bon4"]))
        d = np.array([mean4["fk4"][c] - mean4["bon4"][c] for c in q])
        print(f"mean of the four, fk4 - bon4 paired: {d.mean():+.3f} +/- {d.std(ddof=1) / len(d) ** .5:.3f} ({len(d)} prompts)")
    # HPS as the released evaluation reads it (fks_utils.do_eval): the best HPS of the k images, not
    # the HPS of the image ImageReward picks; per prompt, seeds averaged first
    for x in xs:
        g = {}
        for r in runs:
            if r["sampler"] == x:
                g.setdefault(r["prompt_id"], []).append(max(r["hps"]))
        v = np.array([np.mean(h) for h in g.values()])
        ref = paper.get(x)
        spread = sigma_seeds([dict(r, hps_max=max(r["hps"])) for r in runs if r["sampler"] == x], "hps_max")
        print(f"{x}: HPS best of k {v.mean():.4f} +/- {v.std(ddof=1) / len(v) ** .5:.4f} (standard error, {len(v)} prompts),"
              f" seed spread {spread:.4f}" + (f", paper {ref[1]:.3f}, diff {v.mean() - ref[1]:+.4f}" if ref else ""))
    if args.out:
        Path(args.out).write_text(text)
        print(args.out)


if __name__ == "__main__":
    main()
