"""F3: best-of-4 lands on the paper's ImageReward and FK at k = 4 sits under it, while HPS v2.1, the judge, barely moves for any sampler.

Left panel ImageReward of the best image, right panel the best HPS v2.1 of the k images (the
statistic the released evaluation, fks_utils.do_eval, reports), one bar per sampler (one sample, best-of-4,
FK at the paper setting) read from results/sd_baseline.json rows k1, bon4, fk4. Each bar is
the mean over the 100 prompts of the prompt's value averaged over its three seeds first, the
error bar the spread of a one-seed, 100-prompt mean from seed to seed (make_table_sd.pooled), the
unit in which the post compares a row with the paper. The short grey line across each bar is the paper's value from
results/paper_table1_sd15.json. Both panels start at 0: a zoomed HPS axis would turn a
0.01 difference, inside the paper's own spread, into a visible effect.

From the root: `python scripts/plot_fig4_sd.py`.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

import figstyle
from figstyle import ARM_COLOR, ARM_LABEL, GREY_DARK, aggregate, dress, load_runs
from make_table_sd import pooled

ROOT = Path(__file__).resolve().parent.parent
SAMPLERS = ["k1", "bon4", "fk4"]
COLORS = {"k1": ARM_COLOR["k1"], "bon4": ARM_COLOR["bon4"], "fk4": ARM_COLOR["ctl"]}
LABELS = {"k1": ARM_LABEL["k1"], "bon4": ARM_LABEL["bon4"], "fk4": "FK, paper setting"}
BAR_W = 0.6
LINE_W = 0.9   # the paper's line is wider than the bar so it reads on a grey bar too


def per_prompt(raw):
    """One record per (sampler, prompt): the three seeds of a prompt averaged first."""
    groups = {}
    for r in raw:
        groups.setdefault((r["sampler"], r["prompt_id"]), []).append(r)
    return [{"sampler": s, "ir_max": np.mean([x["ir_max"] for x in g]),
             "hps_max": np.mean([max(x["hps"]) for x in g])}
            for (s, _), g in groups.items()]


def panel(ax, means, sems, paper, ylabel):
    x = np.arange(len(SAMPLERS))
    ax.bar(x, means, width=BAR_W, color=[COLORS[s] for s in SAMPLERS], zorder=3,
           yerr=sems, capsize=4, ecolor=figstyle.INK, error_kw={"elinewidth": 1.0})
    ax.hlines(paper, x - LINE_W / 2, x + LINE_W / 2, color=GREY_DARK, linewidth=1.6, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[s].replace(" (", "\n(") for s in SAMPLERS])
    ax.set_xlim(-0.6, len(SAMPLERS) - 0.4)
    ax.set_ylim(0, None)
    ax.set_ylabel(ylabel)
    dress(ax, grid="y")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", nargs="+", default=[str(ROOT / "results" / "sd_baseline.json")])
    p.add_argument("--paper", default=str(ROOT / "results" / "paper_table1_sd15.json"))
    p.add_argument("--out", default=str(ROOT / "figures" / "f3_reproduction"))
    args = p.parse_args()

    raw = load_runs(args.json)
    runs = per_prompt(raw)
    xs, ir, _, cnt = aggregate(runs, "sampler", "ir_max")
    _, hps, _, _ = aggregate(runs, "sampler", "hps_max")
    idx = [list(xs).index(s) for s in SAMPLERS]
    ir, hps, cnt = (v[idx] for v in (ir, hps, cnt))

    def spread(s, value):
        seeds = {}
        for r in raw:
            if r["sampler"] == s:
                seeds.setdefault(r["prompt_id"], []).append(value(r))
        return pooled(list(seeds.values()))
    ir_sp = np.array([spread(s, lambda r: r["ir_max"]) for s in SAMPLERS])
    hps_sp = np.array([spread(s, lambda r: max(r["hps"])) for s in SAMPLERS])

    paper = {r["sampler"]: r for r in json.loads(Path(args.paper).read_text())["rows"]}
    paper_ir = [paper[s]["ir_max"] for s in SAMPLERS]
    paper_hps = [paper[s]["hps_at_ir_max"] for s in SAMPLERS]   # the paper's HPS column, whatever the key says

    fig, axes = figstyle.figure(3.4, ncols=2)
    panel(axes[0], ir, ir_sp, paper_ir, "ImageReward, best of the k images")
    panel(axes[1], hps, hps_sp, paper_hps, "HPS v2.1, best of the k images")

    Path(args.out).parent.mkdir(exist_ok=True)
    png, svg = figstyle.save(fig, args.out)
    for i, s in enumerate(SAMPLERS):
        print(f"{s:5s} IR={ir[i]:+.4f}, one-seed spread {ir_sp[i]:.4f} (paper {paper_ir[i]:.3f})  "
              f"HPS={hps[i]:.4f}, one-seed spread {hps_sp[i]:.4f} (paper {paper_hps[i]:.3f})  "
              f"({cnt[i]} prompts)")
    print(png, svg)


if __name__ == "__main__":
    main()
