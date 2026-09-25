"""F4: at the paper's setting the four final images descend from one x_T, with the floor and lambda = 2 they keep their own.

One panel per arm, same prompt: levels from top to bottom are x_T then each scheduled step;
an edge joins slot j at step m to its parent anc[m][j] at the previous step, its width the
normalised weight of that parent (logG_at_schedule), its colour the parent's root x_T
(figstyle.ROOT_COLORS, shared with the F5 grid). A node nobody descends from dies there
and is drawn as a light grey cross. Under the leaves, ir_max alone, under the leaf that
carries it.

Data: the session named by data/visual_selection.json["source"] (session D since 24/09, the
images F5 shows), then collapse_lab/out/probe_C.json, then collapse_lab/out/probe.json for R1
and for the `_b1` replays.
The default prompt is data/visual_selection.json["F5"].

    python collapse_lab/o_ancestry_fig.py                 # ctl, floor2      -> figures/f4_ancestry
    python collapse_lab/o_ancestry_fig.py --appendix      # lam0 ctl floor2 R1 -> figures/f4_ancestry_appendix
    python collapse_lab/o_ancestry_fig.py --pid <id> --arms lam0 ctl --out <stem>
"""
import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

LAB = Path(__file__).resolve().parent
ROOT = LAB.parent
sys.path.insert(0, str(ROOT / "scripts"))
import figstyle as fs

MAIN_ARMS = ["ctl", "floor2"]
APPENDIX_ARMS = ["lam0", "ctl", "floor2", "R1"]
# the session the selection was ranked on comes first, so F4 draws the run whose images F5 frames
_sel = json.loads((ROOT / "data" / "visual_selection.json").read_text())
SOURCES = [ROOT / _sel["source"]] * ("source" in _sel) + [LAB / "out" / "probe_C.json", LAB / "out" / "probe.json"]


def load():
    """(arm, prompt_id) -> run, probe_C first, then probe.json, then the `_b1` replay in probe.json."""
    par = {}
    for src in SOURCES:
        for r in json.loads(src.read_text())["runs"]:
            if "ancestors_at_schedule" not in r:
                continue
            key = (r["arm"], r["prompt_id"])
            if key not in par:
                par[key] = (r, src)
    for (arm, pid), v in list(par.items()):
        if arm.endswith("_b1") and (arm[:-3], pid) not in par:
            par[(arm[:-3], pid)] = v
    return par


def roots_per_level(anc):
    """anc: (M, k). For each level 0..M (0 = x_T), the root of every slot."""
    k = len(anc[0])
    levels = [list(range(k))]
    for m in range(len(anc)):
        prev = levels[-1]
        levels.append([prev[anc[m][j]] for j in range(k)])
    return levels


def draw(ax, r, arm, narrow=False):
    """narrow: more than two panels side by side, shorter tick labels and a wrapped arm label."""
    anc = r["ancestors_at_schedule"]
    lg = np.array(r["logG_at_schedule"])
    w = np.exp(lg - lg.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    ts = sorted(r["schedule_t"], reverse=True)
    k = r["k"]; M = len(anc)
    lev = roots_per_level(anc)
    for m in range(M):
        for j in range(k):
            parent = anc[m][j]
            # the weight that had the parent copied: the parent's weight at step m
            ax.plot([parent, j], [m, m + 1], color=fs.ROOT_COLORS[lev[m][parent]],
                    lw=0.6 + 4.5 * w[m][parent], alpha=0.85, zorder=1, solid_capstyle="round")
    for m in range(M + 1):
        for j in range(k):
            alive = m == M or any(anc[m][i] == j for i in range(k))
            if alive:
                ax.scatter(j, m, s=90, color=fs.ROOT_COLORS[lev[m][j]], edgecolor=fs.INK, lw=0.6, zorder=2)
            else:
                ax.scatter(j, m, s=40, color=fs.GREY_LIGHT, marker="x", lw=1.2, zorder=2)
    best = int(np.argmax(r["ir"]))
    ax.annotate(f"best image, IR {r['ir_max']:+.2f}", (best, M), xytext=(0, -14), textcoords="offset points",
                ha="center", va="top", color=fs.INK, fontsize=9)
    ax.set_yticks(range(M + 1))
    ax.set_yticklabels(["x_T"] + [f"t = {t}" for t in ts])
    ax.set_xticks(range(k)); ax.set_xticklabels([str(j) if narrow else f"slot {j}" for j in range(k)])
    ax.set_xlim(-0.5, k - 0.5)
    ax.set_ylim(M + 0.9, -0.5)
    ax.set_xlabel(textwrap.fill(fs.arm_label(arm), 18) if narrow else fs.arm_label(arm))
    fs.dress(ax, grid=None)
    ax.tick_params(length=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pid", default=None, help="prompt id; default data/visual_selection.json['F5']")
    p.add_argument("--arms", nargs="+", default=None)
    p.add_argument("--appendix", action="store_true", help=f"arms {' '.join(APPENDIX_ARMS)}, output _appendix")
    p.add_argument("--out", default=None, help="output stem (PNG and SVG), default figures/f4_ancestry[_appendix]")
    args = p.parse_args()
    sel_path = ROOT / "data" / "visual_selection.json"
    pid = args.pid or json.loads(sel_path.read_text())["F5"]
    arms = args.arms or (APPENDIX_ARMS if args.appendix else MAIN_ARMS)
    par = load()
    found = [a for a in arms if (a, pid) in par]
    missing = [a for a in arms if (a, pid) not in par]
    if missing:
        print(f"no run of {pid} with ancestors_at_schedule for: {' '.join(missing)}")
    if not found:
        raise SystemExit("nothing to draw")
    fig, axes = fs.figure(4.0, ncols=len(found), squeeze=False)
    for ax, a in zip(axes[0], found):
        r, src = par[(a, pid)]
        draw(ax, r, a, narrow=len(found) > 2)
        print(f"{a:7s} <- {src.relative_to(ROOT)}  n_lineages {r['n_lineages']}  ir_max {r['ir_max']:+.4f}  roots {r['root_slots']}")
    r0 = par[(found[0], pid)][0]
    print(f"prompt {pid}: {r0['prompt']}")
    out = Path(args.out) if args.out else ROOT / "figures" / ("f4_ancestry_appendix" if args.appendix else "f4_ancestry")
    png, svg = fs.save(fig, out)
    print("wrote", png, svg)


if __name__ == "__main__":
    main()
