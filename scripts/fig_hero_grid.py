"""F1: on the same initial noise, the free sample, best-of-4 and FK differ by what the prompt asked for.

Reads data/visual_selection.json (the prompts chosen by rule, docs/visual_selection.md, and
the session they were ranked on, its `source`), that session's images/index.json (the saved
finals) and its run records (the rewards those images must match). One row per prompt, three images: lam0 slot 0
(one free sample), the best lam0 slot (best-of-4), the best ctl slot (FK, paper setting),
all four x_T shared. ImageReward under each image, and HPS v2.1 when the session's
hps_finals.json exists (collapse_lab/w_hps_finals.py), nothing else. An image is drawn only
if index.json's ir equals the record's ir for that arm, prompt and slot within 5e-4;
prompts without images are printed as the GPU list.

    python scripts/fig_hero_grid.py            # main: the three F1 prompts -> figures/f1_hero_grid
    python scripts/fig_hero_grid.py --appendix # the ten of W1        -> figures/f1_hero_grid_appendix
"""
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "collapse_lab" / "out"
COLS = [("lam0", "free sample"), ("lam0", "best-of-4"), ("ctl", "FK, paper setting")]
TOL = 5e-4


def load():
    sel = json.loads((ROOT / "data" / "visual_selection.json").read_text())
    src = ROOT / sel.get("source", "collapse_lab/out/probe_C.json")
    runs = json.loads(src.read_text())["runs"]
    C = {(r["arm"], r["prompt_id"]): r for r in runs}
    idx_path = src.parent / "images" / "index.json"
    index = json.loads(idx_path.read_text()) if idx_path.exists() else []
    img = {(e["arm"].split("_")[0], e["prompt_id"], e["slot"]): e for e in index}
    text = {e["id"]: e["prompt"] for e in json.loads((ROOT / "data" / "imagereward-benchmark-prompts.json").read_text())}
    hps_path = src.parent / "hps_finals.json"
    hps = json.loads(hps_path.read_text()) if hps_path.exists() else {}
    return sel, C, img, text, src.parent, hps


def pick(C, img, pid, base, hps):
    """The three (path, ir) cells of a row, or None if an image is missing or does not match."""
    cells = []
    for j, (arm, _) in enumerate(COLS):
        run = C[(arm, pid)]
        slot = 0 if j == 0 else max(range(run["k"]), key=lambda s: run["ir"][s])
        e = img.get((arm, pid, slot))
        if e is None or abs(e["ir"] - run["ir"][slot]) > TOL:
            return None
        cells.append((base / e["path"], run["ir"][slot], hps.get(arm, {}).get(pid, [None] * run["k"])[slot]))
    return cells


def main(appendix):
    sel, C, img, text, base, hps = load()
    check = {e["id"]: e["check"] for e in json.loads((ROOT / "data" / "visual_pool.json").read_text())["eligible"]}
    pids = sel["W1"] if appendix else sel["F1"]
    rows, missing = [], []
    for pid in pids:
        cells = pick(C, img, pid, base, hps)
        (rows if cells else missing).append((pid, cells))
    if missing:
        print("GPU list (no matching images yet):", " ".join(p for p, _ in missing))
    if not rows:
        print("nothing to draw")
        return
    fig, axes = fs.figure(2.4 * len(rows), ncols=3, nrows=len(rows), squeeze=False)
    for i, (pid, cells) in enumerate(rows):
        for j, (path, ir, h) in enumerate(cells):
            ax = axes[i][j]
            ax.imshow(Image.open(path).convert("RGB").resize((256, 256)))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_xlabel(f"IR {ir:+.2f}" + (f"   HPS {h:.3f}" if h is not None else ""))
            if i == 0:
                ax.annotate(COLS[j][1], (0.5, 1.02), xycoords="axes fraction", ha="center", va="bottom", fontsize=9)
        # the pool's one-line note of what to check, wrapped: the full prompt is in the caption table (A.3)
        axes[i][0].set_ylabel(textwrap.fill(check.get(pid, text[pid]), 22), fontsize=9)
    png, svg = fs.save(fig, ROOT / "figures" / ("f1_hero_grid_appendix" if appendix else "f1_hero_grid"))
    print("wrote", png, svg, "rows:", " ".join(p for p, _ in rows))


if __name__ == "__main__":
    main("--appendix" in sys.argv)
