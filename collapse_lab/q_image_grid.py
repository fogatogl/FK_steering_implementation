"""F5: on one prompt, the four FK images at the paper's setting share one x_T, the free sampler's and floor + lambda = 2's do not.

One row per arm (free sampler, FK paper setting, floor + lambda = 2), the four final images
of the slots, a 6 px frame in the colour of the image's root x_T (figstyle.ROOT_COLORS by
root_slots, the F4 tree's colours), the arm's label as a left column; nothing else in the image.

The prompt list comes from data/visual_selection.json and from nowhere else: main F5, the
ten of W1 for --appendix. Only the prompts whose images exist in out/images/index.json are
drawn; each image is drawn only if index.json's ir equals probe_C.json's ir[slot] for that
arm (the `_b1` suffix stripped) and prompt within 5e-4, otherwise it is refused. The selected
prompts without images are printed as the GPU list.

  (default)  : F5 prompt                          -> figures/f5_root_grid
  --appendix : the ten prompts of W1              -> figures/f5_root_grid_appendix
  --b1       : the six prompts of the B1 session, those present in index.json today,
               same check against probe_C.json   -> figures/f5_root_grid_b1
  --choose   : prints the six prompt_id of the pre-registered rule (docs/protocol_sd.md, 22/09)
               from out/probe.json at n = 40: the two prompts ranked 20 and 21 on ctl's ir_max;
               the two where ctl beats bon4 the most; among those where floor2 keeps four
               lineages, the two with floor2's highest ir_max. Ties by prompt_id. Consumed
               by nuit2.sh; its output does not change.
Nothing is computed here: the ir come from index.json and probe_C.json, the roots from root_slots.
"""
import argparse
import json
import sys
import textwrap
from pathlib import Path

LAB = Path(__file__).resolve().parent
ROOT = LAB.parent
OUT = LAB / "out"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(LAB))
from commun import bon4

ARMS = ["lam0", "ctl", "floor2"]
TOL = 5e-4
FRAME_PX = 6
CELL_PX = 256


def par_bras():
    runs = json.loads((OUT / "probe.json").read_text())["runs"]
    par = {}
    for r in runs:
        par.setdefault(r["arm"], {})[r["prompt_id"]] = r
    return par


def choisir(par):
    bon = bon4()
    ctl = par["ctl"]
    ids = sorted(ctl)[:40]
    rang = sorted(ids, key=lambda c: (ctl[c]["ir_max"], c))
    medians = rang[19:21]
    gagne = sorted(ids, key=lambda c: (-(ctl[c]["ir_max"] - bon[c]["ir_max"]), c))
    gagne = [c for c in gagne if c not in medians][:2]
    f2 = par["floor2"]
    quatre = sorted((c for c in ids if c in f2 and f2[c]["n_lineages"] == 4 and c not in medians + gagne),
                    key=lambda c: (-f2[c]["ir_max"], c))[:2]
    return medians + gagne + quatre


def load(src):
    C = {(r["arm"], r["prompt_id"]): r for r in json.loads(src.read_text())["runs"]}
    idx_path = src.parent / "images" / "index.json"
    index = json.loads(idx_path.read_text()) if idx_path.exists() else []
    img = {(e["arm"].split("_")[0], e["prompt_id"], e["slot"]): e for e in index}
    return C, img, index


def cells(C, img, pid, base):
    """The 3 x 4 (path, root) cells of a prompt, or None if an image is missing or does not match."""
    rows = []
    for arm in ARMS:
        run = C.get((arm, pid))
        if run is None:
            return None
        row = []
        for slot in range(run["k"]):
            e = img.get((arm, pid, slot))
            if e is None:
                return None
            if abs(e["ir"] - run["ir"][slot]) > TOL:
                print(f"refused {e['path']}: index ir {e['ir']:+.4f}, record ir {run['ir'][slot]:+.4f}")
                return None
            row.append((base / e["path"], run["root_slots"][slot]))
        rows.append(row)
    return rows


def grille(blocks, out):
    """blocks: [(pid, rows)]; one block of len(ARMS) rows per prompt, a spacer row between blocks."""
    import figstyle as fs
    from PIL import Image
    n_img = len(ARMS)
    nrows = len(blocks) * n_img + (len(blocks) - 1)
    ratios = []
    for b in range(len(blocks)):
        ratios += [1.0] * n_img + ([0.12] if b < len(blocks) - 1 else [])
    fig, axes = fs.figure(fs.WIDTH_IN * sum(ratios) / 4 * 0.86, ncols=4, nrows=nrows, squeeze=False,
                          gridspec_kw={"height_ratios": ratios, "wspace": 0.02, "hspace": 0.02})
    lw_pt = FRAME_PX * 72 / fs.DPI
    for b, (pid, rows) in enumerate(blocks):
        base = b * (n_img + 1)
        for a, row in enumerate(rows):
            for j, (path, root) in enumerate(row):
                ax = axes[base + a][j]
                ax.imshow(Image.open(path).convert("RGB").resize((CELL_PX, CELL_PX)))
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_edgecolor(fs.ROOT_COLORS[root]); s.set_linewidth(lw_pt)
            axes[base + a][0].set_ylabel(textwrap.fill(fs.arm_label(ARMS[a]), 18), rotation=0, ha="right", va="center")
        if b < len(blocks) - 1:
            for ax in axes[base + n_img]:
                ax.axis("off")
    return fs.save(fig, out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--choose", action="store_true")
    p.add_argument("--appendix", action="store_true")
    p.add_argument("--b1", action="store_true")
    args = p.parse_args()
    if args.choose:
        print(" ".join(choisir(par_bras())))
        return
    sel = json.loads((ROOT / "data" / "visual_selection.json").read_text())
    # the b1 images sit in out/images and match session C; the selection names its own session
    src = OUT / "probe_C.json" if args.b1 else ROOT / sel.get("source", "collapse_lab/out/probe_C.json")
    C, img, index = load(src)
    if args.b1:
        pids, out = sorted({e["prompt_id"] for e in index}), "f5_root_grid_b1"
    elif args.appendix:
        pids, out = sel["W1"], "f5_root_grid_appendix"
    else:
        pids, out = [sel["F5"]], "f5_root_grid"
    blocks, missing = [], []
    for pid in pids:
        rows = cells(C, img, pid, src.parent)
        (blocks if rows else missing).append((pid, rows))
    if missing:
        print("GPU list (no matching images yet):", " ".join(pid for pid, _ in missing))
    if not blocks:
        print("nothing to draw")
        return
    png, svg = grille(blocks, ROOT / "figures" / out)
    for pid, rows in blocks:
        print(pid, "roots per arm:", " ".join(f"{arm}={[root for _, root in row]}" for arm, row in zip(ARMS, rows)))
    print("wrote", png, svg, f"| ir from images/index.json checked against {src.relative_to(ROOT)} within", TOL)


if __name__ == "__main__":
    main()
