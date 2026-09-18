"""Sample grid from a sweep JSON: one row per (potential, lambda) at one seed,
free samples on top, every image scored by the evaluator B.

From the root: `python scripts/plot_grid.py --json results/X.json --judge ... --target 1`.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.patheffects
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from figstyle import INK, INK_LIGHT
from smc.classifier import load

ROOT = Path(__file__).resolve().parent.parent


def row(ax, images, scores, label, mean_score, thumb):
    x = F.interpolate(images, size=thumb, mode="area") if images.shape[-1] != thumb else images
    img = ((x.permute(0, 2, 3, 1).numpy() + 1) / 2).clip(0, 1)
    n, h, w = img.shape[0], img.shape[1], img.shape[2]
    gap = max(2, thumb // 32)
    sheet = np.ones((h + 2 * gap, n * (w + gap) + gap, 3))
    for i in range(n):
        sheet[gap:gap + h, gap + i * (w + gap):gap + i * (w + gap) + w] = img[i]
    ax.imshow(sheet, interpolation="nearest")
    for i in range(n):
        ax.text(gap + i * (w + gap) + 2, gap + h * 0.14, f"{scores[i]:.2f}", fontsize=6,
                color="white", family="monospace",
                path_effects=[matplotlib.patheffects.withStroke(linewidth=1.4, foreground="black")])
    ax.set_ylabel(f"{label}\nmean p_B = {mean_score:.2f}", rotation=0, ha="right", va="center",
                  fontsize=9, color=INK, labelpad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True)
    p.add_argument("--judge", required=True, help="evaluator checkpoint (B)")
    p.add_argument("--target", type=int, default=1)
    p.add_argument("--seed", type=int, default=None, help="default: the first seed of the sweep")
    p.add_argument("--thumb", type=int, default=96)
    p.add_argument("--title", default="")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    meta = json.loads(Path(args.json).read_text())
    images = torch.load(ROOT / meta["images"], map_location="cpu")
    seed = args.seed if args.seed is not None else meta["seeds"][0]
    judge = load(args.judge, "cpu")

    keys = [("free (best-of-k)", f"best_of_n_seed{seed}")]
    for r in meta["runs"]:
        if r["method"] == "fk" and r["seed"] == seed and r["lam"] > 0:
            keys.append((f"{r['potential']}, λ={r['lam']:g}",
                         f"{r['potential']}_{r['resampler']}_lam{r['lam']:g}_seed{seed}"))
    keys = [(l, k) for l, k in keys if k in images]

    fig, axes = plt.subplots(len(keys), 1, figsize=(13, 1.0 * len(keys) + 0.9))
    with torch.no_grad():
        for ax, (label, key) in zip(np.atleast_1d(axes), keys):
            x = images[key].float()
            pb = judge(x.clamp(-1, 1)).softmax(1)[:, args.target]
            row(ax, x, pb.tolist(), label, pb.mean().item(), args.thumb)
            print(f"{label:22s} mean p_B={pb.mean():.3f}   p_B>0.5: {int((pb > 0.5).sum()):2d}/{len(pb)}")

    title = args.title or f"{Path(args.json).stem} — seed {seed}, scored by B"
    fig.suptitle(title, fontsize=12, color=INK, x=0.012, ha="left", y=0.995)
    fig.text(0.012, 0.004, f"reward {meta.get('reward')}  ·  {meta.get('sampler', 'ddpm')} "
             f"T={meta['T']} eta={meta.get('eta')}  ·  weights {meta['weights']}",
             fontsize=8, color=INK_LIGHT, family="monospace")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out = Path(args.out) if args.out else ROOT / "figures" / f"grid_{Path(args.json).stem}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
