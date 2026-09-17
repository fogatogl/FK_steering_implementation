"""Figure 3: the 16 final particles of one seed, free vs FK red vs FK classifier.

Every image is scored by the evaluator B (p_B(cat)), whatever reward guided
it: the same judge on every row. The red-reward rows come from the first
sweep (raw weights), the others from the classifier sweep (EMA weights) —
qualitative comparison only.

From the root: `python scripts/plot_fig3.py`.
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

from figstyle import INK, INK_LIGHT
from smc.classifier import load

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = Path("/home/onyxia/work/ddpm/weights")


def row(ax, images, scores, label, mean_score):
    img = ((images.permute(0, 2, 3, 1).numpy() + 1) / 2).clip(0, 1)
    n, h, w = img.shape[0], img.shape[1], img.shape[2]
    gap = 2
    sheet = np.ones((h + 2 * gap, n * (w + gap) + gap, 3))
    for i in range(n):
        sheet[gap:gap + h, gap + i * (w + gap):gap + i * (w + gap) + w] = img[i]
    ax.imshow(sheet, interpolation="nearest")
    for i in range(n):
        ax.text(gap + i * (w + gap) + 1, gap + 5, f"{scores[i]:.2f}", fontsize=5.5,
                color="white", family="monospace",
                path_effects=[matplotlib.patheffects.withStroke(linewidth=1.4, foreground="black")])
    ax.set_ylabel(f"{label}\nmean p_B(cat) = {mean_score:.2f}", rotation=0, ha="right",
                  va="center", fontsize=9, color=INK, labelpad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--red", default=str(ROOT / "samples" / "sweep_lambda.pt"))
    p.add_argument("--classifier", default=str(ROOT / "samples" / "sweep_lambda_classifier.pt"))
    p.add_argument("--judge", default=str(WEIGHTS / "classifier_resnet18_seed1.pt"))
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--target", type=int, default=3)
    p.add_argument("--out", default=str(ROOT / "figures" / "fig3_samples_by_reward.png"))
    args = p.parse_args()

    red = torch.load(args.red, map_location="cpu")
    cls = torch.load(args.classifier, map_location="cpu")
    judge = load(args.judge, "cpu")
    rows = [
        ("free (best-of-16)", cls[f"best_of_n_seed{args.seed}"]),
        ("FK, red reward, λ=2", red[f"difference_systematic_lam2_seed{args.seed}"]),
        ("FK, red reward, λ=8", red[f"difference_systematic_lam8_seed{args.seed}"]),
        ("FK, classifier, λ=1", cls[f"difference_systematic_lam1_seed{args.seed}"]),
        ("FK, classifier, λ=2", cls[f"difference_systematic_lam2_seed{args.seed}"]),
        ("FK, classifier, λ=4", cls[f"difference_systematic_lam4_seed{args.seed}"]),
    ]

    fig, axes = plt.subplots(len(rows), 1, figsize=(13, 1.35 * len(rows) + 0.9))
    summary = {}
    with torch.no_grad():
        for ax, (label, x) in zip(axes, rows):
            pb = judge(x.clamp(-1, 1)).softmax(1)[:, args.target]
            row(ax, x, pb.tolist(), label, pb.mean().item())
            summary[label] = {"mean_pB": pb.mean().item(), "n_cat": int((pb > 0.5).sum())}

    fig.suptitle(f"Figure 3 — the 16 final particles (seed {args.seed}), scored by the evaluator B",
                 fontsize=12.5, color=INK, x=0.012, ha="left", y=0.995)
    fig.text(0.012, 0.004,
             "number on each image = p_B(cat) from the ResNet-18 evaluator, never the guiding reward  ·  "
             "red rows: raw weights (first sweep)  ·  other rows: EMA weights",
             fontsize=8, color=INK_LIGHT, family="monospace")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    for label, s in summary.items():
        print(f"{label:24s} mean p_B(cat)={s['mean_pB']:.3f}  images with p_B>0.5: {s['n_cat']:2d}/16")
    print(out)


if __name__ == "__main__":
    main()
