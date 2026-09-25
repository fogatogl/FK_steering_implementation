"""F8 appendix: the weight collapse of F8 shows in the images too, pixel diversity of the finals falling with lambda at the three scales, and at SD scale it is already what four free draws reweighted by exp(lambda * ImageReward) would give.

Left panel: pixel diversity of the k finals (mean pairwise RMSE at 64 x 64 in [0, 1], the
div_pix of scripts/run_sd_baseline.py) relative to its value at lambda = 0, one line per
scale, direct labels. CIFAR and CelebA from samples/sweep_lambda_classifier.pt and
samples/sweep_lambda_hub_classifier.pt (keys difference_systematic_lam{lam}_seed{seed},
mean and standard error over 3 seeds); SD from the div_pix field of
collapse_lab/out/probe.json, arms lam0 / lam2 / ctl on their shared prompts.
Right panel: ESS / k of the weights exp(lambda * ir) over the four free draws of each bon4
prompt (results/sd_baseline.json, seed 2024), median and quartiles over the 100 prompts,
against lambda; the diamond is the measured ESS / k of fk4 at its first scheduled step
(t = 80, lambda = 10, results/sd_ref_fields100.json, ess_at_schedule[0]). The lambdas of
the curve are those present in the three sweeps. Requires the .pt files, only in the main
checkout (/home/onyxia/work/diffusion-models/samples).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle
from fig_three_scales import ROOT, SCALES, SD_ARMS, load, sem

SAMPLES = Path("/home/onyxia/work/diffusion-models/samples")
PT = {"sweep_lambda_classifier.json": "sweep_lambda_classifier.pt",
      "sweep_lambda_hub_classifier.json": "sweep_lambda_hub_classifier.pt"}
SCALE_COLOR = {"CIFAR 32 px": figstyle.BLUE, "CelebA 256 px": figstyle.GREEN, "SD 512 px": figstyle.ORANGE}


def div_pix(x):
    """x: (k, 3, H, W) in [-1, 1]. Mean pairwise RMSE at 64 x 64 in [0, 1]."""
    x = ((x + 1) / 2).clamp(0, 1)
    x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False, antialias=True)
    k = x.shape[0]
    d = [float(((x[i] - x[j]) ** 2).mean().sqrt()) for i in range(k) for j in range(i + 1, k)]
    return float(np.mean(d))


def diversity_sweep(json_name):
    """{lambda: (div_pix mean, standard error)} from a sweep file and its .pt of finals."""
    d = json.loads((ROOT / "results" / json_name).read_text())
    runs = [r for r in d["runs"] if r.get("method") != "best_of_n"]
    imgs = torch.load(SAMPLES / PT[json_name], map_location="cpu", weights_only=False)
    out = {}
    for lam in sorted({r["lam"] for r in runs}):
        keys = [f"difference_systematic_lam{lam:g}_seed{r['seed']}" for r in runs if r["lam"] == lam]
        dv = np.array([div_pix(imgs[c]) for c in keys if c in imgs])
        out[lam] = (dv.mean(), sem(dv))
    return out


def diversity_sd():
    runs = json.loads((ROOT / "collapse_lab" / "out" / "probe.json").read_text())["runs"]
    by_arm = {}
    for r in runs:
        by_arm.setdefault(r["arm"], {})[r["prompt_id"]] = r
    common = sorted(set.intersection(*(set(by_arm[a]) for a in SD_ARMS)))
    out = {}
    for arm in SD_ARMS:
        d = by_arm[arm]
        dv = np.array([d[c]["div_pix"] for c in common])
        out[d[common[0]]["lam"]] = (dv.mean(), sem(dv))
    return out


def ess_over_k(logw):
    w = np.exp(logw - logw.max())
    w /= w.sum()
    return 1 / (w ** 2).sum() / len(w)


def target(lams):
    """Median and quartiles over prompts of ESS / k for exp(lam * ir) on the bon4 draws."""
    base = json.loads((ROOT / "results" / "sd_baseline.json").read_text())["runs"]
    bon = [r for r in base if r["sampler"] == "bon4"]
    seed = min(r["seed"] for r in bon)
    bon = [r for r in bon if r["seed"] == seed]
    q = {l: np.percentile([ess_over_k(l * np.array(r["ir"])) for r in bon], [25, 50, 75]) for l in lams}
    ref = json.loads((ROOT / "results" / "sd_ref_fields100.json").read_text())["runs"]
    fk = [r for r in ref if r["sampler"] == "fk4"]
    fk80 = np.percentile([r["ess_at_schedule"][0] / r["n"] for r in fk], [25, 50, 75])
    lam_fk = fk[0]["lam"]
    return q, fk80, lam_fk, len(bon), len(fk)


def main():
    div = {}
    for name, json_name in SCALES:
        div[name] = diversity_sweep(json_name) if json_name else diversity_sd()
        free = div[name][min(div[name])][0]
        print(name)
        for lam, (m, s) in sorted(div[name].items()):
            print(f"  lambda {lam:4g}  div_pix = {m:.3f} +/- {s:.3f}  / free = {m / free:.2f}")
    lams = sorted({l for _, e, _ in load() for l in e} - {0.0})
    q, fk80, lam_fk, n_bon, n_fk = target(lams)
    print("target ESS / k, median: " + ", ".join(f"{l:g}: {q[l][1]:.2f}" for l in lams)
          + f"; fk4 at first scheduled step, lambda {lam_fk:g}: {fk80[1]:.2f} (n = {n_bon}, {n_fk})")

    fig, (a1, a2) = figstyle.figure(3.2, ncols=2)
    all_lams = sorted({l for e in div.values() for l in e})
    for name, e in div.items():
        ls = sorted(e)
        free = e[ls[0]][0]
        y = [e[l][0] / free for l in ls]
        a1.errorbar(ls, y, yerr=[e[l][1] / free for l in ls], color=SCALE_COLOR[name],
                    marker="o", ms=4, lw=1.4, capsize=2)
        figstyle.label_end(a1, ls[-1], y[-1], name, SCALE_COLOR[name])
    a1.axhline(1, color=figstyle.GREY_LIGHT, ls=":", lw=1)
    a1.set_xscale("symlog", linthresh=0.5)
    a1.set_xticks(all_lams)
    a1.set_xticklabels([f"{l:g}" for l in all_lams])
    a1.minorticks_off()
    a1.set_xlim(right=all_lams[-1] * 2.6)
    a1.set_ylim(0, 1.1)
    a1.set_xlabel("λ")
    a1.set_ylabel("pixel diversity of the finals / free")
    figstyle.dress(a1)

    med = [q[l][1] for l in lams]
    a2.fill_between(lams, [q[l][0] for l in lams], [q[l][2] for l in lams], color=figstyle.GREY_LIGHT, alpha=0.5, lw=0)
    # colours are the arms': the curve is bon4's free draws (GREY), the diamond ctl (GREY_DARK)
    a2.plot(lams, med, marker="o", ms=4, color=figstyle.ARM_COLOR["bon4"], lw=1.4)
    a2.annotate("4 free draws, reweighted", (lams[1], q[lams[1]][1]), xytext=(6, 6),
                textcoords="offset points", ha="left", va="bottom", color=figstyle.ARM_COLOR["bon4"], fontsize=9)
    a2.errorbar([lam_fk], [fk80[1]], yerr=[[fk80[1] - fk80[0]], [fk80[2] - fk80[1]]],
                fmt="D", color=figstyle.ARM_COLOR["ctl"], ms=6, capsize=3, zorder=4)
    a2.annotate("FK, first scheduled step", (lam_fk, fk80[0]), xytext=(0, -8),
                textcoords="offset points", ha="right", va="top", color=figstyle.ARM_COLOR["ctl"], fontsize=9)
    a2.set_xscale("symlog", linthresh=0.5)
    a2.set_xticks(lams)
    a2.set_xticklabels([f"{l:g}" for l in lams])
    a2.minorticks_off()
    a2.set_xlim(right=lams[-1] * 1.5)
    a2.set_ylim(0, 1.1)
    a2.set_xlabel("λ")
    a2.set_ylabel("ESS / k (median, quartiles)")
    figstyle.dress(a2)
    png, svg = figstyle.save(fig, ROOT / "figures" / "f8_three_scales_appendix")
    print(png, svg)


if __name__ == "__main__":
    main()
