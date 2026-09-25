"""F2: FK Steering runs k particles from noise to image and, at a few scheduled steps, reweights them by the reward and resamples, so a low-weight particle dies and a high-weight one is cloned, and the surviving lineage's colour reaches the finals.

A schematic, not a measurement: four particle trajectories drawn as smooth random paths
(fixed rng seed) from x_T on the left to x_0 on the right, each in the colour of its root
(figstyle.ROOT_COLORS). The five vertical dotted lines are the scheduled steps, placed at
schedule_t / steps read from the fk4 row of results/sd_baseline.json (the only numbers in
the drawing). At the first scheduled step the weights are shown as dot sizes and one
resampling is drawn: the lightest particle stops (grey stub), the heaviest forks into two
paths of its colour, one of them taking the freed slot. Text in the image: x_T, noise, x_0,
image, resample.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

import figstyle
from figstyle import ROOT_COLORS, GREY, GREY_LIGHT, INK, INK_LIGHT

ROOT = Path(__file__).resolve().parent.parent
K = 4
RESAMPLE_AT = 0          # index in the schedule where the resampling is drawn
WEIGHTS = [0.55, 0.05, 0.28, 0.12]   # schematic normalised weights at that step, root order
# the lightest particle sits next to the heaviest, so the clone moves one slot without crossing
N_PTS = 250


def schedule_positions():
    """Scheduled steps as fractions of the run, left = x_T, from the fk4 row of the baseline."""
    runs = json.loads((ROOT / "results" / "sd_baseline.json").read_text())["runs"]
    r = next(r for r in runs if r["sampler"] == "fk4")
    steps = r["steps"]
    # schedule_t counts from the noise end; the drawing runs noise (left) to image (right)
    return sorted(1 - t / steps for t in r["schedule_t"]), steps


def path(rng, x, y0, y1, wobble):
    """A smooth path from y0 at x[0] to y1 at x[-1], noise that fades as the image forms."""
    u = (x - x[0]) / (x[-1] - x[0])
    base = y0 + (y1 - y0) * (3 * u ** 2 - 2 * u ** 3)   # eased, so a fork opens smoothly
    noise = np.cumsum(rng.normal(size=len(x)))
    noise -= np.linspace(noise[0], noise[-1], len(x))
    return base + wobble * noise / np.abs(noise).max() * (1 - u)


def main():
    sched, steps = schedule_positions()
    x_res = sched[RESAMPLE_AT]
    rng = np.random.default_rng(3)
    slots = np.linspace(0.8, -0.8, K)

    fig, ax = figstyle.figure(figstyle.WIDTH_IN / 2.2)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-1.3, 1.3)
    ax.axis("off")
    for s in sched:
        ax.axvline(s, color=GREY_LIGHT, ls=":", lw=1.2, zorder=1)

    heaviest, lightest = int(np.argmax(WEIGHTS)), int(np.argmin(WEIGHTS))
    x1 = np.linspace(0, x_res, N_PTS)
    x2 = np.linspace(x_res, 1, N_PTS)
    for i in range(K):
        y = path(rng, x1, slots[i], slots[i] + rng.normal(scale=0.06), 0.2)
        ax.plot(x1, y, color=ROOT_COLORS[i], lw=1.8, zorder=3)
        ax.scatter([x_res], [y[-1]], s=900 * WEIGHTS[i], color=ROOT_COLORS[i], zorder=5,
                   edgecolor="#FFFFFF", linewidth=0.8)
        if i == lightest:
            # killed: a short grey stub, then nothing
            ax.plot([x_res, x_res + 0.03], [y[-1], y[-1]], color=GREY, lw=1.8, zorder=3)
            ax.scatter([x_res + 0.03], [y[-1]], s=14, color=GREY, zorder=4)
        else:
            ax.plot(x2, path(rng, x2, y[-1], slots[i], 0.14), color=ROOT_COLORS[i], lw=1.8, zorder=3)
        if i == heaviest:
            y_fork = y[-1]
    # the clone of the heaviest particle takes the freed slot
    ax.plot(x2, path(rng, x2, y_fork, slots[lightest], 0.14), color=ROOT_COLORS[heaviest], lw=1.8, zorder=3)

    ax.text(0, 1.12, "$x_T$", ha="center", va="bottom", fontsize=11, color=INK)
    ax.text(0, -1.12, "noise", ha="center", va="top", fontsize=9, color=INK_LIGHT)
    ax.text(1, 1.12, "$x_0$", ha="center", va="bottom", fontsize=11, color=INK)
    ax.text(1, -1.12, "image", ha="center", va="top", fontsize=9, color=INK_LIGHT)
    ax.text(x_res, -1.12, "resample", ha="center", va="top", fontsize=9, color=INK)

    png, svg = figstyle.save(fig, ROOT / "figures" / "f2_algorithm")
    print(f"scheduled steps at t = {[round((1 - s) * steps) for s in sched]} of {steps}, drawn at x = {[round(s, 2) for s in sched]}")
    print(png, svg)


if __name__ == "__main__":
    main()
