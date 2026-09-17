"""What the figure scripts share."""
import json
from pathlib import Path

import numpy as np

INK = "#2b2b2b"
INK_LIGHT = "#8a8a8a"
BLUE = "#4a6fa5"
RED = "#c0563a"
GREEN = "#6b8f47"


def dress(ax, grid="both"):
    ax.grid(axis=grid, color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cccccc")
    ax.tick_params(colors=INK_LIGHT, labelsize=9)


def load_runs(paths):
    runs = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        runs += d["runs"] if isinstance(d, dict) else d
    return runs


def aggregate(runs, key_x, key_y):
    """(sorted x, mean per x, std per x, number of seeds per x)."""
    xs = sorted({r[key_x] for r in runs})
    mean, std, count = [], [], []
    for x in xs:
        v = np.array([r[key_y] for r in runs if r[key_x] == x], dtype=float)
        mean.append(v.mean())
        std.append(v.std(ddof=1) if len(v) > 1 else 0.0)
        count.append(len(v))
    return np.array(xs), np.array(mean), np.array(std), np.array(count)
