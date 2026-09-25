"""What the figure scripts share: the palette, the arm labels, the export.

The rules are the editorial plan's, section 2.2. One colour per arm for the whole post,
Okabe-Ito (Wong, Nature Methods 2011); greys for the references; no title inside the
image; DejaVu Sans at 9 pt or more once the figure sits in a 700 px column, so the PNG
is exported at 1400 px wide; SVG next to it; transparent background, constrained layout
for the margins (a tight bbox would crop the width below 1400).
"""
import json
from pathlib import Path

import numpy as np

# ---- Okabe-Ito -------------------------------------------------------------------
BLACK = "#000000"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
YELLOW = "#F0E442"
BLUE = "#0072B2"
VERMILION = "#D55E00"
PINK = "#CC79A7"
GREY_DARK = "#333333"
GREY = "#999999"
GREY_LIGHT = "#BBBBBB"
GRID = "#E6E6E6"
SPINE = "#CCCCCC"
INK = "#2B2B2B"        # text and spines
INK_LIGHT = "#8A8A8A"  # tick labels
RED = VERMILION        # old name, kept for the scripts that import it
PALETTE = {BLACK, ORANGE, SKY, GREEN, YELLOW, BLUE, VERMILION, PINK,
           GREY_DARK, GREY, GREY_LIGHT, GRID, SPINE, INK, INK_LIGHT, "#FFFFFF"}

# ---- arms: one colour, one English label, for figures, grid frames and the demo ------
ARM_COLOR = {
    "ctl": GREY_DARK, "bon4": GREY, "lam0": GREY_LIGHT, "k1": GREY_LIGHT,
    "floor2": BLUE, "floor": SKY, "lam2": GREEN,
    "fadapt": ORANGE, "adapt": ORANGE, "late": PINK, "R0": VERMILION, "R1": VERMILION,
}
ARM_LABEL = {
    "ctl": "FK, paper setting (λ = 10)", "bon4": "best-of-4",
    "lam0": "free sampler", "k1": "free sampler (one sample)",
    "floor2": "floor + λ = 2", "floor": "floor (released code)", "lam2": "λ = 2",
    "fadapt": "adaptive λ", "adapt": "adaptive λ", "late": "no step at t = 80",
    "R0": "released code", "R1": "released choices in smc/",
    "thr05": "floor, resample if ESS < k/2", "rise": "floor + adaptive \u03bb, cap 100",
    "stat0": "floor, statistic form", "multi": "multinomial every step",
    "vae": "guide on pipeline VAE", "idx": "indices 20 to 99",
}
ARM_LABEL["fadapt"] = "floor + adaptive \u03bb"
ARM_LINESTYLE = {"lam0": ":", "k1": ":"}
ROOT_COLORS = [BLUE, ORANGE, GREEN, PINK]   # x_T roots 1 to 4, grid frames and trees
OTHER_ARMS = "other arms"                   # legend entry for the unlabelled greys

WIDTH_PX = 1400   # 2x a 700 px distill column
DPI = 200
WIDTH_IN = WIDTH_PX / DPI         # constrained layout keeps the margins tight at this exact width


def arm_color(arm):
    """Colour of an arm, its `_b1` replay included; unknown arms are light grey."""
    return ARM_COLOR.get(arm.split("_")[0], GREY_LIGHT)


def arm_label(arm):
    return ARM_LABEL.get(arm.split("_")[0], arm)


def setup():
    """rcParams for the post: font, sizes, no title, transparent export at a fixed width."""
    import matplotlib as mpl   # here, so make_table_sd.py can import aggregate() without it
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.titlesize": 9,           # titles are not used; kept small if one slips in
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK_LIGHT,
        "ytick.color": INK_LIGHT, "text.color": INK,
        "legend.frameon": False, "figure.dpi": 100,
        "savefig.dpi": DPI, "savefig.transparent": True, "savefig.bbox": "standard",
        "figure.constrained_layout.use": True, "svg.fonttype": "none",
    })


def figure(height_in, width_in=WIDTH_IN, **kw):
    """A figure at the export width (1400 px once saved), constrained layout for the margins."""
    import matplotlib.pyplot as plt
    setup()
    return plt.subplots(figsize=(width_in, height_in), **kw)


def dress(ax, grid="y"):
    """No top or right spine, a thin horizontal grid or none."""
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
    ax.tick_params(colors=INK_LIGHT, labelsize=9)


def label_end(ax, x, y, text, color, dx=4):
    """Direct label at the end of a series, instead of a legend (rule 3)."""
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points",
                va="center", ha="left", color=color, fontsize=9)


def save(fig, path):
    """PNG at WIDTH_PX and SVG beside it; path with or without suffix."""
    path = Path(path)
    stem = path.with_suffix("")
    fig.savefig(stem.with_suffix(".png"), dpi=DPI)
    fig.savefig(stem.with_suffix(".svg"))
    try:
        from PIL import Image
        w = Image.open(stem.with_suffix(".png")).size[0]
        if w < WIDTH_PX:
            print(f"{stem.name}.png is {w} px wide, under {WIDTH_PX}: widen the figure")
    except ImportError:
        pass
    return stem.with_suffix(".png"), stem.with_suffix(".svg")


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
