"""Chargement partage, et la regle qui evite le double comptage.

fk4_stat.json et fk4_diff.json portent les MEMES 20 prompts aux MEMES x_T, et
d_forms.py montre que leurs r_phi coincident a zero pres aux pas t = 80 et t = 60.
Les empiler donnerait 40 lignes dont 20 sont des copies : les moyennes ne
bougeraient pas, les erreurs-types seraient divisees par racine de 2 a tort.
Tout ce qui se lit au premier pas se lit donc sur un seul des deux fichiers.
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"


def load(p):
    return json.loads((R / p).read_text())["runs"]


def wilson(k, n, z=1.96):
    """95 % Wilson score interval of a proportion k / n, as the string 'k/n = p [lo, hi]'."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / (1 + z * z / n)
    return f"{k}/{n} = {p:.0%} [{max(0, centre - half):.0%}, {min(1, centre + half):.0%}]"


def fk_avec_r():
    """Les 20 runs fk4 qui portent r_at_schedule, une fois chacun."""
    return load("sd_variants/fk4_stat.json")


def bon4(seed=2024):
    return {r["prompt_id"]: r for r in load("sd_baseline.json")
            if r["sampler"] == "bon4" and r["seed"] == seed}


def fk4(seed=2024):
    return {r["prompt_id"]: r for r in load("sd_baseline.json")
            if r["sampler"] == "fk4" and r["seed"] == seed}


# Finding 19, settled on 23/09 evening by session D: the smc.models path reproduces to the
# fourth decimal within one GPU model, free and steered arms alike, and not across two. The
# device is recorded since 23/09 18:47; before, the seconds per run separate the two groups
# the records hold: the A2 at 85-92 s, the other card (not recorded) at 52-65 s.
def groupe(r):
    if "session" in r:
        return "A2" if "A2" in r["session"]["device"] else r["session"]["device"]
    return "A2" if r["seconds"] > 75 else "fast"


def references():
    """{group: {prompt_id: ctl run}}: FK at the paper's setting in each machine group.
    fast: session C (100 prompts, equal to the 20/09 fk4 at seed 2024 to 5e-5).
    A2: session A's ctl in probe.json (40 prompts), completed by session D where it exists
    (the two agree slot by slot where they overlap)."""
    out = R.parent / "collapse_lab" / "out"
    refs = {"fast": {}, "A2": {}}
    for name in ("session_D/probe_D.json", "probe.json", "probe_C.json"):
        if (out / name).exists():
            for r in json.loads((out / name).read_text())["runs"]:
                if r["arm"] == "ctl":
                    refs[groupe(r)][r["prompt_id"]] = r
    return refs
