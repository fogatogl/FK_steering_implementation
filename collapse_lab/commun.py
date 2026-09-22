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


def fk_avec_r():
    """Les 20 runs fk4 qui portent r_at_schedule, une fois chacun."""
    return load("sd_variants/fk4_stat.json")


def bon4(seed=2024):
    return {r["prompt_id"]: r for r in load("sd_baseline.json")
            if r["sampler"] == "bon4" and r["seed"] == seed}


def fk4(seed=2024):
    return {r["prompt_id"]: r for r in load("sd_baseline.json")
            if r["sampler"] == "fk4" and r["seed"] == seed}
