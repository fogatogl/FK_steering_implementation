"""Figure F0 (etendue) : la table 1 du papier, ce depot, et le code publie, sur les memes prompts.

Panneau gauche : ImageReward de la meilleure des k particules, par ligne, avec le papier en
repere gris (results/paper_table1_sd15.json, qui cite sa source). Panneau droit : la difference
appariee par prompt contre best-of-4 (bon4, seed 2024), erreur-type sur les prompts, n en
legende. Lignes : k1, bon4, ctl (smc/, sd_ref_fields100.json), R1 (smc/ avec les quatre choix
du code publie, collapse_lab/out/probe.json), R0g24 (leur code, nos x_T), R0 (leur code, leur
graine), toutes deux dans results/sd_authors_R0.json. Aucune valeur codee en dur.

Depuis la racine : /home/onyxia/work/.venvs/ddpm/bin/python scripts/plot_fig0_reference.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figstyle import BLUE, GREEN, INK, INK_LIGHT, RED, dress

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"


def par_prompt(runs, cle, valeur, **filtre):
    return {r["prompt_id"]: r[valeur] for r in runs
            if all(r.get(k) == v for k, v in filtre.items()) and r.get(cle) is not None}


base = json.loads((R / "sd_baseline.json").read_text())["runs"]
ref100 = json.loads((R / "sd_ref_fields100.json").read_text())["runs"]
auteurs = json.loads((R / "sd_authors_R0.json").read_text())["runs"]
probe = json.loads((ROOT / "collapse_lab" / "out" / "probe.json").read_text())["runs"]
papier = {r["sampler"]: r for r in json.loads((R / "paper_table1_sd15.json").read_text())["rows"]}

lignes = [
    ("k1",    "un tirage",                       par_prompt(base, "ir_max", "ir_max", sampler="k1", seed=2024), INK_LIGHT),
    ("bon4",  "best-of-4",                       par_prompt(base, "ir_max", "ir_max", sampler="bon4", seed=2024), INK_LIGHT),
    ("ctl",   "FK,\nce depot",                   par_prompt(ref100, "ir_max", "ir_max", sampler="fk4", seed=2024), BLUE),
    ("R1",    "ce depot,\nleurs 4 choix",        {r["prompt_id"]: r["ir_max"] for r in probe if r["arm"] == "R1"}, BLUE),
    ("R0g24", "leur code,\nnos x_T",             par_prompt(auteurs, "ir_max", "ir_max", sampler="authors_paper_g", seed=2024), RED),
    ("R0",    "leur code,\nleur graine",         par_prompt(auteurs, "ir_max", "ir_max", sampler="authors_paper", seed=42), RED),
]
bon = lignes[1][2]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
x = np.arange(len(lignes))
moy = [np.mean(list(d.values())) for _, _, d, _ in lignes]
err = [np.std(list(d.values()), ddof=1) / len(d) ** .5 for _, _, d, _ in lignes]
a1.bar(x, moy, width=0.6, color=[c for *_, c in lignes], yerr=err, capsize=4, ecolor=INK, zorder=3)
for i, (nom, _, d, _) in enumerate(lignes):
    a1.text(i, moy[i] + err[i] + 0.03, f"{moy[i]:.3f}\nn={len(d)}", ha="center", fontsize=8, color=INK)
for nom, xpos in (("k1", 0), ("bon4", 1), ("fk4", 2)):
    a1.hlines(papier[nom]["ir_max"], xpos - 0.42, xpos + 0.42, color=INK, ls="--", lw=1.2, zorder=4)
a1.text(-0.5, 1.09, "tirets : table 1 du papier, " + ", ".join(f"{n} {papier[n]['ir_max']:.3f}" for n in ("k1", "bon4", "fk4")),
        fontsize=8, color=INK, ha="left", va="top")
a1.set_xticks(x); a1.set_xticklabels([f"{n}\n{e}" for n, e, _, _ in lignes], fontsize=7.5, rotation=0)
a1.tick_params(axis="x", pad=2)
a1.set_ylabel("ImageReward de la meilleure particule")
a1.set_title("les niveaux ; tirets : la table 1 du papier (SD v1.5)", fontsize=9, loc="left")
a1.set_ylim(0, 1.15)
dress(a1, grid="y")

diffs = []
for nom, etiq, d, c in lignes[2:]:
    com = sorted(set(d) & set(bon))
    dd = np.array([d[p] - bon[p] for p in com])
    diffs.append((nom, dd.mean(), dd.std(ddof=1) / len(dd) ** .5, len(com), c))
y = np.arange(len(diffs))
a2.barh(y, [m for _, m, *_ in diffs], xerr=[s for _, _, s, *_ in diffs], color=[c for *_, c in diffs], capsize=4, ecolor=INK, height=0.6, zorder=3)
a2.axvline(0, color=INK, lw=0.8)
a2.axvline(papier["fk4"]["ir_max"] - papier["bon4"]["ir_max"], color=INK, ls="--", lw=1.2)
a2.text(papier["fk4"]["ir_max"] - papier["bon4"]["ir_max"] + 0.005, len(diffs) - 0.6, "papier\n+0.161", fontsize=8, color=INK)
a2.set_yticks(y); a2.set_yticklabels([f"{n}  (n={k})" for n, _, _, k, _ in diffs], fontsize=8)
a2.invert_yaxis()
a2.set_xlabel("difference appariee par prompt contre best-of-4 (meme seed 2024 pour ctl, R1, R0g24)")
a2.set_title("l'ecart a best-of-4 : aucune ligne n'atteint le papier", fontsize=9, loc="left")
dress(a2, grid="x")
fig.suptitle("Figure F0 — la reproduction, ce depot et le code publie sous la configuration de la table 1", fontsize=11, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = ROOT / "figures" / "fig0_reference"
for ext in ("png", "svg"):
    fig.savefig(f"{out}.{ext}", dpi=160, facecolor="white")
for nom, m, s, k, _ in diffs:
    print(f"{nom:6s} - bon4 : {m:+.3f} +/- {s:.3f} (n={k})")
print(f"{out}.png")
