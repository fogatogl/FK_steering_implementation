"""Table 1 (ligne SD v1.5) : ce qu'on mesure, en regard du papier, en Markdown sur stdout.

Depuis la racine : `python scripts/make_table_sd.py [--out docs/table_sd.md]`.
"""
import argparse
from pathlib import Path

import numpy as np

from figstyle import aggregate, load_runs

ROOT = Path(__file__).resolve().parent.parent

# FK Steering, table 1, SD v1.5 : (ImageReward, HPS). La clé est un préfixe du nom
# d'échantillonneur : la ligne FK n'est pas encore produite et son nom n'est pas arrêté.
PAPIER = {"k1": (0.187, 0.245), "bon4": (0.737, 0.265), "fk": (0.898, 0.263)}

COLS = ["échantillonneur", "prompts", "ImageReward", "papier", "Δ",
        "HPS", "papier", "Δ", "lignes UNet", "appels UNet", "secondes"]


def papier(sampler):
    for prefixe, valeurs in PAPIER.items():
        if sampler.startswith(prefixe):
            return valeurs
    return None


def sigma_seeds(runs, key):
    """σ de la moyenne sur les prompts d'un seed à l'autre ; None s'il n'y a qu'un seed."""
    seeds = sorted({r["seed"] for r in runs})
    if len(seeds) < 2:
        return None
    moyennes = [np.mean([r[key] for r in runs if r["seed"] == s]) for s in seeds]
    return float(np.std(moyennes, ddof=1))


def cellule(valeur, sigma):
    return f"{valeur:.3f}" + (f" ± {sigma:.3f}" if sigma is not None else "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", nargs="+", default=[str(ROOT / "results" / "sd_baseline.json")])
    p.add_argument("--out")
    args = p.parse_args()

    runs = load_runs(args.json)
    if not runs:
        raise SystemExit("aucun enregistrement")

    xs, ir, _, _ = aggregate(runs, "sampler", "ir_max")
    _, hps, _, _ = aggregate(runs, "sampler", "hps_at_ir_max")
    _, lignes_unet, _, _ = aggregate(runs, "sampler", "n_unet_rows")
    _, appels, _, _ = aggregate(runs, "sampler", "n_unet_calls")
    _, secondes, _, _ = aggregate(runs, "sampler", "seconds")
    # budget croissant plutôt qu'alphabétique : k1, bon4, puis FK quand il arrivera
    n_par_sampler = {r["sampler"]: r["n"] for r in runs}
    ordre = np.argsort([n_par_sampler[s] for s in xs], kind="stable")

    table = [f"| {' | '.join(COLS)} |", "|---|" + "---:|" * (len(COLS) - 1)]
    for i in ordre:
        sub = [r for r in runs if r["sampler"] == xs[i]]
        ref = papier(xs[i])
        n_prompts = len({r["prompt_id"] for r in sub})
        p_ir, p_hps = (f"{ref[0]:.3f}", f"{ref[1]:.3f}") if ref else ("—", "—")
        d_ir, d_hps = (f"{ir[i] - ref[0]:+.3f}", f"{hps[i] - ref[1]:+.3f}") if ref else ("—", "—")
        table.append(
            f"| {xs[i]} | {n_prompts} | {cellule(ir[i], sigma_seeds(sub, 'ir_max'))} | {p_ir} | {d_ir} "
            f"| {cellule(hps[i], sigma_seeds(sub, 'hps_at_ir_max'))} | {p_hps} | {d_hps} "
            f"| {lignes_unet[i]:.0f} | {appels[i]:.0f} | {secondes[i]:.1f} |")

    seeds = sorted({r["seed"] for r in runs})
    texte = "\n".join(table) + (
        f"\n\nAgrégat du fichier tel quel — enregistrements : {len(runs)}, prompts "
        f"distincts : {len({r['prompt_id'] for r in runs})}, seeds : {seeds}. "
        "Δ = mesuré − papier ; ImageReward et HPS sont ceux de la particule que "
        "l'ImageReward désigne. Le budget apparié se lit sur les lignes UNet, pas sur "
        "les appels : les N particules passent dans un seul forward batché.\n")

    print(texte)
    if args.out:
        Path(args.out).write_text(texte)
        print(args.out)


if __name__ == "__main__":
    main()
