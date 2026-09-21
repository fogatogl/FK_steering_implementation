"""Ce que les 300 runs de sd_baseline.json disent de l'effondrement, sans GPU.

Quatre lectures qui vont dans la section 3 du billet :

1. l'ESS au premier pas du calendrier contre le gain apparié fk4 - bon4, en
   Spearman et par tercile : est-ce que le run qui s'effondre est le run qui
   perd, run par run, ou seulement en moyenne ;
2. un bootstrap sur les prompts pour les trois différences appariées, à côté des
   erreurs-types normales de docs/results.md : l'écart entre les deux dit si
   l'asymétrie de la distribution des gains compte ;
3. le temps de mur, et le N de best-of-N qu'il faudrait pour égaler FK à durée
   égale plutôt qu'à lignes d'UNet égales ;
4. avec --fk sur un fichier qui porte root_slots : la fraction des runs où la
   racine que FK garde n'est pas celle que best-of-4 aurait choisie.

La moyenne des seeds se fait par prompt avant tout : les trois seeds d'un prompt
ne sont pas trois observations indépendantes de l'écart entre méthodes.

Depuis la racine, dans le venv ddpm : `python scripts/analyze_sd_collapse.py`.
"""
import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np

from compare_sd_variants import apparie, par_cle
from figstyle import BLUE, INK, INK_LIGHT, RED, dress

ROOT = Path(__file__).resolve().parent.parent


def par_prompt(a, b, key):
    """Différences a - b moyennées sur les seeds : une valeur par prompt, triée."""
    d = {}
    for k in sorted(set(a) & set(b)):
        d.setdefault(k[0], []).append(a[k][key] - b[k][key])
    return np.array([st.mean(v) for _, v in sorted(d.items())])


def bootstrap(d, tirages, rng):
    tires = d[rng.integers(0, len(d), size=(tirages, len(d)))].mean(axis=1)
    return np.percentile(tires, [2.5, 97.5])


def rangs(v):
    r = np.empty(len(v))
    r[np.argsort(v, kind="stable")] = np.arange(len(v), dtype=float)
    # rang moyen sur les ex aequo : l'ESS vaut exactement 1.0 sur une part des
    # runs, et un ordre arbitraire entre eux déplacerait le rho
    for val in np.unique(v):
        m = v == val
        r[m] = r[m].mean()
    return r


def spearman(x, y):
    return float(np.corrcoef(rangs(x), rangs(y))[0, 1])


def figure(ess, gain, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordre = np.argsort(ess)
    tiers = np.array_split(ordre, 3)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhline(0, color=INK_LIGHT, linewidth=0.8)
    ax.scatter(ess, gain, s=14, color=BLUE, alpha=0.45, linewidths=0)
    ax.plot([ess[t].mean() for t in tiers], [np.median(gain[t]) for t in tiers],
            "o-", color=RED, markersize=6, linewidth=1.6, label="median per ESS tercile")
    dress(ax)
    ax.set_xlabel("ESS at the first scheduled step (t = 80), out of k = 4", color=INK, fontsize=10)
    ax.set_ylabel("ImageReward, fk4 - bon4, same $x_T$", color=INK, fontsize=10)
    ax.set_title("An early collapse is not what loses the run", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_LIGHT)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"\n{out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default=str(ROOT / "results" / "sd_baseline.json"))
    p.add_argument("--fk", nargs="+", default=None,
                   help="un JSON dont les enregistrements fk4 portent root_slots, pour le taux de mauvaise racine")
    p.add_argument("--draws", type=int, default=10000)
    p.add_argument("--out", default=str(ROOT / "figures" / "fig7_ess_vs_gain.png"))
    args = p.parse_args()

    runs = json.loads(Path(args.baseline).read_text())["runs"]
    fk, bon, k1 = par_cle(runs, "fk4"), par_cle(runs, "bon4"), par_cle(runs, "k1")
    communs = sorted(set(fk) & set(bon))
    print(f"{len(communs)} runs appariés, {len({c[0] for c in communs})} prompts, "
          f"{len({c[1] for c in communs})} seeds\n")

    # 1. l'ESS au premier pas contre le gain, run par run
    ess = np.array([fk[c]["ess_at_schedule"][0] for c in communs])
    gain = np.array([fk[c]["ir_max"] - bon[c]["ir_max"] for c in communs])
    ordre = np.argsort(ess)
    print(f"ESS au premier pas : médiane {np.median(ess):.2f}, "
          f"{(ess < 1.5).mean():.0%} des runs sous 1.5, maximum {ess.max():.2f}")
    print(f"Spearman(ESS, gain) = {spearman(ess, gain):+.3f} sur {len(ess)} runs")
    for nom, t in zip(("bas", "milieu", "haut"), np.array_split(ordre, 3)):
        print(f"  tercile {nom:6s} ESS {ess[t].min():.2f}-{ess[t].max():.2f} : "
              f"gain médian {np.median(gain[t]):+.4f}, moyen {gain[t].mean():+.4f}, "
              f"{(gain[t] > 0).mean():.0%} gagnés")

    # 2. bootstrap sur les prompts, contre l'erreur-type normale
    rng = np.random.default_rng(2024)
    print(f"\nbootstrap sur les prompts, {args.draws} tirages :")
    for libelle, a, b, cle in (("fk4 - bon4, ImageReward", fk, bon, "ir_max"),
                               ("fk4 - bon4, HPS v2.1   ", fk, bon, "hps_at_ir_max"),
                               ("bon4 - k1,  ImageReward", bon, k1, "ir_max")):
        d = par_prompt(a, b, cle)
        lo, hi = bootstrap(d, args.draws, rng)
        se = d.std(ddof=1) / len(d) ** .5
        print(f"  {libelle} : {d.mean():+.4f}  IC95 [{lo:+.4f}, {hi:+.4f}]  "
              f"normal [{d.mean() - 1.96 * se:+.4f}, {d.mean() + 1.96 * se:+.4f}]  "
              f"{(d > 0).sum()}/{len(d)} prompts gagnés")

    # 3. le budget en secondes, et non en lignes d'UNet
    s_fk = st.mean(fk[c]["seconds"] for c in communs)
    s_bon = st.mean(bon[c]["seconds"] for c in communs)
    print(f"\ntemps de mur : fk4 {s_fk:.1f} s, bon4 {s_bon:.1f} s, rapport {s_fk / s_bon:.3f}")
    print(f"  best-of-N à durée égale : N = {4 * s_fk / s_bon:.2f} échantillons "
          f"({s_bon / 4:.1f} s par échantillon)")

    # 4. la racine gardée contre celle que best-of-4 aurait choisie
    for chemin in (args.fk or []):
        var = par_cle(json.loads(Path(chemin).read_text())["runs"], "fk4")
        avec = {c: r for c, r in var.items() if "root_slots" in r and c in bon}
        if not avec:
            print(f"\n{chemin} : aucun enregistrement fk4 avec root_slots")
        else:
            # le slot i de bon4 et le slot i de fk4 partent du même x_T : le
            # wrapper reproduit la pipeline au bit près à lambda = 0 (run 11).
            # Si cette correspondance tombe, ce taux ne veut rien dire.
            mauvaises = [max(range(len(bon[c]["ir"])), key=lambda j: bon[c]["ir"][j])
                         not in set(avec[c]["root_slots"]) for c in sorted(avec)]
            lign = st.mean(len(set(avec[c]["root_slots"])) for c in sorted(avec))
            print(f"\nracine gardée, sur {len(mauvaises)} runs de {Path(chemin).name} :")
            print(f"  {sum(mauvaises)}/{len(mauvaises)} ({st.mean(mauvaises):.0%}) ne contiennent pas "
                  f"l'argmax de bon4 ; {lign:.2f} racines distinctes par run")
            print(f"  au hasard sur k = 4, un tirage de {lign:.2f} racines en manquerait "
                  f"{1 - lign / 4:.0%}")

    figure(ess, gain, args.out)


if __name__ == "__main__":
    main()
