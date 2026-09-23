"""C. La selection qui tue les lignees porte-t-elle de l'information ?

r_at_schedule[0] est lu au premier pas planifie (t=80), avant tout
reechantillonnage : la particule j de fk4 y est encore exactement la particule j
de bon4, qui partage son x_T (meme prompts, meme seed_effective = seed*1000+i,
meme randn_tensor). bon4["ir"][j] est donc ce que cette racine devient si on la
laisse tranquille. On demande : le classement a t=80 predit-il ce classement final ?
"""
import json, itertools
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent.parent / "results"
load = lambda p: json.loads((R / p).read_text())["runs"]

bon = {r["prompt_id"]: r for r in load("sd_baseline.json")
       if r["sampler"] == "bon4" and r["seed"] == 2024}
k1 = {r["prompt_id"]: r for r in load("sd_baseline.json")
      if r["sampler"] == "k1" and r["seed"] == 2024}

# fk4_diff porte les memes prompts aux memes x_T et la meme ligne 0 : l'empiler
# doublerait chaque observation et diviserait l'erreur-type par racine de 2 a tort.
fk = [r for r in load("sd_variants/fk4_stat.json") if r["prompt_id"] in bon]
print(__doc__.splitlines()[0], "\n")
print(f"{len(fk)} runs fk4 apparies a bon4 sur (prompt_id, seed=2024)\n")

# controle de l'appariement : meme seed_effective des deux cotes
mauvais = [r["prompt_id"] for r in fk if r["seed_effective"] != bon[r["prompt_id"]]["seed_effective"]]
print(f"  controle appariement : {len(mauvais)} desaccords de seed_effective"
      + (f" -> {mauvais[:5]}" if mauvais else " (ok)"))


def kendall(a, b):
    """tau-b sur 4 points ; +1 = meme ordre, 0 = independant, -1 = inverse."""
    n, c, d = len(a), 0, 0
    for i, j in itertools.combinations(range(n), 2):
        s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
        c += s > 0
        d += s < 0
    return (c - d) / (c + d) if c + d else 0.0


ts = [80, 60, 40, 20, 0]
print("\n  ligne 0 (t=80) de r_phi  contre  ir final de la MEME racine sous bon4 :")
tau = np.array([kendall(r["r_at_schedule"][0], bon[r["prompt_id"]]["ir"]) for r in fk])
top1 = np.array([int(np.argmax(r["r_at_schedule"][0]) == np.argmax(bon[r["prompt_id"]]["ir"]))
                 for r in fk])
# erreur-type sur le tau moyen, runs independants
print(f"    Kendall tau moyen        : {tau.mean():+.3f} +/- {tau.std(ddof=1)/len(tau)**.5:.3f}"
      f"   (median {np.median(tau):+.3f})")
print(f"    le meilleur a t=80 est le meilleur final : {top1.mean():.0%} des runs "
      f"({top1.sum()}/{len(top1)}), hasard 25 %")
# le meme tau, mais entre la reward finale du guide (ligne 4) et l'ir final : la borne haute
print(f"\n  pour borner : combien de reward la racine choisie coute-t-elle ?")
perte = np.array([max(bon[r["prompt_id"]]["ir"]) - bon[r["prompt_id"]]["ir"][int(np.argmax(r["r_at_schedule"][0]))]
                  for r in fk])
alea = np.array([max(bon[r["prompt_id"]]["ir"]) - np.mean(bon[r["prompt_id"]]["ir"]) for r in fk])
print(f"    ir(meilleur) - ir(racine choisie a t=80) : {perte.mean():+.4f}")
print(f"    ir(meilleur) - ir(racine au hasard)      : {alea.mean():+.4f}")
ecart = perte - alea   # apparie : meme prompt des deux cotes
print(f"    difference appariee (choisie - hasard)   : {ecart.mean():+.4f} +/- "
      f"{ecart.std(ddof=1)/len(ecart)**.5:.4f}")
print(f"    -> indiscernable du hasard ; le signe est defavorable mais l'erreur-type "
      f"le couvre largement")

print("\n  et l'information monte-t-elle le long de la trajectoire ?")
print("    (tau entre la ligne m et la ligne 4 du MEME run : valable seulement si la lignee")
print("     est unique, sinon les rangs de la ligne m ne suivent pas les cases de la ligne 4)")
uni = [r for r in fk if r["n_lineages"] == 1]
for m, t in enumerate(ts[:-1]):
    # apres effondrement complet les 4 cases de chaque ligne sont des freres, mais la
    # permutation du peigne entre lignes n'est pas enregistree : on ne lit que la dispersion
    sp = np.array([np.ptp(r["r_at_schedule"][m]) for r in uni])
    print(f"    t={t:3d} : etendue mediane {np.median(sp):.3f}  "
          f"-> a lam=10, {10*np.median(sp):.1f} nats entre la meilleure et la pire case")

print("\n  reference : ecart type de r_phi entre les 4 cases, par ligne")
for m, t in enumerate(ts):
    s = np.array([np.std(r["r_at_schedule"][m]) for r in fk])
    mu = np.array([np.mean(r["r_at_schedule"][m]) for r in fk])
    print(f"    t={t:3d} : niveau moyen {mu.mean():+.3f}, ecart type intra-run {s.mean():.3f}, "
          f"runs ou les 4 r_phi sont < 0 : {np.mean([max(r['r_at_schedule'][m]) < 0 for r in fk]):.0%}")
