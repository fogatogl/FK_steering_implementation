"""I. La racine survivante, sur 100 prompts au lieu de 20.

sd_ref_fields100.json ne porte pas r_at_schedule, mais il porte root_slots : la
case de depart dont descendent les quatre images finales. Appariee au ir par case
de bon4 (meme prompt, meme seed_effective, meme x_T), elle repond directement a
"la racine gardee vaut-elle mieux qu'une racine tiree au sort", avec cinq fois
plus de runs que le constat 3.

docs/max_potential.md avait deja mesure un taux de mauvaise racine au niveau du
hasard sur S60. Ce qui est nouveau ici : le RANG de la racine gardee et le cout en
reward, pas seulement un taux binaire, et sur le fichier de reference a 100
prompts.
"""
import json
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent.parent / "results"
load = lambda p: json.loads((R / p).read_text())["runs"]

bon = {r["prompt_id"]: r for r in load("sd_baseline.json")
       if r["sampler"] == "bon4" and r["seed"] == 2024}
fk = [r for r in load("sd_ref_fields100.json") if r["prompt_id"] in bon]
print(__doc__.splitlines()[0], "\n")
mauvais = [r["prompt_id"] for r in fk if r["seed_effective"] != bon[r["prompt_id"]]["seed_effective"]]
print(f"  {len(fk)} runs apparies, {len(mauvais)} desaccords de seed_effective"
      + (" (ok)" if not mauvais else f" -> {mauvais[:3]}"))

rangs, pertes, alea, meilleur = [], [], [], []
for r in fk:
    ir = np.array(bon[r["prompt_id"]]["ir"])
    ordre = np.argsort(np.argsort(-ir))          # 0 = la meilleure racine
    for j in set(r["root_slots"]):               # une ligne par racine survivante
        rangs.append(int(ordre[j]))
        pertes.append(float(ir.max() - ir[j]))
    alea.append(float(ir.max() - ir.mean()))
    meilleur.append(int(np.argmax(ir) in set(r["root_slots"])))

rangs, pertes, alea = np.array(rangs), np.array(pertes), np.array(alea)
n = len(fk)
print(f"\n  rang de la racine gardee parmi les 4 (0 = la meilleure), {len(rangs)} racines :")
print(f"    moyenne {rangs.mean():.3f} +/- {rangs.std(ddof=1)/len(rangs)**.5:.3f}   "
      f"(1.500 au hasard)")
print(f"    loi : " + "  ".join(f"rang {k} : {(rangs == k).mean():.0%}" for k in range(4))
      + "   (25 % chacun au hasard)")
lig = np.mean([len(set(r["root_slots"])) for r in fk])
print(f"\n  la meilleure racine survit dans {np.mean(meilleur):.0%} des runs")
print(f"    niveau du hasard, vu le nombre de racines survivantes ({lig:.2f} par run) : "
      f"{lig/4:.0%}")
print(f"\n  cout en ImageReward :")
print(f"    ir(meilleure) - ir(racine gardee)  : {pertes.mean():+.4f} "
      f"+/- {pertes.std(ddof=1)/len(pertes)**.5:.4f}")
print(f"    ir(meilleure) - ir(racine au sort) : {alea.mean():+.4f} "
      f"+/- {alea.std(ddof=1)/len(alea)**.5:.4f}")
d = np.array([np.mean([bon[r["prompt_id"]]["ir"].__getitem__(j) for j in set(r["root_slots"])])
              - np.mean(bon[r["prompt_id"]]["ir"]) for r in fk])
print(f"    apparie : ir(racine gardee) - ir(moyenne des 4) = {d.mean():+.4f} "
      f"+/- {d.std(ddof=1)/len(d)**.5:.4f} sur {n} prompts")
print(f"\n  (lecture apres verification ci-dessous)")

# --- verification : une observation par run, et un bootstrap sur les prompts ---
# Les runs a deux racines survivantes donnaient deux lignes ci-dessus. Un run = une
# observation, sinon les runs les moins effondres pesent double.
print("\n  verification, une observation par run :")
rang_run = np.array([np.mean([int(np.argsort(np.argsort(-np.array(bon[r["prompt_id"]]["ir"])))[j])
                              for j in set(r["root_slots"])]) for r in fk])
rng = np.random.default_rng(2024)
def boot(v, tirages=20000):
    t = v[rng.integers(0, len(v), size=(tirages, len(v)))].mean(1)
    return np.percentile(t, [2.5, 97.5])
lo, hi = boot(rang_run)
print(f"    rang moyen par run : {rang_run.mean():.3f}  IC95 bootstrap [{lo:.3f}, {hi:.3f}]  "
      f"(1.500 au hasard)")
lo, hi = boot(d)
print(f"    gain sur la racine moyenne : {d.mean():+.4f}  IC95 bootstrap [{lo:+.4f}, {hi:+.4f}]")
print(f"    part de l'ecart racine-moyenne / meilleure-racine recuperee : "
      f"{100*d.mean()/alea.mean():.0f} %")

# et le meme calcul en separant les runs selon le nombre de racines survivantes
for nl in sorted({len(set(r["root_slots"])) for r in fk}):
    sub = [r for r in fk if len(set(r["root_slots"])) == nl]
    dd = np.array([np.mean([bon[r["prompt_id"]]["ir"][j] for j in set(r["root_slots"])])
                   - np.mean(bon[r["prompt_id"]]["ir"]) for r in sub])
    print(f"    {len(sub):3d} runs a {nl} racine(s) : gain {dd.mean():+.4f} "
          f"+/- {dd.std(ddof=1)/max(len(dd),1)**.5:.4f}")

print("\n  A retenir : la racine gardee n'est PAS tiree au sort a n = 100. Elle recupere")
print("  environ un quart de l'ecart disponible, la ou best-of-4 en recupere la totalite")
print("  par construction. Mais root_slots est le produit de TOUS les reechantillonnages,")
print("  pas du seul pas t = 80 : dans les runs ou deux racines survivent a t = 80, c'est")
print("  t = 60 qui tranche, avec une reward guide deja moins bruitee. Le constat 3, qui")
print("  isole le seul pas t = 80, mesure autre chose et reste ce qu'il est.")
