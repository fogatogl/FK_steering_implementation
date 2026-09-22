"""B. Combien d'ancetres distincts le peigne systematique laisse-t-il vivre ?

Reproduit exactement smc/resampling.py:resample_systematic (peigne regulier,
u ~ U(0, 1/k)) sur les poids recalcules depuis r_at_schedule, sans importer smc/.
Le nombre de copies de la particule j est floor ou ceil de k*w_j : pour un poids
max de 0.85 a k=4, la tete prend 3 ou 4 cases et il reste au plus 2 lignees.
"""
import json
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent.parent / "results"
load = lambda p: json.loads((R / p).read_text())["runs"]
poids = lambda lw: np.exp(lw - lw.max(-1, keepdims=True)) / np.exp(lw - lw.max(-1, keepdims=True)).sum(-1, keepdims=True)


def survivants_exact(w, k=4):
    """Esperance et loi du nombre d'ancetres distincts sous le peigne, u integre.

    Le point i du peigne est u + i/k, u ~ U(0,1/k) ; la particule j prend le
    point i ssi cum_{j-1} <= u + i/k < cum_j. j survit ssi au moins un point y
    tombe. On integre exactement en decoupant [0,1/k] aux ruptures.
    """
    cum = np.cumsum(w)
    bornes = sorted({0.0, 1.0 / k} | {c - i / k for c in cum[:-1] for i in range(k)
                                      if 0.0 < c - i / k < 1.0 / k})
    loi = {}
    for a, b in zip(bornes[:-1], bornes[1:]):
        u = 0.5 * (a + b)
        idx = np.searchsorted(cum, u + np.arange(k) / k, side="left")
        loi[len(set(idx.tolist()))] = loi.get(len(set(idx.tolist())), 0.0) + (b - a) * k
    return loi


runs = load("sd_variants/fk4_stat.json")   # un seul fichier : cf. commun.py
r0 = np.array([r["r_at_schedule"][0] for r in runs])
w0 = poids(10.0 * r0)

lois = [survivants_exact(w) for w in w0]
esp = np.array([sum(n * p for n, p in l.items()) for l in lois])
p1 = np.array([l.get(1, 0.0) for l in lois])
print(__doc__.splitlines()[0], "\n")
print(f"Au SEUL premier pas planifie (t=80), sur {len(runs)} runs, lam=10 :")
print(f"  esperance du nombre d'ancetres distincts : {esp.mean():.3f}  (median {np.median(esp):.3f})")
print(f"  P(une seule lignee des ce pas)           : {p1.mean():.3f}")
for n in (1, 2, 3, 4):
    print(f"    P({n} ancetres) moyenne : {np.mean([l.get(n, 0.0) for l in lois]):.3f}")

# chaine : borne sur le nombre final de lignees.
obs = np.array([r["n_lineages"] for r in runs])
print(f"\n  n_lineages observe en fin de run : moyenne {obs.mean():.3f}, "
      f"{(obs == 1).sum()}/{len(obs)} a une seule lignee")
print(f"  (le nombre de lignees ne peut que decroitre : 4 reechantillonnages, "
      f"n_resamplings moyen {np.mean([r['n_resamplings'] for r in runs]):.2f})")

# S80 : un seul point de selection -> teste la prediction du premier pas isolement
s80 = load("sd_variants/S80.json")
o80 = np.array([r["n_lineages"] for r in s80])
print(f"\nControle S80 (calendrier [0, 80] : UN seul reechantillonnage, a t=80) :")
print(f"  n_lineages observe : moyenne {o80.mean():.3f}, loi "
      f"{ {n: int((o80 == n).sum()) for n in sorted(set(o80.tolist()))} } sur {len(o80)} runs")
print(f"  predit par le peigne au pas t=80        : {esp.mean():.3f} ancetres attendus")
print(f"  ESS mediane a t=80 sur S80 : {np.median([r['ess_at_schedule'][0] for r in s80]):.2f}")

# --- contrefactuel statique : le meme peigne a d'autres lambda, et sous le plancher ---
print("\nLe meme premier pas, a d'autres lambda (et sous le plancher des auteurs) :")
print(f"  {'regime':22s} {'ESS mediane':>12s} {'ancetres attendus':>19s} {'P(1 lignee)':>12s} "
      f"{'pas inerte':>11s}")
for nom, lam, plancher in (("lam=10 (reference)", 10.0, False), ("lam=5", 5.0, False),
                           ("lam=2", 2.0, False), ("lam=1", 1.0, False),
                           ("lam=10 + plancher 0", 10.0, True), ("lam=2 + plancher 0", 2.0, True)):
    rr = np.maximum(r0, 0.0) if plancher else r0
    w = poids(lam * rr)
    e = 1.0 / (w ** 2).sum(-1)
    l = [survivants_exact(x) for x in w]
    inerte = (rr.max(1) - rr.min(1)) < 1e-12
    print(f"  {nom:22s} {np.median(e):12.2f} {np.mean([sum(n*p for n,p in x.items()) for x in l]):19.3f} "
          f"{np.mean([x.get(1,0.0) for x in l]):12.3f} {inerte.mean():11.0%}")
print("  (pas inerte = les quatre logG sont egaux, donc ESS = k exactement : a seuil 1.0")
print("   smc/weights.py:should_resample teste ESS < seuil*k au sens strict, donc il ne")
print("   reechantillonne pas et les quatre lignees survivent intactes)")
