"""H. Ce qui peut changer l'ESS, et ce qui ne le peut pas.

Le poids est exp(lambda * r) normalise. Toute transformation de r qui ajoute la
MEME chose a toutes les particules disparait a la normalisation : centrer la
reward, retrancher sa moyenne, retrancher le max courant partage, retrancher la
reward du pas precedent quand les particules sont des clones -- rien de tout cela
ne bouge l'ESS d'un iota. Seules comptent les operations qui compriment les ECARTS
entre particules : baisser lambda, ou creer des ex aequo comme le fait le plancher.

Cela se verifie en trois lignes, et cela dit d'avance quelles pistes sont mortes.
"""
import numpy as np
from smc.weights import normalize_logw, ess
import torch, json
from pathlib import Path

r = torch.tensor([-2.2531, -0.4344, -0.6896, -0.8958])
LAM = 10.0
base = ess(normalize_logw(LAM * r)[0])
print(__doc__.splitlines()[0], "\n")
print(f"  r = {r.tolist()}, lambda = {LAM}")
print(f"  ESS telle quelle                          : {base:.4f}")
for nom, rr in (("centree (r - moyenne)", r - r.mean()),
                ("decalee de +100", r + 100),
                ("moins le max courant partage (0.5)", r - 0.5),
                ("normalisee en ecart type", (r - r.mean()) / r.std()),
                ("planchee a 0 (reward_min_value)", torch.clamp(r, min=0.0)),
                ("lambda divise par 5", r * 0.2)):
    print(f"  {nom:42s}: {ess(normalize_logw(LAM * rr)[0]):.4f}")
print("\n  Les trois premieres lignes sont identiques a la premiere au bit pres : un")
print("  decalage partage ne fait rien. La normalisation en ecart type, elle, change")
print("  l'echelle donc l'ESS, mais elle change aussi la cible exp(lambda * r(x_0)).")

# --- le lambda qui tiendrait l'ESS a k/2, pas par pas, depuis les runs reels ---
from smc.fk import bisect_lambda
R = Path(__file__).resolve().parent.parent / "results"
runs = json.loads((R / "sd_variants" / "fk4_stat.json").read_text())["runs"]
print(f"\n  Le lambda_t qui tiendrait l'ESS a k/2 = 2, calcule par smc.fk.bisect_lambda sur")
print(f"  les r_phi reellement enregistres ({len(runs)} runs) :")
for m, t in enumerate([80, 60, 40, 20, 0]):
    lams = []
    for run in runs:
        rt = torch.tensor(run["r_at_schedule"][m])
        lams.append(bisect_lambda(torch.zeros(4), rt, torch.zeros(4), 2.0, 100.0, 10.0))
    lams = np.array(lams)
    print(f"    t = {t:3d} : median {np.median(lams):6.2f}   Q1 {np.percentile(lams,25):6.2f}  "
          f"Q3 {np.percentile(lams,75):6.2f}   (le run tourne a 10.0 partout)")
print("  Lire : le lambda qui garderait deux particules vivantes vaut environ 2 au premier")
print("  pas et monte ensuite, parce que l'etendue de r_phi se resserre. Le run fait")
print("  l'inverse : lambda constant a 10, donc une pression maximale la ou l'etendue")
print("  est la plus large et le signal le plus faible.")
