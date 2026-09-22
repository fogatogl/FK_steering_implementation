"""A. L'ESS loggee est-elle exactement celle que lam * r_phi implique ?

Detecteur de bug : si l'ESS recalculee depuis r_at_schedule[0] colle a
ess_at_schedule[0], l'effondrement est arithmetique et non un defaut de
weights.py / resampling.py. Aucun import de smc/ n'est necessaire ici : a gate
vide les trois potentiels donnent logG = lam * r_t (smc/fk.py, _potential_terms).
"""
import json, math
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent.parent / "results"
load = lambda p: json.loads((R / p).read_text())["runs"]


def poids(logw):
    lw = np.asarray(logw, float)
    lw = lw - lw.max(-1, keepdims=True)
    w = np.exp(lw)
    return w / w.sum(-1, keepdims=True)


ess = lambda w: 1.0 / (w ** 2).sum(-1)

print(__doc__.splitlines()[0], "\n")

for tag, f in (("fk4_stat", "sd_variants/fk4_stat.json"),
               ("fk4_diff", "sd_variants/fk4_diff.json"),
               ("ref100", "sd_ref_fields100.json")):
    runs = load(f)
    avec = [r for r in runs if "r_at_schedule" in r]
    if not avec:
        print(f"{tag:9s} {len(runs):3d} runs, aucun r_at_schedule (fichier anterieur au 21/09)")
        continue
    r0 = np.array([r["r_at_schedule"][0] for r in avec])
    lam = np.array([[r["lam"]] for r in avec])
    e = ess(poids(lam * r0))
    loggee = np.array([r["ess_at_schedule"][0] for r in avec])
    d = np.abs(e - loggee)
    print(f"{tag:9s} {len(avec):3d} runs avec r_at_schedule : "
          f"|ESS_recalc - ESS_loggee| median {np.median(d):.2e}  max {d.max():.2e}  "
          f"({(d < 1e-3).sum()}/{len(d)} sous 1e-3)")

# --- echelle, sur les 40 runs qui portent r_at_schedule ---
# Un seul fichier : fk4_diff porte les memes 20 prompts aux memes x_T (voir commun.py).
avec = load("sd_variants/fk4_stat.json")
r = np.array([x["r_at_schedule"] for x in avec])          # (n, 5 lignes, 4 slots)
ts = [80, 60, 40, 20, 0]
q = lambda v: f"{np.percentile(v, 25):6.3f} |{np.median(v):6.3f} |{np.percentile(v, 75):6.3f}"

print("\nEchelle de la reward guide r_phi(predict_x0), 20 runs, k=4   (Q1 | med | Q3)")
print(f"{'':14s}{'niveau':>22s}{'etendue max-min':>24s}{'ecart 1er-2e':>24s}"
      f"{'ESS a lam=10':>22s}")
for m, t in enumerate(ts):
    x = r[:, m, :]
    srt = np.sort(x, 1)
    e = ess(poids(10.0 * x))
    print(f"  t = {t:<9d}{q(x.ravel())}{q(x.max(1) - x.min(1)):>24s}"
          f"{q(srt[:, -1] - srt[:, -2]):>24s}{q(e):>22s}")

print("\n  ESS loggee (mediane) par ligne :",
      "  ".join(f"t={t}:{np.median([x['ess_at_schedule'][m] for x in avec]):.2f}"
                for m, t in enumerate(ts)))
print("  poids max a lam=10 (mediane) :",
      "  ".join(f"t={t}:{np.median(poids(10.0 * r[:, m, :]).max(1)):.3f}" for m, t in enumerate(ts)))
