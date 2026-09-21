"""Rejoue, depuis les fichiers et depuis potentials() lui-meme, chaque constat
ecrit dans docs/max_potential.md sur la sous-performance du potentiel max.

Sans GPU, sans relancer un run : tout sort de results/*.json et d'appels directs
a smc.fk.potentials. Un constat qui ne se rejoue plus est un constat a reecrire,
pas un test a assouplir.

Depuis la racine : `python scripts/verify_max_findings.py`
"""
import json
import statistics as st
from pathlib import Path

import torch

from smc.fk import potentials

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
K = 4


def charge(nom):
    return json.loads((RES / nom).read_text())["runs"]


def dit(ok, titre, detail):
    print(f"  [{'ok ' if ok else 'RATE'}] {titre}\n         {detail}")
    return ok


def ratchet():
    """Le premier pas du calendrier compare des niveaux, les suivants des increments."""
    lam = 10.0
    r = torch.tensor([0.5, 0.9, 1.4, 0.7])
    vide = torch.full((K,), -float("inf"))
    premier, gate = potentials(r, vide, lam, 0.0, "max", False, None, "terminal")
    attendu_premier = lam * r
    suivant, _ = potentials(r, gate, lam, 0.0, "max", False, None, "terminal")
    attendu_suivant = lam * torch.clamp(r - gate, min=0.0)

    a = dit(torch.allclose(premier, attendu_premier),
            "premier pas : logG = lam * r_t, le seul classement sur les niveaux",
            f"logG {[round(v, 2) for v in premier.tolist()]}")
    b = dit(torch.allclose(suivant, attendu_suivant),
            "pas suivants : logG = lam * max(0, r_t - max courant), un cliquet",
            f"logG {[round(v, 2) for v in suivant.tolist()]} sur un max deja a "
            f"{[round(v, 2) for v in gate.tolist()]}")
    sature, _ = potentials(torch.full((K,), 0.3), torch.full((K,), 1.5), lam,
                           0.0, "max", False, None, "terminal")
    c = dit(bool((sature == 0).all()),
            "max sature et partage entre clones : logG nul, poids uniformes",
            f"logG {[round(v, 2) for v in sature.tolist()]}, ESS {K:.2f} sur {K}")
    return a and b and c


def noop_egale_resamplings():
    """4 - (pas a poids uniformes) == n_resamplings, sur les 300 runs fk4."""
    fk = [r for r in charge("sd_baseline.json") if r["sampler"] == "fk4"]
    # egalite exacte : un ESS de 3.9997 n'est pas uniforme et reechantillonne bien.
    accord = sum(1 for r in fk
                 if sum(1 for e in r["ess_at_schedule"][:4] if e == K)
                 == 4 - r["n_resamplings"])
    return dit(accord >= 299,
               "un reechantillonnage saute est exactement un pas a poids uniformes",
               f"{accord}/{len(fk)} runs, moyenne n_resamplings "
               f"{st.mean(r['n_resamplings'] for r in fk):.3f} sur 4")


def identique_a_best_of_k():
    """La ou max ne reechantillonne jamais, son r_max est celui de best-of-k."""
    runs = charge("sweep_k_max_classifier.json")
    bon = {(r["k"], r["seed"]): r["r_max"] for r in runs if r["method"] == "best_of_n"}
    paires = [(r, bon[(r["k"], r["seed"])]) for r in runs
              if r.get("potential") == "max" and r["n_resamplings"] == 0
              and (r["k"], r["seed"]) in bon]
    # a la precision du float32 stocke, pas au bit pres : l'ecart est du round-trip.
    ecarts = [abs(r["r_max"] - b) for r, b in paires]
    return dit(len(paires) > 0 and max(ecarts) < 2e-6,
               "max sans reechantillonnage redonne best-of-k a la precision du float",
               f"{len(paires)} paires, ecart maximum {max(ecarts):.1e}")


def racine_au_hasard():
    """La racine gardee contient-elle l'argmax de best-of-4 plus souvent qu'au hasard."""
    bon = {(r["prompt_id"], r["seed"]): r
           for r in charge("sd_baseline.json") if r["sampler"] == "bon4"}
    s60 = [r for r in charge("sd_s60_full.json") if r.get("root_slots")]
    rate = [max(range(len(bon[c]["ir"])), key=lambda j: bon[c]["ir"][j])
            not in set(r["root_slots"])
            for r in s60 if (c := (r["prompt_id"], r["seed"])) in bon]
    hasard = [1 - len(set(r["root_slots"])) / K for r in s60]
    return dit(abs(st.mean(rate) - st.mean(hasard)) < 0.05,
               "la racine gardee n'est pas meilleure que le hasard",
               f"{st.mean(rate):.1%} de ratees contre {st.mean(hasard):.0%} au hasard, "
               f"n={len(rate)}")


def ecart_intra_run():
    """Les k finales de fk4 sont des quasi-copies ; celles de bon4 ne le sont pas."""
    runs = charge("sd_baseline.json")
    ecart = {n: st.mean(max(r["ir"]) - min(r["ir"])
                        for r in runs if r["sampler"] == n) for n in ("fk4", "bon4")}
    return dit(ecart["fk4"] < 0.4 * ecart["bon4"],
               "l'ecart de reward entre les k finales s'effondre sous max",
               f"fk4 {ecart['fk4']:.3f} contre bon4 {ecart['bon4']:.3f}")


def ess_ne_predit_pas():
    """L'ESS du premier pas ne predit pas le gain : rang de Spearman quasi nul."""
    runs = charge("sd_baseline.json")
    par = {}
    for r in runs:
        par.setdefault((r["prompt_id"], r["seed"]), {})[r["sampler"]] = r
    couples = [(d["fk4"]["ess_at_schedule"][0], d["fk4"]["ir_max"] - d["bon4"]["ir_max"])
               for d in par.values() if "fk4" in d and "bon4" in d]
    x, y = zip(*couples)
    rang = lambda v: [sorted(v).index(a) for a in v]  # noqa: E731 - suffisant ici
    rx, ry = rang(x), rang(y)
    n = len(couples)
    rho = (sum(a * b for a, b in zip(rx, ry)) / n - st.mean(rx) * st.mean(ry)) / (
        st.pstdev(rx) * st.pstdev(ry))
    return dit(abs(rho) < 0.15,
               "l'ESS du premier pas ne predit pas le gain sur best-of-4",
               f"Spearman {rho:+.3f} sur {n} runs")


def ess_monotone_en_lambda():
    """L'ESS du premier pas descend quand lambda monte : exp(lam * ecart)."""
    lus = []
    for tag, lam in (("L2", 2), ("L5", 5), (None, 10), ("L20", 20)):
        if tag is None:
            fk = [r for r in charge("sd_baseline.json") if r["sampler"] == "fk4"]
        else:
            fk = charge(f"sd_variants/{tag}.json")
        lus.append((lam, st.median(r["ess_at_schedule"][0] for r in fk)))
    valeurs = [e for _, e in lus]
    return dit(valeurs == sorted(valeurs, reverse=True),
               "l'ESS du premier pas est monotone en lambda",
               "  ".join(f"lam={l}:{e:.2f}" for l, e in lus))


print(__doc__.splitlines()[0])
print()
resultats = [ratchet(), noop_egale_resamplings(), identique_a_best_of_k(),
             racine_au_hasard(), ecart_intra_run(), ess_ne_predit_pas(),
             ess_monotone_en_lambda()]
print(f"\n{sum(resultats)}/{len(resultats)} constats rejoues")
