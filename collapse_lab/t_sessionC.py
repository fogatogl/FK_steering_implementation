"""T. Session C : ctl, lam0 et floor2 a 100 prompts dans un seul processus (out/probe_C.json).

Le seul fichier ou l'appariement par case tient : memes x_T, meme session, donc lam0[j] est
la continuation libre de la racine j que ctl et floor2 voient. Confronte les predictions de
docs/protocol_sd.md ("Pre-registration of session C") et recalcule les constats 3 et 5 bis
de FINDINGS.md sur cet appariement :
  - constat 3 : le classement des r_phi de ctl a t = 80 predit-il l'ir final libre (lam0) ?
    Kendall tau par prompt, top-1 ;
  - constat 5 bis : A = meilleure racine libre (max lam0), M = racine moyenne, B = la racine que
    ctl garde, lue dans lam0, C = ce que ctl en tire (ir_max). A - B est ce que l'effondrement
    coute en racine, C - B ce que le pilotage rend.
Controles : ctl_C contre sd_ref_fields100.json (autre session) par prompt, correlation par case.
"""
import itertools, json
from pathlib import Path
import numpy as np

from commun import bon4, load

LAB = Path(__file__).resolve().parent
P = LAB / "out" / "probe_C.json"
runs = json.loads(P.read_text())["runs"] if P.exists() else json.loads(
    Path("/home/onyxia/work/diffusion-models/collapse_lab/out/probe_C.json").read_text())["runs"]
par = {}
for r in runs:
    par.setdefault(r["arm"], {})[r["prompt_id"]] = r
ref = {r["prompt_id"]: r for r in load("sd_ref_fields100.json") if r["sampler"] == "fk4" and r["seed"] == 2024}
bon = bon4()
rng = np.random.default_rng(0)


def se(x):
    x = np.asarray(x, float)
    return f"{x.mean():+.3f} +/- {x.std(ddof=1) / len(x) ** .5:.3f} (n={len(x)})"


def kendall(a, b):
    c = d = 0
    for i, j in itertools.combinations(range(len(a)), 2):
        s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
        c += s > 0; d += s < 0
    return (c - d) / (c + d) if c + d else 0.0


ctl, lam0, fl2 = par.get("ctl", {}), par.get("lam0", {}), par.get("floor2", {})
print(__doc__.splitlines()[0])
print(f"  bras : ctl {len(ctl)}, lam0 {len(lam0)}, floor2 {len(fl2)}\n")

# --- controles ---------------------------------------------------------------
com = sorted(set(ctl) & set(ref))
if com:
    a = np.array([ctl[p]["ir"] for p in com]).ravel(); b = np.array([ref[p]["ir"] for p in com]).ravel()
    print(f"0. ctl_C contre ref100 (21/09), {len(com)} prompts : ir_max {se([ctl[p]['ir_max'] - ref[p]['ir_max'] for p in com])},"
          f" correlation par case {np.corrcoef(a, b)[0, 1]:.2f}, meme racine gardee {np.mean([set(ctl[p]['root_slots']) == set(ref[p]['root_slots']) for p in com]):.0%}")
com = sorted(set(lam0) & set(bon))
if com:
    a = np.array([lam0[p]["ir"] for p in com]).ravel(); b = np.array([bon[p]["ir"] for p in com]).ravel()
    print(f"   lam0_C contre bon4 (20/09), {len(com)} prompts : ir_max {se([lam0[p]['ir_max'] - bon[p]['ir_max'] for p in com])},"
          f" correlation par case {np.corrcoef(a, b)[0, 1]:.2f} (predit < 0.7)")

# --- floor2 contre ctl, apparie par x_T dans la session -----------------------------
com = sorted(set(fl2) & set(ctl))
if com:
    d = np.array([fl2[p]["ir_max"] - ctl[p]["ir_max"] for p in com])
    idx = rng.integers(0, len(d), (4000, len(d))); m = d[idx].mean(1)
    print(f"\n1. floor2 - ctl (x_T apparies), {len(com)} prompts : ir_max {d.mean():+.3f} [{np.percentile(m, 2.5):+.3f}, {np.percentile(m, 97.5):+.3f}]"
          f" (predit [-0.14, -0.02]) ; ir moyen {se([np.mean(fl2[p]['ir']) - np.mean(ctl[p]['ir']) for p in com])}")
    print(f"   floor2 : {np.mean([fl2[p]['n_lineages'] for p in com]):.2f} racines (predit 2.8-3.1), {np.mean([fl2[p]['n_lineages'] == 1 for p in com]):.0%} a une racine (predit 5-10 %),"
          f" div_pix {np.mean([fl2[p]['div_pix'] for p in com]):.3f} ; ctl : {np.mean([ctl[p]['n_lineages'] for p in com]):.2f} racines, {np.mean([ctl[p]['n_lineages'] == 1 for p in com]):.0%}")
    print(f"   floor2 - bon4 ir_max {se([fl2[p]['ir_max'] - bon[p]['ir_max'] for p in com if p in bon])} ; ctl - bon4 {se([ctl[p]['ir_max'] - bon[p]['ir_max'] for p in com if p in bon])}")

# --- constat 3 : l'information du premier pas, sur l'appariement valide -------------------
com = sorted(set(ctl) & set(lam0))
if com:
    tau = np.array([kendall(ctl[p]["r_at_schedule"][0], lam0[p]["ir"]) for p in com])
    top = np.array([int(np.argmax(ctl[p]["r_at_schedule"][0]) == np.argmax(lam0[p]["ir"])) for p in com])
    print(f"\n2. constat 3 recalcule : Kendall tau entre r_phi(t=80) de ctl et l'ir libre (lam0) de la meme racine :"
          f" {tau.mean():+.3f} +/- {tau.std(ddof=1) / len(tau) ** .5:.3f} (predit < 0.15) ; top-1 {top.mean():.0%} contre 25 % au hasard, n={len(com)}")

    # --- constat 5 bis : la decomposition A / M / B / C sur lam0 -----------------------------
    A = np.array([max(lam0[p]["ir"]) for p in com])
    M = np.array([np.mean(lam0[p]["ir"]) for p in com])
    B = np.array([np.mean([lam0[p]["ir"][j] for j in set(ctl[p]["root_slots"])]) for p in com])   # la ou les racines gardees, lues libres
    C = np.array([ctl[p]["ir_max"] for p in com])
    print(f"\n3. constat 5 bis recalcule (n={len(com)}) : A meilleure racine libre {A.mean():+.3f}, M racine moyenne {M.mean():+.3f},"
          f" B racine(s) gardee(s) par ctl, libre {B.mean():+.3f}, C ce que ctl en tire {C.mean():+.3f}")
    print(f"   B - M {se(B - M)} (la racine gardee vaut-elle mieux que le hasard) ; A - B {se(A - B)} (predit [0.25, 0.50]) ;"
          f" C - B {se(C - B)} (le pilotage) ; C - A {se(C - A)} (= ctl - lam0 sur ir_max)")
    rang = np.array([1 + sum(v > lam0[p]["ir"][j] for v in lam0[p]["ir"]) for p in com for j in set(ctl[p]["root_slots"])])
    print(f"   rang moyen de la racine gardee parmi les 4 libres : {rang.mean():.2f} (1.5 au hasard sur 4 pour un rang 1-4 : 2.5)")
