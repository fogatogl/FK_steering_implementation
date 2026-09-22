"""F. Depouillement de la sonde : quand les lignees meurent, et ce que le plancher change.

Cinq bras sur les memes 20 prompts et les memes x_T (seed 2024, seed_effective =
2024000 + i, identique a scripts/run_sd_baseline.py). Tout est apparie par prompt.
"""
import json, itertools
from pathlib import Path
import numpy as np

LAB = Path(__file__).resolve().parent
R = LAB.parent / "results"
runs = json.loads((LAB / "out" / "probe.json").read_text())["runs"]
par = {}
for r in runs:
    par.setdefault(r["arm"], {})[r["prompt_id"]] = r

base = json.loads((R / "sd_baseline.json").read_text())["runs"]
bon = {r["prompt_id"]: r for r in base if r["sampler"] == "bon4" and r["seed"] == 2024}
# Le fk4 de sd_baseline.json date du 20/09 et precede 5025180 : il ne decrit plus le
# code que la sonde fait tourner. La ligne FK de reference est sd_ref_fields100.json.
ref100 = json.loads((R / "sd_ref_fields100.json").read_text())["runs"]
fk4 = {r["prompt_id"]: r for r in ref100 if r["sampler"] == "fk4" and r["seed"] == 2024}
ts = [80, 60, 40, 20, 0]
print(__doc__.splitlines()[0], "\n")
print(f"  bras presents : " + ", ".join(f"{a} ({len(d)} prompts)" for a, d in par.items()))

# --- 0. controles du pilote -------------------------------------------------
if "ctl" in par:
    com = sorted(set(par["ctl"]) & set(fk4))
    d = np.array([par["ctl"][c]["ir_max"] - fk4[c]["ir_max"] for c in com])
    print(f"\n0a. controle du pilote : ctl contre le fk4 de sd_ref_fields100.json, {len(com)} prompts")
    print(f"    ir_max, difference appariee {d.mean():+.4f} +/- {d.std(ddof=1)/max(len(d),1)**.5:.4f} "
          f"(identique a {np.abs(d).max():.4f} pres au pire)")
if "lam0" in par:
    com = sorted(set(par["lam0"]) & set(bon))
    # case par case : c'est l'hypothese (case j de fk = case j de bon4), et un tri
    # ne testerait que l'egalite des ensembles.
    e = np.array([np.abs(np.array(par["lam0"][c]["ir"]) - np.array(bon[c]["ir"])).max() for c in com])
    print(f"\n0b. controle d'appariement : lam=0 doit redonner les 4 tirages libres de bon4")
    print(f"    ecart max sur les 4 ir, par prompt : median {np.median(e):.4f}, max {e.max():.4f}")
    print(f"    -> si c'est nul, la case j de fk4 et la case j de bon4 partagent bien x_T,")
    print(f"       ce qui est l'hypothese de la mesure d'information (script c_information.py)")

# --- 0c. le detecteur de bug, etendu aux cinq lignes ------------------------
# A seuil 1.0 logW repart de zero a chaque reechantillonnage, et vaut zero aux pas
# non planifies : l'ESS lue au pas m doit donc etre exactement celle de logG_m seul.
print("\n0c. ESS recalculee depuis logG contre ESS enregistree, les cinq lignes")
for a in sorted(par):
    d, pire = par[a], 0.0
    for c in d:
        lg = np.array(d[c]["logG_at_schedule"])
        w = np.exp(lg - lg.max(1, keepdims=True))
        w /= w.sum(1, keepdims=True)
        pire = max(pire, float(np.abs(1 / (w ** 2).sum(1) - np.array(d[c]["ess_at_schedule"])).max()))
    print(f"    {a:8s} ecart max sur {len(d)} runs x 5 lignes : {pire:.2e}"
          + ("   <- arithmetique exacte" if pire < 1e-3 else "   <- A CREUSER"))

# --- 1. ou meurent les lignees ---------------------------------------------
print(f"\n1. nombre de racines x_T distinctes apres chaque pas planifie (moyenne sur les prompts)")
print(f"   {'bras':8s} {'lam':>5s} {'plancher':>9s} | " + " ".join(f"t={t:<5d}" for t in ts)
      + f" | {'resampl':>8s} {'div_pix':>8s} {'ir_max':>8s}")
for a in ("ctl", "floor", "adapt", "fadapt", "lam2", "floor2", "lam0"):
    if a not in par:
        continue
    d = par[a]
    tr = np.array([d[c]["lineages_trace"] for c in sorted(d)])
    print(f"   {a:8s} {d[sorted(d)[0]]['lam']:5.1f} {str(d[sorted(d)[0]]['plancher']):>9s} | "
          + " ".join(f"{v:<7.2f}" for v in tr.mean(0))
          + f" | {np.mean([d[c]['n_resamplings'] for c in d]):8.2f} "
          f"{np.mean([d[c]['div_pix'] for c in d]):8.4f} "
          f"{np.mean([d[c]['ir_max'] for c in d]):+8.4f}")
    if d[sorted(d)[0]].get("adaptatif"):
        lam_m = np.median([d[c]["lam_at_schedule"] for c in sorted(d)], axis=0)
        print(f"   {'':8s} {'lam_t':>5s} {'median':>9s} | " + " ".join(f"{v:<7.2f}" for v in lam_m))

# --- 2. le plancher : combien de pas rend-il inertes ------------------------
print(f"\n2. pas planifies ou les poids sortent uniformes (aucune selection)")
print(f"   {'bras':8s} | " + " ".join(f"t={t:<5d}" for t in ts))
for a in ("ctl", "floor", "adapt", "fadapt", "lam2", "floor2"):
    if a not in par:
        continue
    d = par[a]
    lg = np.array([d[c]["logG_at_schedule"] for c in sorted(d)])      # (n, 5, k)
    unif = (lg.max(2) - lg.min(2)) < 1e-9
    print(f"   {a:8s} | " + " ".join(f"{v:<7.0%}" for v in unif.mean(0)))

# --- 3. l'increment est-il un niveau ? --------------------------------------
print(f"\n3. logG - lam * r_phi est-il CONSTANT entre les cases ? (alors le poids est exp(lam*r),")
print(f"   un classement sur le niveau, et la soustraction du potentiel ne protege de rien)")
print(f"   {'bras':8s} | " + " ".join(f"t={t:<5d}" for t in ts) + "   (part des runs)")
for a in ("ctl", "floor", "adapt", "fadapt", "lam2", "floor2"):
    if a not in par:
        continue
    d = par[a]
    cst = []
    for c in sorted(d):
        lg = np.array(d[c]["logG_at_schedule"])
        r = np.array(d[c]["r_at_schedule"])
        lam_m = np.array(d[c].get("lam_at_schedule", [d[c]["lam"]] * len(lg)))[:, None]
        ecart = lg - lam_m * r
        cst.append((ecart.max(1) - ecart.min(1)) < 1e-3)
    print(f"   {a:8s} | " + " ".join(f"{v:<7.0%}" for v in np.array(cst).mean(0)))

# --- 4. la reward, appariee -------------------------------------------------
print(f"\n4. reward finale, differences appariees par prompt (ImageReward, max sur les 4)")
ref = par.get("ctl", {})
for a in ("floor", "adapt", "fadapt", "lam2", "floor2", "lam0"):
    if a not in par or not ref:
        continue
    com = sorted(set(par[a]) & set(ref))
    d = np.array([par[a][c]["ir_max"] - ref[c]["ir_max"] for c in com])
    print(f"   {a:8s} - ctl : {d.mean():+.4f} +/- {d.std(ddof=1)/len(d)**.5:.4f}  "
          f"({(d > 0).sum()}/{len(d)} prompts gagnes)")
for a in ("ctl", "floor", "adapt", "fadapt", "lam2", "floor2"):
    if a not in par:
        continue
    com = sorted(set(par[a]) & set(bon))
    d = np.array([par[a][c]["ir_max"] - bon[c]["ir_max"] for c in com])
    print(f"   {a:8s} - bon4: {d.mean():+.4f} +/- {d.std(ddof=1)/len(d)**.5:.4f}  "
          f"({(d > 0).sum()}/{len(d)} prompts gagnes)")

# --- 5. l'information du pas qui tue ---------------------------------------
def kendall(a, b):
    c = dd = 0
    for i, j in itertools.combinations(range(len(a)), 2):
        s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
        c += s > 0; dd += s < 0
    return (c - dd) / (c + dd) if c + dd else 0.0

if "ctl" in par:
    com = sorted(set(par["ctl"]) & set(bon))
    print(f"\n5. information du premier pas : r_phi(t=80) contre l'ir final de la meme racine")
    tau = np.array([kendall(par["ctl"][c]["r_at_schedule"][0], bon[c]["ir"]) for c in com])
    top = np.array([int(np.argmax(par["ctl"][c]["r_at_schedule"][0]) == np.argmax(bon[c]["ir"]))
                    for c in com])
    print(f"   Kendall tau {tau.mean():+.3f} +/- {tau.std(ddof=1)/len(tau)**.5:.3f}, "
          f"top-1 {top.mean():.0%} contre 25 % au hasard, n={len(com)}")
