"""R. Les corrections resolvent-elles l'effondrement ? Les trois criteres d'ASSESSMENT.md.

Lit out/probe.json (tous les bras, n inegaux, file coupee tolerée) et results/. Tout est
apparie par prompt sur l'intersection des prompt_id. Les criteres :
  1. le mecanisme : part des prompts a une seule racine, lignees par pas (alignees sur la
     valeur de t, le bras `late` n'ayant pas t = 80), ESS ponderee des racines a la fin ;
  2. le plafond de la cible : cette ESS contre l'ESS de exp(lam * ir) sur les quatre tirages
     libres de bon4 des memes prompts (l_target_ess.py, restreint aux prompts du bras) ;
  3. le prix : ir_max, ir moyen, ir pondere contre ctl et contre bon4, IC bootstrap 95 %.
"""
import json
from pathlib import Path
import numpy as np

from commun import bon4, load

LAB = Path(__file__).resolve().parent
runs = json.loads((LAB / "out" / "probe.json").read_text())["runs"]
par = {}
for r in runs:
    par.setdefault(r["arm"], {})[r["prompt_id"]] = r
bon = bon4()
fk4 = {r["prompt_id"]: r for r in load("sd_ref_fields100.json")
       if r["sampler"] == "fk4" and r["seed"] == 2024}
ORDRE = [a for a in ("ctl", "late", "adapt", "floor", "lam2", "fadapt", "floor2", "thr05", "rise",
                     "stat0", "multi", "vae", "idx", "R1", "lam0") if a in par]
rng = np.random.default_rng(0)
# six records ctl sont restaures depuis sd_ref_fields100.json sans logG ni lineages_trace
# (ASSESSMENT.md, incident du 23/09) : tout ce qui lit les poids se calcule sur les records complets
complet = lambda r: "logG_at_schedule" in r


def ess(logw):
    w = np.exp(logw - logw.max())
    w /= w.sum()
    return 1 / (w ** 2).sum()


def poids_finaux(r):
    """logW a la fin, reconstruit : a seuil 1.0 il repart de zero a chaque pas non terminal
    ou logG a la moindre etendue (ESS < k au sens strict) ; sous 1.0, ou ESS < seuil * k."""
    lg = np.array(r["logG_at_schedule"])
    e = np.array(r["ess_at_schedule"])
    k = r["k"]
    logW = np.zeros(k)
    for m in range(len(lg)):
        logW = logW + lg[m]
        if m == len(lg) - 1:
            break
        if r["threshold"] >= 1.0:
            resampled = (lg[m].max() - lg[m].min()) > 1e-9
        else:
            resampled = e[m] < r["threshold"] * k
        if resampled:
            logW = np.zeros(k)
    w = np.exp(logW - logW.max())
    return w / w.sum()


def ess_racines(r):
    w = poids_finaux(r)
    roots = np.array(r["root_slots"])
    W = np.array([w[roots == u].sum() for u in np.unique(roots)])
    return 1 / (W ** 2).sum()


def par_t(r):
    """lignees apres chaque pas planifie, indexees par la valeur de t."""
    ts = sorted(r["schedule_t"], reverse=True)
    return dict(zip(ts, r["lineages_trace"]))


def inertes(r):
    lg = np.array(r["logG_at_schedule"])
    ts = sorted(r["schedule_t"], reverse=True)
    return dict(zip(ts, (lg.max(1) - lg.min(1)) < 1e-9))


def boot(d, n=4000):
    d = np.asarray(d, dtype=float)
    if len(d) < 2:
        return d.mean(), np.nan, np.nan
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def fmt(m, lo, hi):
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"


print(__doc__.splitlines()[0])
print("  bras presents : " + ", ".join(f"{a} ({len(par[a])})" for a in ORDRE), "\n")

# --- 0. controles ------------------------------------------------------------
print("0. controles")
if "ctl" in par:
    com = sorted(set(par["ctl"]) & set(fk4))
    d = np.abs([par["ctl"][c]["ir_max"] - fk4[c]["ir_max"] for c in com])
    print(f"   ctl contre le fk4 de sd_ref_fields100.json, {len(com)} prompts : ecart max {d.max():.4f}")
pire_e, pire_w = 0.0, 0.0
for a in ORDRE:
    for r in filter(complet, par[a].values()):
        lg = np.array(r["logG_at_schedule"])
        e_rec = np.array([ess(row) for row in lg])
        pire_e = max(pire_e, float(np.abs(e_rec - np.array(r["ess_at_schedule"])).max())) if r["threshold"] >= 1 else pire_e
        pire_w = max(pire_w, abs(1 / (poids_finaux(r) ** 2).sum() - r["ess_at_schedule"][-1]))
print(f"   ESS par pas recalculee depuis logG (seuil 1.0) : ecart max {pire_e:.1e} sur {len(runs)} runs")
print(f"   poids finaux reconstruits, ESS finale contre ess_at_schedule[-1] : ecart max {pire_w:.1e}")
if "late" in par:
    r = next(iter(par["late"].values()))
    print(f"   late : calendrier {sorted(r['schedule_t'])}, {len(r['lineages_trace'])} pas planifies, "
          f"{r['n_resamplings']} reechantillonnages sur le premier run")

# --- 1. le mecanisme ---------------------------------------------------------
TS = [80, 60, 40, 20, 0]
print(f"\n1. le mecanisme : ou meurent les lignees, et ce qu'il en reste pondere")
print(f"   {'bras':7s} {'n':>3s} {'1 racine':>9s} {'>=3':>5s} | " + " ".join(f"t={t:<4d}" for t in TS)
      + f" | {'ESS pond.':>9s} {'inertes t=80/60/40':>19s} {'div_pix':>8s}")
stats = {}
for a in ORDRE:
    d = par[a]
    ps = sorted(d)
    pc = [c for c in ps if complet(d[c])]
    tr = [par_t(d[c]) for c in pc]
    ine = [inertes(d[c]) for c in pc]
    un = np.mean([d[c]["n_lineages"] == 1 for c in ps])
    trois = np.mean([d[c]["n_lineages"] >= 3 for c in ps])
    er = np.array([ess_racines(d[c]) for c in pc])
    stats[a] = dict(un=un, er=er, lin=np.array([d[c]["n_lineages"] for c in ps]))
    cols = " ".join(f"{np.mean([x[t] for x in tr]):<6.2f}" if t in tr[0] else f"{'-':<6s}" for t in TS)
    ins = "/".join(f"{np.mean([x[t] for x in ine]):.0%}" if t in ine[0] else "-" for t in (80, 60, 40))
    print(f"   {a:7s} {len(ps):3d} {un:9.0%} {trois:5.0%} | {cols} | {er.mean():9.2f} {ins:>19s} "
          f"{np.mean([d[c]['div_pix'] for c in ps]):8.3f}")
print("   `1 racine` : part des prompts dont les quatre images finales descendent du meme x_T.")
print("   `ESS pond.` : 1 / sum_r W_r^2, W_r = poids final total des cases issues de la racine r.")

# --- 2. le plafond de la cible ----------------------------------------------
print(f"\n2. le plafond de la cible : ESS de exp(lam * ir) sur les quatre tirages libres (bon4) des"
      f" memes prompts,\n   contre l'ESS ponderee des racines que le bras rend, et ses lignees brutes")
print(f"   {'bras':7s} {'lam':>5s} {'n':>3s} | {'plafond moy':>11s} {'plafond med':>11s} | {'ESS pond.':>9s} {'lignees':>8s} | ratio pond./plafond")
for a in ORDRE:
    d = par[a]
    ps = sorted(set(d) & set(bon))
    lam = d[ps[0]]["lam"]
    pl = np.array([ess(lam * np.array(bon[c]["ir"])) for c in ps])
    er = np.array([ess_racines(d[c]) for c in ps if complet(d[c])])
    lin = np.array([d[c]["n_lineages"] for c in ps])
    print(f"   {a:7s} {lam:5.1f} {len(ps):3d} | {pl.mean():11.2f} {np.median(pl):11.2f} | {er.mean():9.2f} {lin.mean():8.2f} | {er.mean() / pl.mean():.2f}")
print("   ratio ~ 1 : le bras rend la diversite que sa cible porte, ni plus ni moins. Au-dessus de 1 a lam = 10 :")
print("   il rend des racines que la cible ecrase (visibles si l'on ignore les poids, ce que ir_max et div_pix font).")

# --- 3. le prix --------------------------------------------------------------
print(f"\n3. le prix : differences appariees par prompt, moyenne et IC bootstrap 95 %")


def ir_moy(r):
    return float(np.mean(r["ir"]))


def ir_pond(r):
    return float((poids_finaux(r) * np.array(r["ir"])).sum())


ref = par.get("ctl", {})
print(f"   contre ctl")
print(f"   {'bras':7s} {'n':>3s} | {'ir_max':>26s} | {'ir moyen':>26s} | {'ir pondere':>26s} | gagne")
for a in ORDRE:
    if a == "ctl" or not ref:
        continue
    com = sorted(set(par[a]) & set(ref))
    dm = [par[a][c]["ir_max"] - ref[c]["ir_max"] for c in com]
    da = [ir_moy(par[a][c]) - ir_moy(ref[c]) for c in com]
    dw = [ir_pond(par[a][c]) - ir_pond(ref[c]) for c in com if complet(par[a][c]) and complet(ref[c])]
    print(f"   {a:7s} {len(com):3d} | {fmt(*boot(dm)):>26s} | {fmt(*boot(da)):>26s} | {fmt(*boot(dw)):>26s} | {int(np.sum(np.array(dm) > 0))}/{len(com)}")
print(f"\n   contre bon4 (quatre tirages libres, ir_max = best-of-4)")
print(f"   {'bras':7s} {'n':>3s} | {'ir_max':>26s} | {'ir moyen':>26s} | gagne")
for a in ORDRE:
    com = sorted(set(par[a]) & set(bon))
    dm = [par[a][c]["ir_max"] - bon[c]["ir_max"] for c in com]
    da = [ir_moy(par[a][c]) - ir_moy(bon[c]) for c in com]
    print(f"   {a:7s} {len(com):3d} | {fmt(*boot(dm)):>26s} | {fmt(*boot(da)):>26s} | {int(np.sum(np.array(dm) > 0))}/{len(com)}")

# le prix est-il plat ? un seul chiffre : pour chaque prompt, la moyenne d'ir_max sur tous les
# bras de correction presents, contre ctl. C'est le test du "prix plat" du constat 13, qui
# a plus de puissance que six IC separes qui frolent zero du meme cote.
# les corrections au sens du constat 14, pas les bras de bissection (stat0, multi, vae, idx, R1,
# qui imitent le code publie) ni late (qui ne repare rien) ; floor2 est a n = 34 (ASSESSMENT.md)
corr = [a for a in ("adapt", "floor", "lam2", "fadapt", "floor2", "thr05", "rise") if a in par]
if ref and corr:
    com = sorted(c for c in ref if any(c in par[a] for a in corr))
    dm = [np.mean([par[a][c]["ir_max"] for a in corr if c in par[a]]) - ref[c]["ir_max"] for c in com]
    print(f"\n   prix plat : moyenne d'ir_max sur les bras de correction ({', '.join(corr)}) - ctl, "
          f"n={len(com)} : {fmt(*boot(dm))}")
    dispersion = [np.std([par[a][c]["ir_max"] - ref[c]["ir_max"] for a in corr if c in par[a]]) for c in com]
    print(f"   ecart-type entre bras du prix, par prompt, median : {np.median(dispersion):.3f}")

# entre bras de correction, les paires qui tranchent le classement du constat 14
paires = [("floor2", "lam2"), ("floor2", "fadapt"), ("late", "floor"), ("late", "ctl"), ("late", "floor2"), ("fadapt", "floor")]
print(f"\n   entre bras, apparie : lignees, ESS ponderee, ir_max")
for a, b in paires:
    if a not in par or b not in par:
        continue
    com = sorted(set(par[a]) & set(par[b]))
    if len(com) < 2:
        continue
    dl = [par[a][c]["n_lineages"] - par[b][c]["n_lineages"] for c in com]
    de = [ess_racines(par[a][c]) - ess_racines(par[b][c]) for c in com if complet(par[a][c]) and complet(par[b][c])]
    dm = [par[a][c]["ir_max"] - par[b][c]["ir_max"] for c in com]
    print(f"   {a} - {b:7s} n={len(com):2d} | lignees {fmt(*boot(dl))} | ESS pond. {fmt(*boot(de))} | ir_max {fmt(*boot(dm))}")

# --- 4. verdicts, un par critere -------------------------------------------
print(f"\n4. verdicts (criteres d'ASSESSMENT.md, fixes avant les donnees)")
for a in ORDRE:
    if a in ("ctl", "lam0"):
        continue
    d = par[a]
    ps = sorted(set(d) & set(bon))
    lam = d[ps[0]]["lam"]
    un = stats[a]["un"]
    pl = np.mean([ess(lam * np.array(bon[c]["ir"])) for c in ps])
    er = np.mean([ess_racines(d[c]) for c in ps if complet(d[c])])
    c1 = "OUI" if un < 0.25 else "non"
    c2 = er / pl
    com = sorted(set(d) & set(ref)) if ref else []
    m_ctl, lo_ctl, hi_ctl = boot([d[c]["ir_max"] - ref[c]["ir_max"] for c in com]) if com else (np.nan,) * 3
    m_bon, lo_bon, hi_bon = boot([d[c]["ir_max"] - bon[c]["ir_max"] for c in ps])
    c3_ctl = "IC couvre 0" if lo_ctl <= 0 <= hi_ctl else "IC exclut 0"
    c3_bon = "IC couvre 0" if lo_bon <= 0 <= hi_bon else ("au-dessus" if lo_bon > 0 else "sous")
    print(f"   {a:7s} (1) sous 25 % a une racine : {c1} ({un:.0%})   "
          f"(2) ESS pond. / plafond de sa cible : {c2:.2f} ({er:.2f} / {pl:.2f})   "
          f"(3) ir_max - ctl {m_ctl:+.3f} ({c3_ctl}), ir_max - bon4 {m_bon:+.3f} ({c3_bon})")
print("   (3) se lit avec la ligne `prix plat` ci-dessus : six IC qui frolent zero du meme cote ne sont pas six zeros.")
