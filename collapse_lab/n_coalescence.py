"""N. Le modele de coalescence : les lignees finales se lisent-elles sur les poids seuls ?

Pour chaque run de out/probe.json, on rejoue la genealogie depuis les poids enregistres a
chaque pas planifie (logG_at_schedule, logW reconstruit comme dans r_solutions.py), sans
rien savoir du pilotage : la loi du vecteur des racines par case se propage pas a pas,
exactement, sous deux resamplers.
  systematic : le peigne de smc/resampling.py, integre sur u ~ U(0, 1/k) (les affectations
               distinctes sont en nombre fini, on les enumere avec leur mesure) ; applique
               aux pas ou le run a reechantillonne (ESS < seuil * k).
  multinomial : le tirage des auteurs, a CHAQUE pas planifie non terminal, poids uniformes
               compris (leur boucle non adaptative ne teste pas l'ESS) : 4^k affectations.
Sortie : E[racines distinctes a la fin] et P(une seule racine) par run, moyennes par bras,
contre l'observe. Limite, a dire : les poids enregistres sont ceux de la genealogie realisee ;
la prediction est conditionnelle a ces poids, pas une loi a priori. Un point fin : a poids
uniformes le peigne est l'identite, le multinomial ne l'est pas (E[distincts] = k(1-(1-1/k)^k)
= 2.73 a k = 4) ; la coalescence sous systematic est portee par la dispersion des poids, sous
multinomial elle existe aussi a plat.
Controles : premier pas seul de ctl = 1.835 (b_comb.py) ; lam0 = 4.000 exactement.
Prediction pour thr05 (a lancer) : les poids de floor, regle "reechantillonne si ESS < k/2".
"""
import itertools, json
from pathlib import Path
import numpy as np

LAB = Path(__file__).resolve().parent
runs = json.loads((LAB / "out" / "probe.json").read_text())["runs"]
par = {}
for r in runs:
    if "logG_at_schedule" in r:        # six ctl restaures depuis ref100 n'ont pas les poids (ASSESSMENT.md)
        par.setdefault(r["arm"], {})[r["prompt_id"]] = r
ORDRE = [a for a in ("ctl", "late", "adapt", "floor", "lam2", "fadapt", "floor2", "lam0",
                     "stat0", "multi", "vae", "idx", "R1", "thr05", "rise") if a in par]


def softmax(lg):
    w = np.exp(lg - lg.max())
    return w / w.sum()


def affectations_peigne(w):
    """[(proba, idx)] : les affectations distinctes du peigne systematique quand u parcourt
    [0, 1/k), avec la mesure de u qui les produit. cumw[-1] force a 1 comme dans le code."""
    k = len(w)
    cumw = np.cumsum(w); cumw[-1] = 1.0
    bornes = {0.0, 1.0 / k}
    for c in cumw[:-1]:
        for j in range(k):
            b = c - j / k
            if 0.0 < b < 1.0 / k:
                bornes.add(float(b))
    bornes = sorted(bornes)
    out = []
    for lo, hi in zip(bornes[:-1], bornes[1:]):
        u = 0.5 * (lo + hi)
        idx = tuple(int(np.searchsorted(cumw, u + j / k, side="left")) for j in range(k))
        out.append(((hi - lo) * k, idx))
    return out


def affectations_multinomial(w):
    k = len(w)
    return [(float(np.prod([w[i] for i in idx])), idx) for idx in itertools.product(range(k), repeat=k)]


def propager(etats, affs):
    """etats : {tuple racines par case: proba}. Une affectation idx copie la case idx[j] en j."""
    nouveaux = {}
    for s, p in etats.items():
        for q, idx in affs:
            if q < 1e-12:
                continue
            t = tuple(s[i] for i in idx)
            nouveaux[t] = nouveaux.get(t, 0.0) + p * q
    return nouveaux


def pas_du_run(r, seuil=None):
    """Les poids (normalises) a chaque pas planifie non terminal, et si le run y a
    reechantillonne. logW repart de zero apres un reechantillonnage, comme dans fk_steer."""
    lg = np.array(r["logG_at_schedule"]); k = r["k"]
    seuil = r["threshold"] if seuil is None else seuil
    logW = np.zeros(k); out = []
    for m in range(len(lg) - 1):
        logW = logW + lg[m]
        w = softmax(logW)
        ess = 1 / (w ** 2).sum()
        res = (lg[m].max() - lg[m].min()) > 1e-9 if seuil >= 1.0 and seuil <= 1.0 else ess < seuil * k
        if seuil > 1.0:
            res = True
        out.append((w, res))
        if res:
            logW = np.zeros(k)
    return out


def predire(r, resampler, seuil=None, premier_pas_seul=False):
    k = r["k"]
    etats = {tuple(range(k)): 1.0}
    pas = pas_du_run(r, seuil)
    if premier_pas_seul:
        pas = pas[:1]
    n_res = 0.0
    for w, res in pas:
        if resampler == "systematic":
            if not res:
                continue
            affs = affectations_peigne(w)
        else:
            affs = affectations_multinomial(w)   # a chaque pas, poids plats compris
        n_res += 1
        etats = propager(etats, affs)
    e_dist = sum(p * len(set(s)) for s, p in etats.items())
    p_un = sum(p for s, p in etats.items() if len(set(s)) == 1)
    return e_dist, p_un, n_res


print(__doc__.splitlines()[0], "\n")

# --- 0. controles -----------------------------------------------------------------
from commun import fk_avec_r
c = par["ctl"]
memes = [r["prompt_id"] for r in fk_avec_r() if r["prompt_id"] in c]   # les 20 prompts de b_comb.py
premier = np.mean([predire(c[p], "systematic", premier_pas_seul=True)[0] for p in memes])
print(f"0. controles : premier pas seul de ctl, les {len(memes)} prompts de fk4_stat = {premier:.3f} attendu 1.835 (b_comb.py)")
if "lam0" in par:
    print(f"   lam0 : E[racines] = {np.mean([predire(r, 'systematic')[0] for r in par['lam0'].values()]):.3f} attendu 4.000")
    print(f"   lam0 sous multinomial a chaque pas : {np.mean([predire(r, 'multinomial')[0] for r in par['lam0'].values()]):.3f}"
          f" (drift neutre, 4 pas a poids plats ; un seul pas donnerait {4 * (1 - (1 - 1 / 4) ** 4):.3f})")

# --- 1. predit contre observe, par bras ---------------------------------------------
print(f"\n1. racines distinctes a la fin : observe contre predit depuis les poids")
print(f"   {'bras':7s} {'n':>3s} | {'obs':>5s} {'sys':>5s} {'mult':>5s} | {'P(1) obs':>8s} {'P(1) sys':>8s} | {'reech obs':>9s} {'|err| run':>9s} {'corr':>5s}")
resume = {}
for a in ORDRE:
    d = par[a]; ps = sorted(d)
    obs = np.array([d[p]["n_lineages"] for p in ps], float)
    sys_ = np.array([predire(d[p], "systematic") for p in ps])
    mult = np.array([predire(d[p], "multinomial")[0] for p in ps])
    # la colonne qui s'applique au bras est celle de son resampler (multinomial pour multi et R1)
    multinomial = d[ps[0]].get("resampler", "systematic") == "multinomial"
    pred = mult if multinomial else sys_[:, 0]
    corr = np.corrcoef(pred, obs)[0, 1] if obs.std() > 0 and pred.std() > 0 else float("nan")
    resume[a] = (obs.mean(), sys_[:, 0].mean(), mult.mean())
    print(f"   {a:7s} {len(ps):3d} | {obs.mean():5.2f} {sys_[:, 0].mean():5.2f}{'*' if not multinomial else ' '}{mult.mean():5.2f}{'*' if multinomial else ' '}| "
          f"{(obs == 1).mean():8.0%} {sys_[:, 1].mean():8.0%} | {np.mean([d[p]['n_resamplings'] for p in ps]):9.2f} "
          f"{np.abs(pred - obs).mean():9.2f} {corr:5.2f}")
print("   sys = peigne aux pas ou le run a reechantillonne ; mult = multinomial a chaque pas planifie ;")
print("   * = la colonne du resampler que le bras a reellement utilise (|err| et corr se lisent sur elle).")
print("   |err| run = ecart absolu moyen entre E[racines] predite et l'observe, run par run (l'observe est un tirage).")

# --- 2. la prediction pre-enregistree pour thr05 ---------------------------------------
if "floor" in par:
    f = par["floor"]; ps = sorted(f)
    pred = np.array([predire(f[p], "systematic", seuil=0.5) for p in ps])
    print(f"\n2. thr05 predit depuis les poids de floor ({len(ps)} prompts), regle ESS < k/2 :"
          f" E[racines] {pred[:, 0].mean():.2f}, P(1 racine) {pred[:, 1].mean():.0%}, reechantillonnages {pred[:, 2].mean():.2f}")
    if "thr05" in par:
        t = par["thr05"]; obs = np.array([t[p]["n_lineages"] for p in sorted(t)], float)
        print(f"   thr05 observe ({len(obs)} prompts) : {obs.mean():.2f} racines, P(1) {(obs == 1).mean():.0%}, "
              f"reech {np.mean([t[p]['n_resamplings'] for p in t]):.2f}")

# --- 3. l'ecart entre les deux resamplers, sur les memes poids -----------------------------
if "ctl" in par:
    o, s, m = resume["ctl"]
    print(f"\n3. ctl : le multinomial des auteurs, sur les memes poids, laisserait {m:.2f} racines contre {s:.2f}"
          f" au peigne (observe {o:.2f}) : le depot de reference s'effondre au moins autant, plancher mis a part.")
