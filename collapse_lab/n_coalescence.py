"""N. The coalescence model: can the final lineages be read from the weights alone?

For every run of out/probe.json the genealogy is replayed from the weights recorded at each
scheduled step (logG_at_schedule, logW rebuilt as in r_solutions.py), knowing nothing about
the steering: the law of the root-per-slot vector is propagated step by step, exactly, under
two resamplers.
  systematic : the comb of smc/resampling.py, integrated over u ~ U(0, 1/k) (the distinct
               assignments are finite in number; they are enumerated with their measure);
               applied at the steps where the run resampled (ESS < threshold * k).
  multinomial: the authors' draw, at EVERY non-terminal scheduled step, flat weights
               included (their non-adaptive loop never tests the ESS): 4^k assignments.
Output: E[distinct roots at the end] and P(a single root) per run, averaged per arm, against
the observed value. A limit, to be stated: the recorded weights are those of the realised
genealogy; the prediction is conditional on these weights, not an a priori law. A fine point:
with flat weights the comb is the identity, the multinomial is not (E[distinct] =
k(1-(1-1/k)^k) = 2.73 at k = 4); under systematic the coalescence is carried by the spread of
the weights, under multinomial it exists even when they are flat.
Checks: first step alone of ctl = 1.835 (b_comb.py); lam0 = 4.000 exactly.
Prediction for thr05 (to be run): the weights of floor, rule "resample if ESS < k/2".
"""
import itertools, json
from pathlib import Path
import numpy as np

LAB = Path(__file__).resolve().parent
runs = json.loads((LAB / "out" / "probe.json").read_text())["runs"]
par = {}
for r in runs:
    if "logG_at_schedule" in r:        # six ctl restored from ref100 carry no weights (ASSESSMENT.md)
        par.setdefault(r["arm"], {})[r["prompt_id"]] = r
# sessions C and D (one process each, 100 prompts) enter as their own arms, suffixed
for name, tag in (("probe_C.json", "_C"), ("session_D/probe_D.json", "_D")):
    if (LAB / "out" / name).exists():
        for r in json.loads((LAB / "out" / name).read_text())["runs"]:
            par.setdefault(r["arm"] + tag, {})[r["prompt_id"]] = r
ORDRE = [a for a in ("ctl", "late", "adapt", "floor", "lam2", "fadapt", "floor2", "lam0",
                     "stat0", "multi", "vae", "idx", "R1", "thr05", "rise",
                     "ctl_C", "floor2_C", "lam0_C", "ctl_D", "floor2_D", "lam0_D") if a in par]


def softmax(lg):
    w = np.exp(lg - lg.max())
    return w / w.sum()


def affectations_peigne(w):
    """[(proba, idx)]: the distinct assignments of the systematic comb as u sweeps [0, 1/k),
    with the measure of u that produces each. cumw[-1] forced to 1 as in the code."""
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
    """etats: {tuple of roots per slot: proba}. An assignment idx copies slot idx[j] into j."""
    nouveaux = {}
    for s, p in etats.items():
        for q, idx in affs:
            if q < 1e-12:
                continue
            t = tuple(s[i] for i in idx)
            nouveaux[t] = nouveaux.get(t, 0.0) + p * q
    return nouveaux


def pas_du_run(r, seuil=None):
    """The (normalised) weights at each non-terminal scheduled step, and whether the run
    resampled there. logW restarts from zero after a resampling, as in fk_steer."""
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
            affs = affectations_multinomial(w)   # at every step, flat weights included
        n_res += 1
        etats = propager(etats, affs)
    e_dist = sum(p * len(set(s)) for s, p in etats.items())
    p_un = sum(p for s, p in etats.items() if len(set(s)) == 1)
    return e_dist, p_un, n_res


print(__doc__.splitlines()[0], "\n")

# --- 0. checks ------------------------------------------------------------------------
from commun import fk_avec_r
c = par["ctl"]
memes = [r["prompt_id"] for r in fk_avec_r() if r["prompt_id"] in c]   # the 20 prompts of b_comb.py
premier = np.mean([predire(c[p], "systematic", premier_pas_seul=True)[0] for p in memes])
print(f"0. checks: first step alone of ctl, the {len(memes)} prompts of fk4_stat = {premier:.3f}, expected 1.835 (b_comb.py)")
if "lam0" in par:
    print(f"   lam0: E[roots] = {np.mean([predire(r, 'systematic')[0] for r in par['lam0'].values()]):.3f}, expected 4.000")
    print(f"   lam0 under multinomial at every step: {np.mean([predire(r, 'multinomial')[0] for r in par['lam0'].values()]):.3f}"
          f" (neutral drift, 4 steps with flat weights; a single step would give {4 * (1 - (1 - 1 / 4) ** 4):.3f})")

# --- 1. predicted against observed, per arm ---------------------------------------------
print(f"\n1. distinct roots at the end: observed against predicted from the weights")
print(f"   {'arm':7s} {'n':>3s} | {'obs':>5s} {'sys':>5s} {'mult':>5s} | {'P(1) obs':>8s} {'P(1) sys':>8s} | {'resamp obs':>10s} {'|err| run':>9s} {'corr':>5s}")
resume = {}
for a in ORDRE:
    d = par[a]; ps = sorted(d)
    obs = np.array([d[p]["n_lineages"] for p in ps], float)
    sys_ = np.array([predire(d[p], "systematic") for p in ps])
    mult = np.array([predire(d[p], "multinomial")[0] for p in ps])
    # the column that applies to the arm is that of its resampler (multinomial for multi and R1)
    multinomial = d[ps[0]].get("resampler", "systematic") == "multinomial"
    pred = mult if multinomial else sys_[:, 0]
    corr = np.corrcoef(pred, obs)[0, 1] if obs.std() > 0 and pred.std() > 0 else float("nan")
    resume[a] = (obs.mean(), sys_[:, 0].mean(), mult.mean())
    print(f"   {a:7s} {len(ps):3d} | {obs.mean():5.2f} {sys_[:, 0].mean():5.2f}{'*' if not multinomial else ' '}{mult.mean():5.2f}{'*' if multinomial else ' '}| "
          f"{(obs == 1).mean():8.0%} {sys_[:, 1].mean():8.0%} | {np.mean([d[p]['n_resamplings'] for p in ps]):10.2f} "
          f"{np.abs(pred - obs).mean():9.2f} {corr:5.2f}")
print("   sys = comb at the steps where the run resampled; mult = multinomial at every scheduled step;")
print("   * = the column of the resampler the arm actually used (|err| and corr are read on it).")
print("   |err| run = mean absolute gap between predicted E[roots] and the observed value, run by run (the observed value is one draw).")
steered = [a for a in ORDRE if not a.startswith("lam0") and len(par[a]) >= 17]   # a session still running is left out
gaps = {a: abs(resume[a][0] - (resume[a][2] if par[a][next(iter(par[a]))].get("resampler") == "multinomial" else resume[a][1]))
        for a in steered}
print(f"   over the {len(steered)} steered arms (lam0, which never resamples, is exact by construction): "
      f"max |mean observed - mean predicted| {max(gaps.values()):.2f}, {sum(v <= 0.05 for v in gaps.values())} within 0.05")
print("   observed roots, standard error over prompts: " + ", ".join(
    f"{a} {np.mean(o):.2f} +/- {np.std(o, ddof=1) / len(o) ** .5:.2f}"
    for a in ORDRE for o in [[par[a][p]["n_lineages"] for p in par[a]]] if len(o) > 1))

# --- 2. the pre-registered prediction for thr05 --------------------------------------------
if "floor" in par:
    f = par["floor"]; ps = sorted(f)
    pred = np.array([predire(f[p], "systematic", seuil=0.5) for p in ps])
    print(f"\n2. thr05 predicted from the weights of floor ({len(ps)} prompts), rule ESS < k/2:"
          f" E[roots] {pred[:, 0].mean():.2f}, P(1 root) {pred[:, 1].mean():.0%}, resamplings {pred[:, 2].mean():.2f}")
    if "thr05" in par:
        t = par["thr05"]; obs = np.array([t[p]["n_lineages"] for p in sorted(t)], float)
        print(f"   thr05 observed ({len(obs)} prompts): {obs.mean():.2f} +/- {obs.std(ddof=1) / len(obs) ** .5:.2f} roots "
              f"({(obs.mean() - pred[:, 0].mean()) / (obs.std(ddof=1) / len(obs) ** .5):.1f} standard errors from the forecast), P(1) {(obs == 1).mean():.0%}, "
              f"resamplings {np.mean([t[p]['n_resamplings'] for p in t]):.2f}")

# --- 3. the gap between the two resamplers, on the same weights ------------------------------
if "ctl" in par:
    o, s, m = resume["ctl"]
    print(f"\n3. ctl: the authors' multinomial, on the same weights, would leave {m:.2f} roots against {s:.2f}"
          f" for the comb (observed {o:.2f}): the reference repository collapses at least as much, floor aside.")
