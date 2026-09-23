"""G. Le bras `floor` reproduit-il vraiment le code publie ?

L'affirmation a verifier : envelopper la reward dans max(r, 0) et la passer a
fk_steer donne, pas par pas, les MEMES poids normalises que
`FKD.resample` de fkd_class.py avec `potential_type="max"` et
`reward_min_value=0.0`. Si c'est faux, le contrefactuel GPU ne teste pas ce que
je dis qu'il teste.

Les deux implementations sont pilotees par les memes rewards tirees au hasard et
les memes indices de reechantillonnage, pour que seule la formation des poids
differe. `potentials()` de smc/fk.py est appele tel quel, en lecture.
"""
import torch
from smc.fk import potentials
from smc.weights import normalize_logw

torch.manual_seed(0)
K, T, LAM = 4, 6, 10.0


def auteurs(rs, indices):
    """Transcription de fkd_class.py, branche MAX, reward_min_value = 0."""
    population_rs = torch.zeros(K)
    sortie = []
    for t in range(T):
        cand = torch.maximum(rs[t], population_rs)
        w = torch.exp(LAM * cand)
        sortie.append(w / w.sum())
        idx = indices[t]
        population_rs = cand[idx]
    return sortie


def depot(rs, indices):
    """fk_steer, forme statistique, avec la reward planchee a 0 (ce que fait le bras floor)."""
    gate = torch.full((K,), -float("inf"))
    acc = torch.zeros(K)
    logW = torch.zeros(K)
    sortie = []
    for t in range(T):
        r = torch.clamp(rs[t], min=0.0)        # <- le plancher, par la reward
        logG, gate = potentials(r, gate, LAM, 0.0, "max", False, acc, "terminal", "statistic")
        logW = logW + logG
        w, _ = normalize_logw(logW)
        sortie.append(w)
        acc = acc + logG
        idx = indices[t]
        gate, acc, logW = gate[idx], acc[idx], torch.zeros(K)
    return sortie


print(__doc__.splitlines()[0], "\n")
pire = 0.0
for essai in range(200):
    # des rewards qui commencent negatives et montent, comme ImageReward le long du debruitage
    rs = torch.stack([torch.randn(K) * 0.4 + niveau
                      for niveau in torch.linspace(-1.9, 0.9, T)])
    indices = [torch.randint(0, K, (K,)) for _ in range(T)]
    a, d = auteurs(rs, indices), depot(rs, indices)
    pire = max(pire, max(float((x - y).abs().max()) for x, y in zip(a, d)))
print(f"  200 trajectoires x {T} pas, k = {K}, lambda = {LAM}")
print(f"  ecart maximum sur les poids normalises : {pire:.2e}")
print("  " + ("-> le bras `floor` est bien le potentiel max des auteurs, plancher compris"
             if pire < 1e-5 else "-> LES DEUX NE COINCIDENT PAS, le contrefactuel ne teste pas ce qu'il pretend"))

# et le meme test SANS plancher : l'ecart doit alors etre enorme, sinon le plancher
# ne serait pas la variable qu'on manipule.
def depot_sans(rs, indices):
    gate = torch.full((K,), -float("inf"))
    acc, logW, sortie = torch.zeros(K), torch.zeros(K), []
    for t in range(T):
        logG, gate = potentials(rs[t], gate, LAM, 0.0, "max", False, acc, "terminal", "statistic")
        logW = logW + logG
        w, _ = normalize_logw(logW)
        sortie.append(w)
        acc = acc + logG
        idx = indices[t]
        gate, acc, logW = gate[idx], acc[idx], torch.zeros(K)
    return sortie

torch.manual_seed(0)
ecarts, ess_a, ess_s = [], [], []
for essai in range(200):
    rs = torch.stack([torch.randn(K) * 0.4 + niveau for niveau in torch.linspace(-1.9, 0.9, T)])
    indices = [torch.randint(0, K, (K,)) for _ in range(T)]
    a, s = auteurs(rs, indices), depot_sans(rs, indices)
    ecarts.append(max(float((x - y).abs().max()) for x, y in zip(a, s)))
    ess_a.append(1 / a[0].pow(2).sum().item())
    ess_s.append(1 / s[0].pow(2).sum().item())
print(f"\n  sans le plancher, meme tirage : ecart maximum {max(ecarts):.3f} sur les poids")
print(f"  ESS au premier pas : {sum(ess_a)/len(ess_a):.2f} avec plancher, "
      f"{sum(ess_s)/len(ess_s):.2f} sans")

# --- la reserve : le bras GPU tourne en forme `increment`, pas `statistic` ---
# Manipuler une seule variable (le plancher) impose de garder la forme du bras ctl.
# Les deux formes coincident exactement quand le `gate` est partage, donc au premier
# pas (gate vide) et apres tout reechantillonnage qui ne laisse qu'un ancetre.
def depot_increment(rs, indices):
    gate = torch.full((K,), -float("inf"))
    acc, logW, sortie = torch.zeros(K), torch.zeros(K), []
    for t in range(T):
        r = torch.clamp(rs[t], min=0.0)
        logG, gate = potentials(r, gate, LAM, 0.0, "max", False, acc, "terminal", "increment")
        logW = logW + logG
        w, _ = normalize_logw(logW)
        sortie.append(w)
        acc = acc + logG
        idx = indices[t]
        gate, acc, logW = gate[idx], acc[idx], torch.zeros(K)
    return sortie


torch.manual_seed(0)
par_pas = [[] for _ in range(T)]
collapse = [[] for _ in range(T)]
for essai in range(200):
    rs = torch.stack([torch.randn(K) * 0.4 + niveau for niveau in torch.linspace(-1.9, 0.9, T)])
    indices = [torch.randint(0, K, (K,)) for _ in range(T)]
    a, inc = auteurs(rs, indices), depot_increment(rs, indices)
    for t in range(T):
        par_pas[t].append(float((a[t] - inc[t]).abs().max()))
        collapse[t].append(int(len(set(indices[t - 1].tolist())) == 1) if t else 1)
print("\n  forme `increment` + plancher contre le code publie (forme statistique) :")
for t in range(T):
    print(f"    pas {t} : ecart max sur les poids {max(par_pas[t]):.2e}, "
          f"median {sorted(par_pas[t])[len(par_pas[t])//2]:.2e}"
          + ("   <- identiques par construction (gate vide)" if t == 0 else ""))
print("  Lecture : au pas qui decide de la lignee les deux formes sont le meme calcul.")
print("  Plus loin elles divergent des que deux ancetres survivent, ce qui est justement")
print("  ce que le plancher rend possible : la reserve est reelle et va dans le sens")
print("  d'un bras `floor` legerement different du code publie APRES le premier pas.")
