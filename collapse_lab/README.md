# `collapse_lab/` — pourquoi les quatre images finales descendent d'un seul x_T

Dossier d'enquete, separe du depot : rien ici n'est importe par `smc/`, par
`scripts/` ni par `tests/`, et rien ici n'a modifie `smc/`. Les scripts lisent
`results/*.json` et appellent `smc.fk` / `smc.weights` en lecture.

Les conclusions sont dans **`FINDINGS.md`**. Ce fichier dit seulement comment
rejouer.

## Sans GPU, quelques secondes chacun

    python collapse_lab/a_step0_ess.py      # l'ESS loggee est-elle celle que lam*r implique
    python collapse_lab/b_comb.py           # combien d'ancetres le peigne laisse vivre
    python collapse_lab/c_information.py    # le classement a t=80 predit-il le classement final
    python collapse_lab/d_forms.py          # max et difference coincident-ils au premier pas
    python collapse_lab/e_variants.py       # ce que les 18 variantes sur disque disent
    python collapse_lab/g_equivalence.py    # le bras `floor` est-il le potentiel des auteurs
    python collapse_lab/h_invariances.py    # ce qui peut bouger l'ESS, et ce qui ne le peut pas
    python collapse_lab/i_root100.py        # la racine gardee, sur 100 prompts
    python collapse_lab/j_decomposition.py  # d'ou vient le gain de fk4 sur bon4

`commun.py` porte la regle anti-double-comptage : `fk4_stat.json` et
`fk4_diff.json` sont les memes 20 prompts aux memes x_T, et leur premier pas est
identique a zero pres (constat 4). Les empiler diviserait les erreurs-types par
racine de 2 sans ajouter une observation.

## Avec GPU

    /home/onyxia/work/.venvs/sd/bin/python collapse_lab/probe.py --arms ctl floor --limit 40
    /home/onyxia/work/.venvs/sd/bin/python collapse_lab/f_probe.py
    /home/onyxia/work/.venvs/sd/bin/python collapse_lab/k_figure.py

`nuit.sh` enchaine les bras dans l'ordre : `lam0` court d'abord, parce qu'il decide si
l'appariement avec `bon4` tient, puis les bras qui portent la correction. Il reprend
depuis `out/probe.json`, les paires (prompt, bras) deja faites sont sautees.

`probe.py` fait tourner `fk_steer` sur les memes prompts, les memes seeds et donc
les memes x_T que `scripts/run_sd_baseline.py`, en gardant en plus **la matrice
des ancetres**, qu'aucun fichier de `results/` ne conserve. Il verifie
l'appariement avant de depenser la moindre seconde de GPU : le fichier de prompts
doit etre `data/imagereward-benchmark-prompts.json`, dans cet ordre, sinon
`seed_effective` ne correspond pas a `sd_baseline.json` et rien n'est comparable
(`prompts_subset_40.json` a un autre ordre, et c'est le piege).

Les bras :

| bras | lambda | plancher | ce qu'il isole |
|---|---|---|---|
| `ctl` | 10 | non | le `fk4` de reference, controle du pilote |
| `floor` | 10 | oui | le `reward_min_value = 0.0` du code publie, seul |
| `lam2` | 2 | non | la seule force du tilt |
| `floor2` | 2 | oui | les deux |
| `lam0` | 0 | non | controle d'appariement : doit redonner `bon4` case par case |
| `adapt` | 10 max | non | lambda bisecte a ESS = k/2 aux pas non terminaux |
| `fadapt` | 10 max | oui | le plancher et le lambda adaptatif ensemble |

Pour `adapt` et `fadapt`, 10 est le plafond et le lambda du pas terminal :
`bisect_lambda` rend le plafond des que l'ESS y est deja au-dessus de la cible, donc
lambda_t ne peut que descendre. Ces deux bras testent « moins fort tot », pas le profil
montant du constat 8 ; « plus fort tard » demanderait `lam_max = 100` et serait un
autre bras.

Le plancher passe par la **reward** (`max(r, 0)` avant `fk_steer`), pas par
`smc/fk.py` : `g_equivalence.py` verifie a 9e-7 que c'est bien le potentiel `max`
des auteurs, plancher compris.

## Ce que ce dossier ne fait pas

Il ne corrige rien dans `smc/`. `FINDINGS.md` section 9 dit ou les corrections
vivraient et quel compromis chacune porte ; le code est a ecrire par l'auteur.
