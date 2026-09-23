# Pourquoi les quatre images finales descendent d'un seul x_T

## En bref

**Le probleme.** FK avec k = 4 rend quatre images issues du meme x_T dans 96 runs sur
100 ; diversite pixel 0.09 contre 0.33 pour quatre tirages libres.

**La cause.** Deux degenerescences distinctes, que l'ESS ne separe pas (constat 15).
*Les poids* : a chaque pas planifie le poids vaut `exp(lambda * r_phi)` a un facteur
partage pres, et a lambda = 10 le premier pas (t = 80) a deja une etendue de 9 nats
sur une reward guide toute negative et sans information -- ESS 1.24 sur 4. Le code des
auteurs planche la statistique `max` a 0 et rend ce pas inerte ; `smc/fk.py` ne le fait
pas, seul ecart a la reference (constat 6). *Les chemins* : reechantillonner quatre
particules quatre fois fait coalescer l'arbre des ancetres vers une racine meme a poids
moderes -- `adapt` tient l'ESS a 2.0 et finit quand meme a 1.40 lignees.

**Le cout.** Rien de mesurable sur `ir_max` (+0.056 +/- 0.052 contre best-of-4, n = 100),
mais quatre clones ont l'`ir` d'une seule image : l'`ir` moyen perd 0.53 +/- 0.12 contre
quatre tirages libres (constat 13).

**Les pistes** (constats 14, 17, 18, 20). Toutes ramenent `ir_max` au niveau de
best-of-4, a une ou deux erreurs-types du bruit ; elles se distinguent par la diversite
gardee et par l'`ir` moyen des quatre (-0.15 a -0.56). Mesure : plancher + lambda = 2
(3.0 lignees, -0.01 sur `ir_max` apparie par x_T a n = 100) > plancher + lambda bisecte
(2.12) > seuil ESS < k/2 ~ plancher seul ~ lambda = 2 seul (1.7-2.0) > lambda bisecte
seul (1.40) ; calendrier sans t = 80 : 1.14. Le nombre de racines finales se predit depuis
les poids et le resampler seuls, a 0.07 pres sur quinze bras (constat 17). A lambda = 10 la
cible elle-meme ne porte que 1.2 a 1.5 particules sur 4 (constat 15) : rien ne « resout »
l'effondrement a ce lambda, et la seule correction qui passe sous 25 % de runs a une racine
change la cible. Mort : centrer la reward, changer de potentiel, reechantillonner moins.

**La reference** (constat 16). La configuration du papier est celle de ce depot ; le code
publie rend en moyenne moins que best-of-4 (-0.11 +/- 0.04 sur quatre runs) avec une
dispersion d'un run a l'autre plus large que l'effet du papier. Ecart borne, pas ferme.

---

## Le probleme, et la cause la plus probable

**Le probleme.** Avec k = 4, les quatre images que FK rend a la fin descendent d'une
seule racine x_T dans 96 runs sur 100. La diversite pixel tombe a 0.09 contre 0.33 pour
quatre tirages libres. Ce n'est plus un echantillonneur a quatre particules : c'est une
recherche gloutonne sequentielle sur une racine choisie tot.

**La cause la plus probable.** Le premier pas planifie (t = 80) decide de tout, et c'est
le pas ou la reward guide ne sait rien. Le poids y vaut `exp(lambda * r_phi)` a un
facteur partage pres -- un classement sur le **niveau** de la reward, pas sur son
increment, parce que le terme soustrait par le potentiel est commun aux particules et
disparait a la normalisation. A lambda = 10, une etendue de 0.90 devient 9 nats, l'ESS
tombe a 1.24 sur 4, et un seul passage du peigne ne laisse que 1.8 ancetres. Or le
classement a ce pas ne predit pas la qualite finale (Kendall tau = +0.067 +/- 0.071).
Quatre pas comme celui-la et il ne reste qu'une lignee.

**Ce qui l'aggrave, et qui est un ecart au code de reference.** Les auteurs initialisent
la statistique du potentiel `max` a `reward_min_value = 0.0` ; ici elle part de moins
l'infini. ImageReward etant negative pour les quatre particules dans 90 % des runs a
t = 80, leur pas est **inerte** la ou le notre depense 9 nats sur du bruit. C'est le seul
endroit ou ce depot est plus agressif que sa reference, et il tombe exactement sur le pas
qui tue la lignee. Le contrefactuel (constat 10) le confirme : avec le plancher, 3.9
ancetres survivent a t = 80 au lieu de 1.8.

**Ce que la correction ne promet pas.** Garder des lignees coute environ 0.10
d'`ir_max` contre `ctl`, et ce prix est **plat** : il est le meme pour le bras qui garde
0.35 lignee de plus et pour celui qui en garde 2.95 (constat 13). Sortir de
l'effondrement ramene `ir_max` au niveau de best-of-4, ensuite la diversite ne coute plus
rien sur `ir_max` ; c'est l'`ir` **moyen** des quatre images qui paie, et lui suit ce que
l'on achete. Le meilleur compromis mesure est le plancher **avec lambda = 2** (`floor2`) :
2.85 lignees au lieu de 1.05, diversite pixel 0.263 contre 0.092, pour le meme -0.10 que
les autres bras (n = 20). Le lambda adaptatif seul ne repare presque rien : une cible
d'ESS a k/2 fait reechantillonner a tous les pas et divise les lignees plus lentement au
lieu de les garder. Les pistes, mesurees et non mesurees, sont au constat 14.

---

Etat au 22/09 apres la nuit complete. Les constats 1 a 6 et 8 sont fermes et rejouables
sans GPU depuis `collapse_lab/*.py`. Le constat 7 porte les predictions ecrites avant les
donnees, le constat 10 ce que la sonde en dit sur 10 prompts, le constat 11 un controle
qui echoue et qui met en doute les constats 3, 5 et 5 bis, le constat 12 les deux bras de
correction sur 20 prompts, le constat 13 les sept bras au n final (40 ou 20 prompts) et
les trois lectures du constat 12 qu'il renverse, le constat 14 les facons de reparer, le
constat 15 une relecture independante : poids et genealogie sont deux degenerescences.

## La chaine causale, en une phrase

A chaque pas planifie, le poids d'une particule est `exp(lambda * r_phi(x_t))` a
un facteur **partage** pres, donc un classement sur le **niveau** de la reward
guide ; a `lambda = 10` l'etendue de ce niveau vaut 9 nats au premier pas, l'ESS
tombe a 1.3 sur 4, et le peigne systematique donne 3 ou 4 cases sur 4 a une seule
particule. Quatre rebelotes et il ne reste qu'une racine. Le pas qui decide est
celui ou la reward guide ne predit rien.

## 1. Ce n'est pas un bug de `weights.py` ni de `resampling.py`

L'ESS recalculee depuis `r_at_schedule[0]` colle a `ess_at_schedule[0]` a
**4.3e-05** en median et 4.8e-04 au pire, sur les 20 runs qui portent le champ,
et le meme chiffre sur `fk4_diff`, qui est le meme jeu de prompts
(`collapse_lab/a_step0_ess.py`). L'effondrement est de l'arithmetique exacte.

## 2. Un seul reechantillonnage suffit a tuer la moitie des lignees

Le peigne de `smc/resampling.py` donne a la particule j un nombre de cases egal a
`floor` ou `ceil` de `k * w_j`. En integrant sur `u ~ U(0, 1/k)` a partir des
poids du pas t = 80 : **1.835 ancetres distincts attendus**, et P(une seule
lignee des ce pas) = 0.46 (`collapse_lab/b_comb.py`).

Controle independant : `S80.json`, dont le calendrier `[0, 80]` ne contient qu'un
seul reechantillonnage, observe **1.850** lignees. Prediction et mesure coincident
sur un fichier qui n'a pas servi a la calculer.

## 2 bis. Le chiffre de reference, sur 100 prompts

`sd_ref_fields100.json`, la configuration exacte de la ligne FK :

| pas | ESS mediane (sur k=4) | Q1 - Q3 | part des runs sous 1.5 |
|---|---|---|---|
| t = 80 | **1.24** | 1.01 - 2.00 | **58 %** |
| t = 60 | 1.57 | 1.10 - 2.25 | 46 % |
| t = 40 | 1.97 | 1.19 - 3.16 | 29 % |
| t = 20 | 2.94 | 1.82 - 3.73 | 17 % |
| t = 0 | 2.61 | 1.84 - 3.22 | 13 % |

`n_lineages` vaut 1 dans **96 runs sur 100** et 2 dans les 4 autres ;
`n_resamplings` moyen 3.78 sur 4 possibles. Une ESS de 1.24 sur 4 particules veut
dire qu'une particule porte a elle seule pres de 90 % du poids.

## 3. La pression de selection est maximale la ou le signal est nul

| pas | niveau moyen de r_phi | etendue mediane | nats a lambda=10 | ESS mediane |
|---|---|---|---|---|
| t = 80 | -1.62 | 0.90 | 9.0 | 1.34 |
| t = 60 | -0.33 | 0.77 | 7.7 | 1.48 |
| t = 40 | +0.32 | 0.52 | 5.2 | 1.92 |
| t = 20 | +0.68 | 0.32 | 3.2 | 2.58 |
| t = 0  | +0.84 | 0.27 | 2.7 | 2.39 |

Et l'information va dans l'autre sens. La case j de `fk4` et la case j de `bon4`
partagent x_T, et la ligne 0 est lue avant tout reechantillonnage :
`bon4["ir"][j]` est donc ce que cette racine devient si on la laisse tranquille.
**Cette phrase est l'hypothese que le constat 11 met en echec** : le bras `lam0` ne
reproduit pas `bon4` case par case. Ce qui suit, et le constat 5 bis, en dependent.

- Kendall tau entre le classement a t = 80 et le classement final : **+0.067 +/- 0.071**
- la meilleure racine a t = 80 est la meilleure finale dans **3 runs sur 20** (hasard 25 %)
- `ir(meilleure racine) - ir(racine choisie)` = **+0.627**, contre **+0.568** pour une
  racine tiree au sort : difference appariee **+0.059 +/- 0.132**, donc
  **indiscernable du hasard**. Le signe est defavorable, l'erreur-type le couvre.

n = 20 et non 40 : `fk4_stat` et `fk4_diff` portent les memes prompts aux memes
x_T, et le constat 4 montre que leur ligne 0 est identique a zero pres. Les
empiler diviserait l'erreur-type par racine de 2 sans ajouter une observation
(`collapse_lab/commun.py`).

## 4. La soustraction du potentiel ne protege de rien

`_potential_terms` soustrait `prev` (le `gate`, le max courant, la reward
precedente selon le potentiel). Mais `prev` est **partage entre les cases** des
que celles-ci descendent d'un meme ancetre, et il vaut 0 au premier pas par
construction (`gate` vide pour `max`, zero pour `difference` et `sum`). Un terme
partage disparait a la normalisation : le poids est `exp(lambda * r_t)`.

Mesure directe sur un run de la sonde, t = 60 : `logG - 10 * r_phi` vaut
3.5707 pour les quatre cases, exactement constant. Le "cliquet" ne se met a
aplatir qu'a t = 40 et t = 20, quand le record devient dur a battre -- bien apres
que la lignee soit morte.

Consequence verifiable, et verifiee : **le choix du potentiel ne peut rien
changer**. `fk4_stat` (max, forme statistique) et `fk4_diff` (difference)
partagent prompts et seeds ; leurs `r_phi` sont identiques **a zero pres** a
t = 80 et t = 60, ils gardent la meme racine dans 19 prompts sur 20, et l'ecart
appariee sur `ir_max` est de **-0.0043 +/- 0.0240** (`collapse_lab/d_forms.py`).
C'est la reponse a la question laissee ouverte par `docs/max_potential.md` :
`difference` n'a pas ferme l'ecart parce que le pas qui decide de la lignee ne
regarde pas le potentiel.

## 5. Les 18 variantes deja sur disque disent la meme chose

`collapse_lab/e_variants.py` : `div_pix` reste au plancher de 0.09 (contre 0.33
pour `bon4`) pour le calendrier dense, les rampes, les seuils, les deux formes et
les trois potentiels. Les seules variantes qui remontent la diversite sont celles
qui **affaiblissent le premier pas** (`T2`, lambda_1 = 0.4 : 1.55 lignees,
div_pix 0.139) ou qui n'en ont qu'un (`S80` : 1.85 lignees, div_pix 0.233).

## 5 bis. Ce n'est pas best-of-n : c'est pire que best-of-n sur la racine

Avec une ESS de 1.3 a chaque pas planifie, la methode degeneree est un **glouton
sequentiel** : on tire 4 racines, on en garde une tot, on la clone, les clones
divergent sous le bruit de DDIM a eta = 1, on en garde un au pas suivant, et ainsi
de suite. Les quatre images finales sont quatre freres separes depuis t = 20
environ.

La difference avec best-of-4 se chiffre. Sur les 100 prompts de
`sd_ref_fields100.json`, apparies case par case avec `bon4`
(`collapse_lab/j_decomposition.py`) :

| | ImageReward |
|---|---|
| A meilleure des 4 racines, menee librement | +0.7698 |
| M racine moyenne, libre | +0.2232 |
| B **la racine que fk garde**, menee librement | +0.3592 |
| C **ce que fk en tire reellement** | +0.8257 |

- **B - M = +0.136 +/- 0.057** : la racine gardee vaut mieux qu'un tirage au sort.
  Elle recupere environ **un quart** de l'ecart disponible (IC95 bootstrap sur les
  prompts [+0.023, +0.246], rang moyen 1.220 sur 4 contre 1.500 au hasard, IC95
  [1.020, 1.430], `collapse_lab/i_root100.py`).
- **B - A = -0.411 +/- 0.060** : l'effondrement coute 0.41 de qualite de racine par
  rapport a ce que best-of-4 choisit, qui prend la meilleure par construction.
- **C - B = +0.467 +/- 0.073** : le pilotage le long de la trajectoire, a racine
  fixee, remonte plus que cela.
- **C - A = +0.056 +/- 0.052**, et la decomposition ferme exactement.

Donc : `fk4` **part plus mal que best-of-4** et le rattrape en pilotant. Le gain
publie n'est pas "FK choisit une meilleure racine", c'est "FK choisit moins bien
et pilote mieux". L'effondrement est le **prix** du pilotage, pas son moyen.

**Correction a une lecture anterieure.** `docs/max_potential.md` constat 4 conclut
que "la racine gardee n'est pas meilleure que le hasard", a partir d'un taux
binaire (l'argmax de bon4 survit-il) mesure sur `S60`. Le test binaire est peu
puissant et `S60` n'a pas de pas a t = 80. Sur le fichier de reference a 100
prompts, le rang et le cout en reward rejettent le hasard a 2,5 erreurs-types. Ce
qui reste vrai, c'est que **le seul pas t = 80** n'apporte rien (constat 3) : ce
que `root_slots` mesure, c'est la composition de tous les reechantillonnages, et
dans les 54 % de runs ou deux racines survivent a t = 80, c'est t = 60 qui
tranche, avec une reward guide deja moins bruitee.

## 6. Le code publie des auteurs a un garde-fou que `smc/fk.py` n'a pas

`fkd_class.py` de <https://github.com/zacharyhorvitz/Fk-Diffusion-Steering> :

```python
self.population_rs = torch.ones(self.num_particles, device=...) * reward_min_value  # 0.0
...
if self.potential_type == PotentialType.MAX:
    rs_candidates = torch.max(rs_candidates, self.population_rs)
    w = torch.exp(self.lmbda * rs_candidates)
```

La statistique du potentiel `max` est **plancheee a 0**. `smc/fk.py` applique ce
plancher a `prev` (`torch.where(torch.isneginf(gate), zeros, gate)`) mais **pas a
`curr`** (`curr = torch.maximum(gate, r_t)` avec `gate` a -inf). Or ImageReward
est negative pour toutes les particules dans **90 %** des runs a t = 80 et
**40 %** a t = 60. Chez les auteurs ces pas donnent `max(r, 0) = 0` partout, donc
des poids uniformes et **aucune selection** ; ici ils donnent 9 nats d'ecart sur
du bruit.

Le plancher ne protege que `max`. Pour `diff`, `population_rs` vaut 0 au premier
pas et le poids est `exp(lambda * (r_t - 0))` : un classement sur le niveau, lui
aussi. La configuration d'evaluation publiee -- `lmbda=10`,
`resample_frequency=5`, `resample_t_start=5`, `resample_t_end=30`,
`potential_type="diff"`, `adaptive_resampling` **desactive** par defaut -- place
son premier reechantillonnage a l'indice de boucle 5 sur 100, soit t d'environ
950, encore plus bruite que le t = 80 d'ici. **L'effondrement n'est donc pas
propre a ce depot : il est dans le regime du code publie aussi.** Leur
configuration qualitative, elle, est `lmbda=2.0`, `adaptive_resampling=True`,
`resample_frequency=20`, `t_start=20`, `t_end=80`.

Les autres ecarts vont tous dans le sens d'un depot plus **doux** que la
reference, pas plus dur : le peigne systematique de `smc/resampling.py` a une
variance plus faible que le multinomial des auteurs, et a seuil 1.0
`should_resample` teste `ESS < k` au sens strict, donc ne reechantillonne pas a
poids exactement uniformes la ou leur boucle non adaptative tire quand meme. Le
plancher manquant est le seul endroit ou ce depot est plus agressif que la
reference.

## 7. Contrefactuel GPU : predictions ecrites avant les donnees

Cinq bras, memes prompts, memes x_T, `collapse_lab/probe.py` : `ctl` (le fk4 de
reference), `floor` (le plancher des auteurs, obtenu en enveloppant la reward
passee a `fk_steer`, sans toucher a `smc/`), `lam2`, `floor2`, `lam0` (controle
d'appariement). `ctl` et `floor` tournent sur 40 prompts, les autres sur 20.
Depouillement : `collapse_lab/f_probe.py`.

Le peigne, applique au premier pas aux poids reellement enregistres
(`collapse_lab/b_comb.py`), donne le nombre d'ancetres attendus juste apres
t = 80 :

| regime | ESS mediane | ancetres apres t=80 | pas inerte |
|---|---|---|---|
| lam=10 (`ctl`) | 1.34 | **1.835** | 0 % |
| lam=10 + plancher (`floor`) | 4.00 | **3.849** | **90 %** |
| lam=2 (`lam2`) | 2.72 | **2.898** | 0 % |
| lam=2 + plancher (`floor2`) | 4.00 | **3.970** | 90 % |

**Predictions, ecrites le 21/09 avant le depouillement.**

1. `ctl` : 1.8 lignees apres t = 80, puis decroissance jusqu'a 1.05 a la fin ;
   reproduit le `ir_max` du `fk4` de `sd_baseline.json` a l'erreur d'arrondi pres.
2. `floor` : t = 80 inerte dans environ 90 % des runs, donc environ 3.8 lignees
   apres ce pas. Le plancher **ne supprime pas** l'effondrement, il le **retarde**
   jusqu'au premier pas ou une particule passe au-dessus de 0 : le run d'essai
   montre un prompt ou une seule des quatre est positive a t = 80, logG =
   [3.57, 0, 0, 0], ESS 1.17, effondrement immediat. Prediction finale : entre 2 et 3
   lignees, nettement au-dessus de 1, nettement en dessous de 4.
3. `lam2` : environ 2.9 lignees apres t = 80, environ 1.3 a 1.6 a la fin.
4. `floor2` : le plus protecteur, environ 4.0 apres t = 80.
5. `lam0` : 4 lignees partout, et les quatre `ir` egaux case par case a ceux de
   `bon4` (derive fp16 attendue, pas egalite au bit).
6. **`ir_max` ne bouge pas.** Aucun bras ne doit s'ecarter de `ctl` de plus d'une
   erreur-type (environ 0.05 a 0.07 ici). C'est la prediction la plus exposee, et
   elle suit du constat 3 : si la racine est choisie au hasard, en perdre trois ne
   coute rien en moyenne. L'effondrement et l'ecart de 0.078 sur la reward sont
   deux questions distinctes, et ce fichier ne traite que la premiere.


**La nuit du 21 au 22 a ete tuee avec le terminal apres 20 runs sur 80** ; la sonde a ete
relancee detachee le 22 au matin, avec `lam0` en premier. Le depouillement partiel est au
constat 10, le controle `lam0` au constat 11.

## 8. Ce qui ne peut pas marcher, et pourquoi (`collapse_lab/h_invariances.py`)

Le poids est `exp(lambda * r)` normalise. **Toute transformation de r qui ajoute
la meme chose a toutes les particules disparait a la normalisation.** Sur le
premier pas d'un run reel, ESS = 1.1761 ; centrer la reward, la decaler de +100,
lui retrancher un max courant partage : **1.1761 a chaque fois, au bit pres**.
Centrer la reward guide est donc une piste morte, et c'est aussi la raison pour
laquelle la soustraction de `prev` dans `_potential_terms` ne protege de rien
(constat 4).

Ce qui bouge l'ESS, c'est ce qui comprime les **ecarts** entre particules :
baisser lambda (2.70 a lambda = 2), ou creer des ex aequo, ce que fait le plancher
(4.0000, le pas devient inerte).

Le lambda qui tiendrait l'ESS a k/2, calcule par `smc.fk.bisect_lambda` sur les
r_phi enregistres :

| pas | lambda pour ESS = 2 (median) | Q1 - Q3 |
|---|---|---|
| t = 80 | **3.91** | 2.55 - 15.49 |
| t = 60 | 4.24 | 3.00 - 7.02 |
| t = 40 | 8.53 | 5.01 - 11.19 |
| t = 20 | 11.93 | 8.15 - 20.07 |
| t = 0 | 15.22 | 9.73 - 62.39 |

Le run tourne a 10.0 partout : **2,5 fois trop fort au pas ou la reward ne predit
rien, et trop faible aux pas ou elle predit**. Le profil correct monte le long du
debruitage, ce qui est l'inverse des rampes des runs 13 a 15, qui baissaient
lambda au debut sans le remonter a la fin.

## 9. Ce que cela implique, sans ecrire le code

Trois choses a faire, de la moins chere a la plus chere, chacune dans un fichier
qui appartient a l'auteur.

**Le plancher.** `_potential_terms`, branche `max` de `smc/fk.py` : le plancher a
0 est applique a `prev` et pas a `curr`. Le code publie initialise sa statistique
a `reward_min_value = 0.0`, pas a moins l'infini, et lit `max(r_t, statistique)`.
C'est le seul endroit ou ce depot est plus agressif que sa reference, et il tombe
exactement sur le pas qui decide de la lignee. Le rendre parametrable plutot que
constant : le bon plancher depend de la reward, 0 pour ImageReward n'a rien
d'universel.

**Le lambda adaptatif.** `fk_steer` a deja `adaptive_lam` et `bisect_lambda`
depuis le 21/09, et `scripts/run_sd_baseline.py` ne les expose pas. La table
ci-dessus dit ce qu'il ferait : environ 4 au premier pas au lieu de 10, environ 15
au dernier. Le compromis est a trancher : il change la cible intermediaire mais
pas `exp(lambda * r(x_0))`, puisque `acc` porte la correction et que le pas
terminal garde lambda.

**Et la question qui reste ouverte.** Reparer l'effondrement pourrait rapporter
plus que ce que `docs/max_potential.md` laissait attendre -- sous reserve du constat 11,
qui met en doute l'appariement dont ce chiffre sort. Le constat 5 bis chiffre
a **0.411 +/- 0.060** ce que l'effondrement coute au niveau de la racine : un
echantillonneur qui garderait quatre lignees distinctes et les piloterait toutes
aurait ce 0.411 a portee, la ou le gain publie n'est que de 0.056. Mais rien ne
dit qu'il le prendrait : le pilotage, lui, profite de la concentration, et le
Spearman de +0.015 entre l'ESS du premier pas et le gain
(`docs/max_potential.md`) dit que les runs les plus effondres ne perdent pas plus
que les autres. Diversite et reward restent deux questions ; ce fichier traite la
premiere et borne la seconde.

Le classement mesure de ces pistes, et celles qui restent a tester : constat 14.

## 10. Le contrefactuel GPU : ce que la sonde a rendu

La nuit du 21 a ete tuee avec le terminal apres 20 runs sur 80 (`ctl` et `floor` sur les
dix premiers prompts, les trois autres bras jamais lances). Relancee detachee le 22.
Sur ces dix prompts, apparies :

| bras | t=80 | t=60 | t=40 | t=20 | t=0 | reech. | div_pix | ir_max |
|---|---|---|---|---|---|---|---|---|
| `ctl` | 1.80 | 1.40 | 1.20 | 1.10 | 1.10 | 3.90 | 0.105 | +1.066 |
| `floor` | 3.90 | 2.80 | 2.10 | 1.80 | 1.80 | 2.30 | 0.167 | +0.895 |
| `lam0` | 4.00 | 4.00 | 4.00 | 4.00 | 4.00 | 0.00 | 0.338 | +0.961 |

Part des pas ou les poids sortent uniformes, donc sans aucune selection : `floor` 90 % a
t = 80, 40 % a t = 60, 30 % a t = 40. `ctl` : 0 % partout sauf 10 % a t = 40.

Confrontation aux predictions du constat 7, ecrites avant :

1. **Tenue.** `ctl` : 1.80 ancetres apres t = 80 contre 1.835 predits par le peigne, et
   1.10 a la fin.
2. **Tenue sur le mecanisme, ratee sur le chiffre final.** `floor` : t = 80 inerte dans
   90 % des runs (predit 90 %), 3.90 ancetres (predit 3.849). Mais l'effondrement reprend
   plus vite qu'annonce : 1.80 a la fin, sous la fourchette 2-3. Le plancher **retarde**
   l'effondrement d'un a deux pas, il ne le supprime pas.
6. **Sous tension.** `floor - ctl` sur `ir_max` vaut **-0.171** de moyenne, la ou la
   prediction voulait moins d'une erreur-type. Mais la mediane est de -0.025 et trois
   prompts portent toute la moyenne, dont deux ou `floor` n'a jamais reechantillonne,
   c'est-a-dire ou il *est* best-of-4. A n = 10 la direction est contraire a la
   prediction et compatible avec le constat 5 bis (le pilotage paye, la diversite non) ;
   rien n'est tranche.

Le controle du pilote 0a de `f_probe.py` comparait `ctl` au `fk4` de `sd_baseline.json`,
ecrit le 20/09, donc **avant** le commit 5025180 : il affichait un faux ecart de +0.090.
Contre `sd_ref_fields100.json`, qui est la ligne FK du code actuel, `ctl` est identique
aux dix prompts, ESS a t = 80 comprise, ecart apparie +0.0000. Le pilote est propre ; le
fichier de reference ne l'etait pas.

## 11. Le controle d'appariement echoue, et il emporte les constats 3, 5 et 5 bis

`lam0` (lambda = 0, aucun reechantillonnage) devait rendre les quatre tirages libres de
`bon4`, case par case, a la derive fp16 pres. Sur dix prompts :

- ecart maximum par prompt sur les quatre `ir` : **0.76** en median, 2.97 au pire ;
- correlation intra-prompt entre `lam0[j]` et `bon4[j]` sur les 40 cases : **-0.107**
  (controle avec les prompts permutes : +0.059) ;
- l'argmax coincide dans 3 prompts sur 10, soit le hasard ;
- mais les marges collent : `ir` moyen 0.416 contre 0.468, `ir_max` moyen 0.961 contre
  0.949, `div_pix` 0.338 contre 0.33.

Donc `lam0` est un echantillonneur libre correct **en loi**, et l'identite de la case ne
survit pas d'un echantillonneur a l'autre. En lisant le code, `initial_state` tire le
meme `randn_tensor((4, 4, 64, 64))` que `prepare_latents` du pipeline et les deux passent
par le meme `scheduler.step(..., generator)` : x_T devrait etre partage au bit pres et
seul l'arrondi fp16 de `eps` differe (CLIP encode en batch 4 contre un encode etendu).
Deux lectures restent ouvertes, et elles ont la meme consequence :

- soit le flux de bruit diverge malgre tout, et `bon4[j]` n'est pas la continuation de la
  racine j ;
- soit x_T est bien partage, et a eta = 1 sur 100 pas la racine n'explique presque rien
  de la variance de la reward finale -- ce que la correlation de -0.107 dit directement.

Dans les deux cas, la phrase « `bon4["ir"][j]` est ce que la racine j devient si on la
laisse tranquille » n'est pas mesurable ainsi. **Le constat 3 (tau = +0.067), le constat 5
et la decomposition A / M / B / C du constat 5 bis reposent sur elle et sont a
reprendre.** Ce qui ne bouge pas : les constats 1, 2, 2 bis, 4, 6 et 8, qui ne lisent
jamais `bon4`.

Le test qui tranche est au niveau des latents, pas de la reward : un prompt, capturer les
latents du pipeline aux pas 0 et 1 par `callback_on_step_end`, et les comparer a
`initial_state` puis un `step`. S'ils coincident, c'est la seconde lecture.

*Ajout du 23/09.* Le test a ete fait (`m_latents.py`) : x_T coincide au bit et les
trajectoires restent correlees a 0.98 ou plus jusqu'au dernier pas ; la troisieme lecture
est la bonne, le chemin avec reechantillonnage n'est pas rejouable d'une session a l'autre
(constat 19). La session C rejoue `ctl` et `lam0` dans un meme processus et recalcule les
constats 3 et 5 bis sur cet appariement (constat 20) : ils tiennent, en plus faible.

## 12. Les deux corrections, evaluees hors de `smc/`

Aucun des deux bras ne demande de modifier `smc/` : le plancher passe par la reward
(`torch.clamp(r, min=0)`, equivalent au potentiel des auteurs a 9e-7, constat 6 et
`g_equivalence.py`), le lambda adaptatif par `adaptive_lam` / `ess_target` / `lam_max`,
que `fk_steer` porte depuis le 21/09.

| bras | lambda | plancher | ce qu'il isole |
|---|---|---|---|
| `adapt` | 10 en plafond | non | lambda bisecte a ESS = k/2 aux pas non terminaux |
| `fadapt` | 10 en plafond | oui | les deux |

Reserve de reglage : `bisect_lambda` rend le plafond des que l'ESS y est deja au-dessus
de la cible, et le pas terminal garde lambda. Avec `lam_max = 10`, lambda_t ne peut donc
que **descendre** : ces deux bras testent « moins fort tot », pas le profil montant que
la table du constat 8 appelle (environ 4 a t = 80, environ 15 a t = 0). « Plus fort
tard » demanderait `lam_max = 100`, et c'est un troisieme bras.

**Predictions, ecrites avant le depouillement.** `adapt` : lambda median autour de 4 a
t = 80, ESS clouee a 2.0 a chaque pas non inerte, donc environ 2.5 a 2.9 ancetres apres
t = 80 -- mais comme l'ESS cible est sous k, il reechantillonne a **tous** les pas, et
les lignees continuent de se diviser par deux : 1.5 a 2 a la fin, au-dessus de `ctl`,
proche de `floor`. `fadapt` : t = 80 inerte dans environ 90 % des runs (`base` plat,
`bisect_lambda` rend son defaut, logG nul), puis bisection au premier pas ou une reward
passe au-dessus de zero au lieu de l'effondrement immediat ; le plus protecteur, au moins
3 ancetres apres t = 60. Sur la reward, les deux doivent rester **sous** `ctl`, pour la
raison du constat 5 bis.

### Ce que les deux bras ont rendu, sur 20 prompts

Apparies contre `ctl`, qui est `sd_ref_fields100.json` case par case (constat 10) :

| bras | n | lignees | div_pix | ir_max | contre `bon4` |
|---|---|---|---|---|---|
| `adapt` | 20 | **+0.20 +/- 0.09** | +0.041 +/- 0.014 | **-0.028 +/- 0.033** | +0.111 +/- 0.094 |
| `fadapt` | 20 | **+1.20 +/- 0.26** | +0.117 +/- 0.032 | -0.103 +/- 0.077 | +0.036 +/- 0.093 |
| `floor` | 11 | +0.91 +/- 0.44 | +0.082 +/- 0.054 | -0.198 +/- 0.079 | -0.015 +/- 0.149 |
| `lam0` | 10 | +2.90 +/- 0.10 | +0.241 +/- 0.025 | -0.105 +/- 0.104 | +0.012 +/- 0.168 |

`ctl` sur ces 20 prompts : 1.05 lignees, div_pix 0.089, ir_max +0.961.

**Le lambda adaptatif seul ne repare presque rien, et c'est la surprise.** Il fait ce qu'on
lui demande : lambda median **3.91** a t = 80, exactement la table du constat 8, et l'ESS
clouee a 2.0 a t = 80 et t = 60. Il finit quand meme a 1.25 lignees, sous la fourchette
1.5-2 predite. La raison est structurelle : une cible d'ESS a k/2 est **sous** le seuil de
reechantillonnage 1.0, donc le bras reechantillonne a tous les pas planifies (3.95 sur 4)
et donne a chaque fois deux cases sur quatre a une seule particule. Tenir l'ESS a k/2 ne
garde pas les lignees, cela les divise plus lentement.

**Ce qui marche, c'est de rendre le pas inerte, pas de l'adoucir.** `floor` et `fadapt`
laissent t = 80 inerte dans 90 % des runs et finissent a 2.0 et 2.25 lignees. Et `fadapt`
domine `floor` seul sur les deux axes a ce n : plus de diversite (+1.20 contre +0.91) pour
la moitie du cout en reward (-0.103 contre -0.198). Lecture : une fois les premiers pas
neutralises par le plancher, la bisection empeche les pas suivants d'effondrer ce qui
reste. Les erreurs-types se chevauchent, 20 prompts contre 11 : c'est une direction, pas
un resultat.

**Le classement en reward confirme encore le constat 5 bis.** Tout bras qui garde des
lignees les paie, et le prix suit ce qu'il achete : `lam0` (aucun pilotage, 4 lignees)
-0.105, `fadapt` -0.103, `floor` -0.198. `adapt` est le seul presque gratuit, et c'est
aussi celui qui n'achete rien. La prediction 6 du constat 7 -- « `ir_max` ne bouge pas » --
est maintenant fausse dans le sens ou **reparer l'effondrement coute de la reward**.

Un fil laisse pendant : avec lambda adaptatif, `logG - lambda_t * r_phi` cesse d'etre
constant entre les cases des t = 60 (0 % des runs, contre 50 % pour `ctl`). L'explication
probable est le cliquet du `max` qui se met a mordre une fois lambda_t assez petit pour
qu'aucune particule ne batte le record -- le potentiel ferait enfin quelque chose -- mais
ce n'est pas verifie et ce n'est pas ce que le bras testait.

**A 40 prompts, trois des lectures ci-dessus ne tiennent plus** : `adapt` n'est pas
gratuit, `fadapt` ne divise pas par deux le cout de `floor`, et le prix ne suit pas ce
qu'il achete sur `ir_max`. Les -0.20 de `floor` a n = 10 et 11 etaient du bruit. Voir le
constat 13 ; le texte ci-dessus est garde tel qu'ecrit a 20 prompts.

## 13. La nuit complete : sept bras, 40 ou 20 prompts

`nuit.sh` est alle au bout (`NUIT TERMINEE` dans `out/nuit.log`) : `ctl`, `floor`,
`adapt`, `fadapt` sur 40 prompts, `lam0`, `lam2`, `floor2` sur 20. Depouillement :
`collapse_lab/f_probe.py`. Controles d'abord :

- `ctl` contre le `fk4` de `sd_ref_fields100.json`, 40 prompts : ecart apparie
  **0.0000** au pire. Le pilote est la ligne FK de reference.
- ESS recalculee depuis `logG` contre ESS enregistree, sept bras x 5 pas : 1.2e-04 au
  pire. Arithmetique exacte partout, y compris sous lambda adaptatif.
- `lam0` contre `bon4` case par case, 20 prompts : ecart max median **0.89**, 2.96 au
  pire. Le constat 11 tient a n double ; l'appariement case par case reste casse.

### Ou meurent les lignees

Nombre moyen de racines x_T distinctes apres chaque pas planifie :

| bras | n | t=80 | t=60 | t=40 | t=20 | t=0 | reech. | div_pix |
|---|---|---|---|---|---|---|---|---|
| `ctl` | 40 | 1.70 | 1.20 | 1.07 | 1.05 | 1.05 | 3.75 | 0.092 |
| `adapt` | 40 | 2.45 | 1.75 | 1.55 | 1.40 | 1.40 | 3.88 | 0.153 |
| `floor` | 40 | 3.83 | 3.08 | 2.20 | 1.73 | 1.73 | 2.00 | 0.151 |
| `lam2` | 20 | 3.00 | 2.15 | 1.85 | 1.85 | 1.85 | 4.00 | 0.215 |
| `fadapt` | 40 | 3.90 | 3.35 | 2.67 | 2.12 | 2.12 | 2.02 | 0.200 |
| `floor2` | 20 | 4.00 | 3.50 | 3.10 | 2.85 | 2.85 | 2.15 | 0.263 |
| `lam0` | 20 | 4.00 | 4.00 | 4.00 | 4.00 | 4.00 | 0.00 | 0.335 |

Part des pas inertes (poids uniformes) : 90 % a t = 80 pour les trois bras a plancher,
puis 50 % a t = 60, environ 30 % a t = 40, 15 a 22 % ensuite. Sans plancher : 0 a 12 %.

Confrontation aux predictions encore ouvertes :

- constat 7, prediction 3 (`lam2`) : 3.00 apres t = 80 contre environ 2.9 predit,
  **tenue** ; 1.85 a la fin contre 1.3-1.6, **ratee par le haut**.
- constat 7, prediction 4 (`floor2`) : 4.00 apres t = 80, **tenue**. 2.85 a la fin.
- constat 12, `adapt` : 2.45 apres t = 80 contre 2.5-2.9, 1.40 a la fin contre 1.5-2,
  **les deux juste en dessous**. Lambda median a t = 80 : 3.53 (3.91 a 20 prompts).
- constat 12, `fadapt` : au moins 3 ancetres apres t = 60 predit, 3.35 obtenu, **tenue**.

### La reward, appariee contre `ctl`

`ctl` sur ses 40 prompts : 1.05 lignees, `ir_max` +0.958, `ir` moyen des quatre +0.824.

| bras | n | lignees | div_pix | ir_max | ir moyen |
|---|---|---|---|---|---|
| `adapt` | 40 | +0.35 +/- 0.08 | +0.061 +/- 0.012 | -0.099 +/- 0.044 | -0.171 +/- 0.056 |
| `floor` | 40 | +0.68 +/- 0.19 | +0.059 +/- 0.021 | -0.091 +/- 0.055 | -0.153 +/- 0.075 |
| `lam2` | 20 | +0.80 +/- 0.16 | +0.126 +/- 0.020 | -0.132 +/- 0.120 | -0.198 +/- 0.125 |
| `fadapt` | 40 | +1.08 +/- 0.17 | +0.108 +/- 0.020 | -0.106 +/- 0.058 | -0.214 +/- 0.078 |
| `floor2` | 20 | +1.80 +/- 0.24 | +0.175 +/- 0.024 | -0.099 +/- 0.076 | -0.302 +/- 0.111 |
| `lam0` | 20 | +2.95 +/- 0.05 | +0.247 +/- 0.015 | -0.098 +/- 0.072 | -0.532 +/- 0.115 |

Entre bras de correction, apparies :

| comparaison | n | lignees | div_pix | ir_max | ir moyen |
|---|---|---|---|---|---|
| `fadapt - floor` | 40 | +0.40 +/- 0.08 | +0.050 +/- 0.010 | -0.015 +/- 0.021 | -0.061 +/- 0.033 |
| `floor2 - fadapt` | 20 | +0.60 +/- 0.15 | +0.058 +/- 0.014 | +0.004 +/- 0.029 | -0.098 +/- 0.039 |
| `floor2 - lam2` | 20 | +1.00 +/- 0.26 | +0.048 +/- 0.023 | +0.033 +/- 0.116 | |

**1. Le prix en `ir_max` est plat.** Six bras qui gardent de 0.35 a 2.95 lignees de plus
perdent tous entre 0.09 et 0.13, indiscernables entre eux. Et c'est l'ecart de `ctl` a
best-of-4 : `ctl - bon4` = +0.112 +/- 0.091 sur ces 40 prompts, alors que chaque bras de
correction tombe entre +0.006 et +0.040 de `bon4`. Des qu'on desserre la selection,
`ir_max` revient au niveau de best-of-4 ; au-dela, garder plus de lignees ne coute plus
rien sur `ir_max`. Le choix n'est pas un curseur, il est binaire : tout miser sur une
lignee et prendre environ +0.1 sur best-of-4 (non significatif a n = 40, +0.056 +/- 0.052
a n = 100 au constat 5 bis), ou garder de la diversite au niveau de best-of-4.

**2. Le prix en `ir` moyen, lui, suit ce qu'on achete.** Il va de -0.15 (`floor`) a -0.53
(`lam0`) dans l'ordre des lignees gardees. C'est mecanique : sous effondrement les quatre
images sont quatre clones de la meilleure, donc `ir` moyen est presque `ir_max` ; quatre
images differentes ont une moyenne plus basse. `ir_max` sur une population effondree ne
mesure qu'une image. Les deux chiffres sont a rapporter ensemble.

**3. `floor2` domine.** A `ir_max` egal (+0.004 +/- 0.029 contre `fadapt`), il garde
+0.60 lignee et +0.058 de diversite pixel de plus. Le plancher est ce qui rend t = 80
inerte, lambda = 2 est ce qui empeche t = 60 et t = 40 de refaire l'effondrement que le
plancher a seulement retarde (`floor2 - lam2` : +1.00 lignee a reward egale). Reserve :
n = 20 contre 40.

**4. Les trois lectures du constat 12 qui tombent.** `adapt` coute -0.099 +/- 0.044, pas
-0.028 : il n'est pas gratuit, il est domine (moins de lignees que `floor` pour le meme
prix). `fadapt` ne divise pas par deux le cout de `floor` (-0.015 +/- 0.021 entre eux) ;
ce qu'il achete en plus, ce sont 0.40 lignee, a 5 erreurs-types. Et le mecanisme annonce
tient en partie : la bisection de `fadapt` ne mord que dans 5 % des runs a t = 80 (le
plancher rend le pas inerte avant), mais dans 28 %, 48 % et 35 % a t = 60, 40 et 20, avec
un lambda median autour de 4 quand elle mord. Le lambda median de 10.00 affiche par
`f_probe.py` pour `fadapt` est celui des pas ou elle ne mord pas.

**5. L'information du premier pas, a 40 prompts** : Kendall tau +0.117 +/- 0.055, top-1
30 % contre 25 %. Le chiffre monte depuis +0.067, mais il lit `bon4[j]` comme la
continuation de la racine j, et le controle `lam0` dit que ce n'est pas mesurable ainsi.
Il reste sous le constat 11 et ne sert d'argument a rien tant que le test sur les latents
n'est pas fait.

## 14. Comment reparer : ce qui est mesure, ce qui ne l'est pas

Toutes les pistes ci-dessous sont a ecrire par l'auteur, dans les fichiers qu'indique le
constat 9. Ce qui suit donne les trade-offs, pas le choix.

### Mesure, classe par ce que les donnees disent

A `ir_max` indiscernable (environ -0.10 contre `ctl`, constat 13), du plus protecteur au
moins protecteur :

1. **plancher + lambda = 2** (`floor2`) : 2.85 lignees, div_pix 0.263 sur 0.335
   possible. Deux changements : le plancher dans la branche `max` de `_potential_terms`
   (`smc/fk.py`), et lambda dans la configuration. Trade-off : lambda = 2 est la
   configuration qualitative des auteurs, pas celle d'evaluation (lambda = 10) ; le gain
   publie de FK sur best-of-n est mesure a 10.
2. **plancher + lambda bisecte** (`fadapt`) : 2.12 lignees. Plus cher en plomberie
   (`adaptive_lam`, `ess_target`, `lam_max` a exposer dans `scripts/run_sd_baseline.py`)
   pour moins de lignees que `floor2`.
3. **plancher seul** (`floor`) ou **lambda = 2 seul** (`lam2`) : 1.73 et 1.85 lignees. Le
   plancher retarde l'effondrement d'un a deux pas, lambda = 2 le ralentit ; aucun des
   deux ne suffit.
4. **lambda bisecte seul** (`adapt`) : 1.40 lignees. A ecarter en l'etat, pour la raison
   structurelle du constat 12 (cible d'ESS sous le seuil, reechantillonnage a chaque pas).

Deux reserves communes. Le plancher a 0 est propre a ImageReward, dont le premier pas est
negatif pour les quatre particules dans 90 % des runs : pour une autre reward il faut un
autre plancher, donc un parametre et pas une constante. Et aucune piste ne rend un
meilleur `ir_max` que `ctl` : elles rendent de la diversite au prix de l'avance de `ctl`
sur best-of-4.

### Non mesure, a tester

- **Commencer le calendrier plus tard.** Le levier le moins cher, et deja dans les donnees
  du constat 5 : `S80` (un seul reechantillonnage) finit a 1.85 lignees, `T2` (premier pas
  affaibli) a 1.55. Retirer t = 80 du calendrier revient a rendre ce pas inerte sans
  toucher au potentiel, pour toute reward. La configuration qualitative des auteurs
  commence a `t_start = 20` sur 100. Trade-off : moins de pas de selection, donc moins de
  pilotage, et le plancher reste necessaire aux pas suivants si la reward y est encore
  negative.
- **Un seuil de reechantillonnage sous la cible d'ESS.** `adapt` echoue parce que la
  cible (ESS = k/2) est sous le seuil (ESS < k), donc chaque pas reechantillonne. Le
  reechantillonnage adaptatif standard (Chopin et Papaspiliopoulos, chapitre 10) ne
  reechantillonne que sous ESS < k/2. Avec une cible au-dessus du seuil, les pas
  bisectes ne reechantillonneraient plus du tout et les poids s'accumuleraient dans
  `logW` jusqu'au pas terminal. Trade-off : c'est plus de lignees par construction, mais
  la selection est alors reportee plutot qu'adoucie, et a lambda = 10 au pas terminal
  elle peut effondrer en un seul coup.
- **Le profil montant de lambda** que la table du constat 8 appelle (environ 4 a t = 80,
  environ 15 a t = 0). `adapt` et `fadapt` ne peuvent que descendre sous `lam_max = 10` ;
  « plus fort tard » demande `lam_max = 100`. Bras manquant.
- **Rapporter `ir` moyen a cote de `ir_max`.** Pas une reparation, une mesure : `ir_max`
  sur une population effondree est la reward d'une seule image, et cache ce que fait le
  pilotage (constat 13, point 2).

- **Plus de particules que d'images rendues.** La coalescence de la genealogie est une
  affaire de k et de nombre de reechantillonnages (constat 15) : k = 16 particules et
  rendre les quatre meilleures de racines distinctes. Trade-off : quatre fois le cout
  GPU, et ce n'est plus la configuration comparee a best-of-4 a budget egal.

### Ce qui ne peut pas marcher

Centrer, decaler ou normaliser la reward par une quantite partagee entre particules, et
changer de potentiel (`max`, `difference`, `sum`) : invariants au premier pas (constats 4
et 8), ESS identique au bit pres. Et, plus largement, tout reglage qui agit sur les
**poids** sans changer le nombre de reechantillonnages : il ralentit la coalescence, il
ne l'arrete pas (constat 15, `adapt`).

### Ce qui reste a trancher avant d'ecrire quoi que ce soit

Le test sur les latents du constat 11 (un prompt, `callback_on_step_end` aux pas 0 et 1
contre `initial_state` puis un `step`). Il ne change aucune des pistes ci-dessus, qui ne
lisent jamais `bon4` case par case, mais il decide si les constats 3, 5 et 5 bis
survivent.

## 15. Relecture independante : degenerescence des poids, degenerescence des chemins

Tout ce qui precede mesure l'effondrement a l'ESS, puis compte les lignees. Ce sont deux
quantites differentes, et le dossier les a laissees se confondre.

**Ce que l'ESS mesure.** `collapse_lab/l_target_ess.py` prend les quatre images de
`bon4` (quatre racines menees librement), les repondere par `exp(lambda * ir)` et lit
l'ESS, sur 100 prompts :

| lambda | ESS mediane sur 4 | Q1 - Q3 | part < 1.5 |
|---|---|---|---|
| 0.5 | 3.85 | 3.71 - 3.93 | 0 % |
| 1 | 3.51 | 3.16 - 3.74 | 0 % |
| 2 | 2.83 | 2.28 - 3.26 | 6 % |
| 5 | 1.89 | 1.25 - 2.45 | 37 % |
| **10** | **1.23** | **1.02 - 1.90** | **59 %** |

C'est l'echantillonnage preferentiel de la cible p(x0) exp(lambda r) avec le prior pour
proposition, autrement dit best-of-4 repondere, l'estimateur **sans** pilotage. A lambda
= 10 il vaut 1.23 : quatre tirages libres representent la cible aussi mal que le premier
pas de `fk4` (ESS 1.24, Q1 - Q3 1.01 - 2.00, 58 % sous 1.5, constat 2 bis ; les deux
distributions sont les memes). C'est **pourquoi FK a besoin de pas intermediaires** a ce
lambda : deplacer les particules vers la cible avant le poids terminal. Et le pilotage
le fait : l'ESS de `exp(10 * r)` sur les quatre images finales de chaque bras vaut 2.1 a
2.9 pour tous les bras pilotes (`floor` 2.89, `floor2` 2.47, `lam2` 2.07), contre 1.22
pour `lam0`. Un echantillonneur exact de la cible, lui, aurait une ESS de 4 : l'ESS
1.23 n'est pas un plafond de la cible, c'est la distance du prior a la cible.

L'observation qui reste : l'etendue mediane de l'ir final libre est 1.04, celle de r_phi
a t = 80 est 0.90. La reward guide a des le premier pas l'etendue de la reward finale,
sans en avoir l'information (constat 3).

**Ce que l'ESS ne mesure pas.** Le nombre de racines distinctes est une propriete de la
**genealogie**, pas des poids. Reechantillonner k particules R fois fait coalescer
l'arbre des ancetres en O(k) generations quelle que soit la moderation des poids
(Jacob, Murray et Rubenthaler 2015 sur le stockage des chemins ; Chopin et
Papaspiliopoulos, chapitres sur le reechantillonnage, numeros a verifier). Avec k = 4
et R = 4, la coalescence est presque sure. `adapt` en est la demonstration propre : ESS
tenue a 2.0 a chaque pas, 3.88 reechantillonnages, **1.40 lignees**. Et sur les sept
bras, le compte final de lignees suit le nombre de reechantillonnages avant lambda :

| bras | lambda | reech. | lignees |
|---|---|---|---|
| `lam2` | 2 | 4.00 | 1.85 |
| `adapt` | 3.5-10 | 3.88 | 1.40 |
| `ctl` | 10 | 3.75 | 1.05 |
| `floor2` | 2 | 2.15 | 2.85 |
| `fadapt` | 4-10 | 2.02 | 2.12 |
| `floor` | 10 | 2.00 | 1.73 |
| `lam0` | 0 | 0 | 4.00 |

Par prompt, l'ESS repondere de `bon4` ne predit pas le nombre de lignees (correlation
-0.02 pour `ctl`, -0.39 pour `floor2`) : meme signal.

**Ce que ca change a la lecture.**

- Le plancher marche parce qu'il rend des pas **inertes**, donc supprime des
  reechantillonnages (3.75 -> 2.00), pas parce qu'il adoucit les poids. Lambda = 2 seul
  reechantillonne a tous les pas et n'y gagne que 0.8 lignee. Les deux ensemble : moins
  de reechantillonnages, et ceux qui restent a poids moderes.
- Aucun reglage qui agit sur les poids seuls (potentiel, centrage, lambda bisecte a
  cible sous le seuil) ne peut arreter la coalescence : il la ralentit. Ce qui l'arrete,
  c'est moins de reechantillonnages (calendrier plus court, seuil sous la cible d'ESS,
  pas inertes) ou plus de particules que d'images rendues (constat 14).
- Le prix plat de -0.10 sur `ir_max` (constat 13) se relit ainsi : les pas intermediaires
  a lambda = 10 sont ce qui porte l'avance de `ctl` sur best-of-4 ; tout bras qui en
  neutralise ou en adoucit tombe au niveau de l'echantillonnage preferentiel simple.

Ce que cette relecture ne remet pas en cause : le premier pas est le mauvais endroit
pour choisir (constat 3, sous la reserve du constat 11), et le plancher manquant est un
ecart a la reference (constat 6).


## 16. La reference : l'ecart au papier est borne, pas ferme

Source : `docs/reference_config.md`, `results/sd_authors_R0.json`, `collapse_lab/ref/`,
depouillement `ref/parse_authors.py` ; le detail horodate dans `collapse_lab/ASSESSMENT.md`,
section C de l'etat final.

La configuration du papier (max, [0, 20, 40, 60, 80], lambda 10, k 4, DDIM eta 1, 100 pas,
CFG 7.5, ImageReward sur l'estimee de Tweedie) est celle de ce depot ; les defauts du script
publie (`diff`, 5-30-5) sont une autre configuration, que l'annexe du papier note plus bas.
Le code publie differe par quatre choix d'implementation non ecrits : statistique `max`
planchee a 0, multinomial a chaque pas planifie (poids plats compris), reechantillonnage
adaptatif de la population terminale, VAE du pipeline pour le decodage du guide.

Sur les 100 prompts du benchmark, SD v1.5, contre `bon4` apparie :

| lecture | code | n | `ir_max` | contre `bon4` |
|---|---|---|---|---|
| table 1 du papier | | | 0.898 | +0.161 |
| `ctl`, ce depot | `smc/` | 100 | 0.826 | +0.056 +/- 0.052 |
| `R1`, `smc/` avec leurs quatre choix | `smc/` | 100 | 0.756 | -0.011 +/- 0.041 |
| leur code, quatre runs groupes | le leur | 220 | 0.702 | -0.110 +/- 0.036 |
| les memes quatre runs sur leurs 40 prompts communs | le leur | 40 x 4 | | -0.35, -0.04, -0.13, +0.10 |

Leur pipeline sans FK, a generateur egal, rend les quatre rewards de `bon4` a la
quatrieme decimale ; leur scorer ImageReward vaut l'officiel a la troisieme. Les deux
implementations partent donc des memes images et les notent pareil. Avec le filtre, leur
moyenne est sous best-of-4 et leurs quatre runs s'ecartent entre eux de plus que l'effet du
papier : deux d'entre eux partagent x_T et bruit DDIM et ne different que par le flux du
tirage multinomial, et ils rendent -0.13 et +0.10 (ecart-type par prompt entre leurs runs :
mediane 0.25 ; `ctl` de ce depot bouge de 0.04 entre deux sessions sur les memes prompts).
Un run sur quatre atteint le +0.16 a une erreur-type ; aucune moyenne ne l'atteint. Lecture
du code (23/09, `fkd_class.py`, `fkd_pipeline_sd.py`) : aucun reensemencement, aucun biais
d'un chemin de graine sur l'autre ; sans generateur le multinomial avance le flux global et
change le bruit DDIM des pas suivants, avec generateur il n'y touche pas ; le poids terminal
divise par le produit float32 des poids intermediaires, qui peut deborder. Rien de cela
n'explique 0.23 entre deux runs a bruit egal (trois erreurs-types) : **la dispersion de leur
filtre est mesuree, pas expliquee.** Regle pre-enregistree : ecart **borne**, pas ferme ; il
n'est pas dans l'implementation, et une part est dans la variance d'un filtre a quatre
particules qui garde une racine.

## 17. La coalescence se lit sur les poids seuls

`n_coalescence.py`. Rejouer le peigne systematique (integre sur son decalage `u`) ou le
multinomial sur les poids enregistres a chaque pas planifie predit le nombre moyen de
racines finales de quinze bras a **0.07 pres** (`ctl` 1.08 contre 1.06, `floor` 1.73 contre
1.73, `fadapt` 2.16 contre 2.12, `floor2` 2.94 contre 2.94, `R1` 1.25 contre 1.19) et la part
de runs a une racine a trois points pres ; correlation par run 0.87 a 0.99 sur les bras qui
ont de l'etendue. Le pilotage n'entre pas dans la prediction. `adapt` tient l'ESS a 2.0 a
chaque pas et finit a 1.4 racine : l'ESS mesure la degenerescence des poids, pas celle des
chemins (constat 15). A poids plats le peigne est l'identite et le multinomial ne l'est pas :
quatre passages plats laissent 1.58 racines sur 4, et le code publie perd des racines avant
toute information. Prediction pre-enregistree sur `thr05`, ecrite par ce modele avant la
mesure : 1.84 racines, 61 % a une racine, 1.12 reechantillonnement ; mesure : 2.00, 62 %,
0.97. Le plan disait 2.4 a 2.8. Figure : `out/fig_coalescence.png`.

## 18. Reechantillonner moins ne garde pas les lignees

`thr05` (plancher, reechantillonner seulement si ESS < k/2) : 0.97 reechantillonnement par
run, 2.0 racines, 62 % a une racine (predit par le constat 17, voir ci-dessus). Quand le
seuil declenche enfin, les poids accumules sont pointus et un seul passage prend presque
tout. `ir_max` -0.06 +/- 0.10 contre `ctl`. `late` (calendrier sans t = 80) ne repare rien
non plus : 86 % a une racine a n = 300, 1.14 racine, pour +0.04 +/- 0.03 sur `ir_max`. Ce
dernier chiffre dit autre chose : retirer le pas t = 80 ne coute rien au score, donc ce pas
ne porte pas d'information utile a la selection, sans avoir besoin de `bon4` pour le dire.

## 19. Rejouabilite : le chemin libre l'est, le chemin avec reechantillonnage ne l'est pas

Le pipeline diffusers a generateur seme rend le 23/09 les rewards de `bon4` (20/09) a la
troisieme decimale. Le chemin `smc.models.StableDiffusion` : deterministe dans une session ;
entre sessions, le 21/09 et le matin du 22/09 rendent les memes chiffres a 0.0000, et le soir
du 22/09 d'autres racines et d'autres `ir_max` (jusqu'a 1.6 d'ecart sur un prompt), memes
poids, meme code, meme graine. La session C (constat 20) separe les deux cas : `lam0`
(lambda = 0, aucune reward lue) rend `bon4` case par case a la correlation 1.00, `ctl` rend
le `ctl` du 21/09 a 0.70 avec la meme racine dans 30 % des prompts. Le chemin libre est donc
rejouable d'une session a l'autre ; celui qui lit la reward du guide (VAE ft-mse, ImageReward,
BERT en fp16) ne l'est pas, et 1e-3 sur une reward suffit a deplacer une dent du peigne a
quatre cases.

*Test du 23/09 apres-midi (`nuit4.sh`, `u_determinism.py`, non pre-enregistre).* `ctl` sur les
prompts 0 et 1 dans trois processus separes : deux sans rien changer, un avec
`cudnn.benchmark = False` et `use_deterministic_algorithms(True)` ; puis la version de
`probe.py` du 22/09 matin (celle de la session A) sur les memes prompts. Les quatre rendent
les **memes quatre rewards** a la quatrieme decimale, egales a celles de la session C du matin
et de la session B du 22/09 soir (six prompts communs, `ctl_b1` = `ctl_C` a 0.0000), a travers
un redemarrage du pod entre la session C et le test. Elimines : le processus, les drapeaux
deterministes, le chemin de cache (`~/.cache` pour B, `work/hf_cache` pour C et le test, memes
revisions), le pod, le venv `sd` (aucune installation depuis le 20/09), `smc/` (inchange depuis
le 21/09 17h27, avant toutes les sessions), la reecriture de `probe.py`. Ce qui reste : les
sessions qui rendent la reference du 21/09 (la reference elle-meme, la session A du 22/09
matin) tournaient a **87 a 90 s par run** ; toutes celles depuis le 22/09 soir tournent a
**55 a 60 s**, meme code, meme pipeline. Les deux groupes different par le chemin d'execution
de la machine (noyaux fp16 choisis, materiel), pas par quoi que ce soit dans le depot ; la
machine du premier groupe n'existe plus et le point ne peut pas etre pousse plus loin.
Consequence inchangee : tout appariement par case entre fichiers de sessions differentes est
invalide ; les moyennes par prompt entre bras FK restent utilisables ; a l'interieur du
groupe depuis le 22/09 soir, l'appariement par case tient.

## 20. Session C : les constats 3 et 5 bis tiennent, reformules

`out/probe_C.json`, `t_sessionC.py` : `ctl`, `lam0`, `floor2` a 100 prompts dans **un seul
processus**, le seul appariement par case valide (constat 19). Predictions en face dans
`docs/protocol_sd.md`, « Pre-registration of session C ».

- **Constat 3.** Kendall tau entre le classement de `r_phi(t = 80)` dans `ctl` et l'`ir`
  libre de la meme racine (`lam0`) : **+0.137 +/- 0.050**, top-1 dans 35 % des prompts contre
  25 % au hasard (n = 100). Le premier pas porte un peu d'information, pas aucune ; le
  +0.067 du constat 3 etait lu sur un appariement invalide.
- **Constat 5 bis.** A meilleure racine libre 0.779 ; M racine moyenne 0.233 ; B la racine que
  `ctl` garde, lue libre, 0.466 ; C ce que `ctl` en tire 0.799. B - M = +0.233 +/- 0.047 (la
  racine gardee vaut un tiers du chemin vers la meilleure, rang moyen 2.04 sur 4) ;
  **A - B = +0.313 +/- 0.042** (l'effondrement coute encore 0.31 de racine) ; C - B = +0.333
  +/- 0.038 (le pilotage rend un peu plus) ; C - A = +0.021 +/- 0.038 (le solde sur best-of-4).
- **`floor2` apparie par x_T** : `ir_max` **-0.012 [-0.082, +0.060]** contre `ctl` (predit
  [-0.14, -0.02], rate par le haut), `ir` moyen des quatre -0.250 +/- 0.045, 3.03 racines, 4 %
  a une racine. Le prix des lignees sur `ir_max` est de l'ordre de l'avance de `ctl` sur
  best-of-4 (+0.030 +/- 0.037 dans cette session, +0.056 le 21/09), pas plus ; le prix sur
  la moyenne des quatre reste.

Ce que cela change au constat 13 : « prix plat de -0.10 » etait trop dit. A n = 40 chaque IC
couvre zero, la moyenne des sept bras est -0.07 [-0.18, +0.04], et `floor2` apparie par x_T
a n = 100 vaut -0.01. La phrase tenable : **aucune configuration ne bat best-of-4 de plus
que le bruit sur `ir_max`, pendant que la diversite varie d'un facteur trois et l'`ir` moyen
de -0.15 a -0.56.** Les gros effets sont sur les lignees et la moyenne.
