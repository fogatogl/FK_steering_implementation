# Pourquoi les quatre images finales descendent d'un seul x_T

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

**Ce que la correction ne promet pas.** Garder quatre lignees ne rend pas quatre images
meilleures. Sur les dix premiers prompts, le bras plancher perd 0.17 d'ImageReward
contre le controle. L'effondrement est le prix du pilotage, pas son moyen (constat 5 bis)
-- diversite et reward restent deux questions.

---

Etat au 22/09 07 h. Les constats 1 a 6 et 8 sont fermes et rejouables sans GPU depuis
`collapse_lab/*.py`. Le constat 7 porte les predictions ecrites avant les donnees, le
constat 10 ce que la sonde en dit, le constat 11 un controle qui echoue et qui met en
doute les constats 3, 5 et 5 bis, le constat 12 les deux bras de correction en cours.

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

Les deux premiers runs, anecdotiques mais dans le sens predit : `adapt` ESS = [2.0, 2.0,
2.84, 3.61, 3.47], lignees [2, 1, 1, 1, 1] ; `fadapt` ESS = [4.0, 4.0, 2.0, 2.77, 2.61],
lignees [4, 4, 2, 2, 2].

