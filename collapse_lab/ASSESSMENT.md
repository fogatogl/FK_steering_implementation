# Les corrections resolvent-elles l'effondrement ? Protocole, predictions, verdict

*Bloc propose pour `collapse_lab/README.md` (l'auteur l'insere s'il le veut) :*

    python collapse_lab/r_solutions.py      # les trois criteres de "resolu", par bras, IC bootstrap
    python collapse_lab/n_coalescence.py    # les lignees predites depuis les poids seuls, contre l'observe
    python collapse_lab/p_two_rewards.py    # F4 : ir_max et ir moyen apparies contre ctl (venv ddpm)
    python collapse_lab/o_ancestry_fig.py --pid <id> --arms lam0 ctl floor2 R1   # F1, runs du 22/09 et apres
    python collapse_lab/q_image_grid.py     # F2 depuis out/images/ (--choose : les six prompts par regle)
    python collapse_lab/ref/parse_authors.py  # R0 : le code des auteurs contre bon4, ctl, la table 1
    python scripts/fig_three_scales.py      # F5 : CIFAR / CelebA / SD (venv ddpm, samples/*.pt)
    # GPU : collapse_lab/nuit2.sh (R1, R0, B1, bissection, thr05, rise) ; m_latents.py (constat 11)
    # Le code des auteurs : /home/onyxia/work/fkd_ref/ (clone, hors depot), venv sd + shim google.genai

Fichier separe de `FINDINGS.md`, qui reste a l'auteur. Ici : les criteres fixes avant les
donnees, les predictions horodatees, puis ce que les donnees ont rendu. Depouillement :
`python collapse_lab/r_solutions.py` (sans GPU). Runs : `collapse_lab/nuit2.sh`.

## Ce qu'on teste

Le constat 14 de `FINDINGS.md` classe cinq corrections mesurees (`floor2` > `fadapt` >
`floor`, `lam2` > `adapt`) et en laisse trois non mesurees. Le classement repose sur
20 prompts pour `floor2` et `lam2`, 40 pour les autres. Et le constat 15 change la question
: a lambda = 10 la cible `p(x0) exp(lambda r)` restreinte a quatre tirages libres a une ESS
mediane de 1.23. Quatre particules ne peuvent pas porter de diversite a ce lambda, quoi
que fasse le noyau. « Resolu » doit donc se lire par couche.

## Criteres, fixes le 22/09 a 18h30, avant le premier run de `nuit2.sh`

Tout est apparie par prompt, sur l'intersection des `prompt_id`, contre `ctl` (la ligne FK
de reference, identique a `sd_ref_fields100.json`) et contre `bon4`.

1. **Le mecanisme (couche 2).** L'effondrement n'a plus lieu au pas non informatif.
   - part des prompts qui finissent sur **une seule racine** x_T ; `ctl` : 95 %. Succes
     propose : **sous 25 %**.
   - lignees apres t = 80 et t = 60, alignees sur la valeur de t (le bras `late` n'a pas
     t = 80).
   - **ESS ponderee des racines** a la fin : poids finaux reconstruits (a seuil 1.0,
     `logW` repart de zero a chaque pas ou ESS < k), sommes par racine, `1 / sum W_r^2`.
     C'est la diversite que le sampler pese, pas celle qu'il rend.
2. **Le plafond de la cible (couche 1).** ESS ponderee des racines contre le plafond de
   `l_target_ess.py` au lambda du bras : **2.83** a lambda = 2, **1.23** a lambda = 10.
   Un bras a lambda = 10 qui rend plus de racines que 1.23 rend des particules que sa
   cible ecrase. L'evaluation du depot (`ir_max`, `div_pix` sur les quatre images) ignore
   les poids finaux : choix legitime pour « quatre candidats », pas une preuve que le
   sampler est correct pour sa cible. Les deux lectures cote a cote.
3. **Le prix.** `ir_max`, `ir` moyen, `ir` pondere, apparies contre `ctl` et contre
   `bon4`, IC bootstrap sur les prompts. La question : la correction coute-t-elle toute
   l'avance de FK sur best-of-4 (+0.11 +/- 0.09 a n = 40) ?

## Regle sans GPU (ecrit avant, verifie en lisant `smc/fk.py`)

- **Seuil de reechantillonnage sous la cible d'ESS** (constat 14, piste 2). Si la cible
  bisectee est au-dessus du seuil, aucun pas planifie ne reechantillonne. `logW` n'est
  jamais remis a zero et vaut `acc` ; au pas terminal la branche `last and acc is not
  None` de `_potential_terms` rend `lam * r_0 - acc` ; les poids finaux valent
  `exp(lam * r_0)` sur quatre trajectoires libres. C'est de l'importance sampling sur
  `bon4`, exactement ce que `l_target_ess.py` mesure : ESS 1.23 a lambda = 10. Le bras
  « resout » l'effondrement en ne faisant plus de SMC. Regle, pas a lancer.
- **`lam_max = 100`** (piste 3). La table du constat 8 met le lambda bisecte au-dessus de
  10 au seul pas t = 20 (11.9 en mediane). Le bras differerait d'`adapt` a un pas sur
  quatre, celui ou l'effondrement est deja fait. Ecarte.
- **`S60` n'est pas un controle de `late`.** Le fichier date du 20/09 13:42, avant
  5025180 ; contre la reference actuelle il vaut +0.048 +/- 0.104 sur 20 prompts. `late`
  passe par la sonde.

## Predictions, ecrites le 22/09 a 18h30, avant les donnees

Ce que `probe.json` dit deja, a 20 prompts, et qui sert de base :

| bras | n | 1 racine | lignees | ESS pond. racines | div_pix | ir_max | ir moyen |
|---|---|---|---|---|---|---|---|
| `ctl` | 40 | 95 % | 1.05 | 1.01 | 0.092 | +0.958 | +0.824 |
| `adapt` | 40 | 60 % | 1.40 | 1.10 | 0.153 | +0.859 | +0.653 |
| `floor` | 40 | 68 % | 1.73 | 1.64 | 0.150 | +0.867 | +0.671 |
| `lam2` | 20 | 35 % | 1.85 | 1.67 | 0.215 | +0.829 | +0.621 |
| `fadapt` | 40 | 30 % | 2.12 | 1.72 | 0.200 | +0.852 | +0.610 |
| `floor2` | 20 | 5 % | 2.85 | 2.59 | 0.263 | +0.862 | +0.517 |
| `lam0` | 20 | 0 % | 4.00 | 4.00 | 0.335 | +0.863 | +0.287 |

1. `floor2` a 40 prompts : sous 10 % de prompts a une racine, 2.6 a 2.9 lignees, ESS
   ponderee des racines entre 2.4 et 2.8, soit au plafond de sa cible (2.83). `ir_max`
   reste a -0.10 +/- 0.06 de `ctl` et a moins de 0.05 de `bon4`.
2. `lam2` a 40 prompts : 30 a 45 % a une racine, 1.7 a 2.0 lignees. `floor2 - lam2`
   garde environ +1.0 lignee a `ir_max` egal.
3. `late` : entre `ctl` et `floor`. Retirer t = 80 revient a rendre ce pas inerte dans
   100 % des runs au lieu de 90 %, mais sans plancher a t = 60 (ou la reward est encore
   negative partout dans 40 % des runs) l'effondrement reprend un pas plus tot que pour
   `floor` : **1.4 a 1.8 lignees, 40 a 70 % a une racine**, `S80` (1.85 avec un seul pas)
   comme borne haute. Prediction la plus exposee : `ir_max` **indiscernable de `ctl`**
   (moins d'une erreur-type), parce que t = 80 ne porte aucune information (constat 3) et
   que le pilotage garde ses trois autres pas. Si elle tient, `late` domine `floor2` sur
   le prix et le classement du constat 14 change ; si `late` paie aussi -0.10, le prix
   plat du constat 13 est confirme sur un bras qui ne touche ni lambda ni potentiel.
4. Lecture d'ensemble attendue : aucun bras a lambda = 10 ne descend sous 25 % de prompts
   a une racine, parce que la cible l'interdit ; le seul qui y arrive change la cible
   (lambda = 2). « Resoudre l'effondrement » a lambda = 10 avec k = 4 n'est pas un
   probleme de sampler.

## Verdict partiel, 22/09 19h45 : `floor2` et `lam2` a 40 prompts, `late` a 20 et a 300

Sortie de `r_solutions.py` sur `probe.json` (40 `ctl`, `floor`, `adapt`, `fadapt`, `floor2`,
`lam2` ; 20 `lam0`, `late`). Controles : `ctl` identique a `sd_ref_fields100.json` (0.0000),
ESS recalculee a 1.2e-4, poids finaux reconstruits a 1.0e-4.

| bras | n | 1 racine | lignees | ESS pond. | plafond cible | div_pix | ir_max - ctl | ir moyen - ctl |
|---|---|---|---|---|---|---|---|---|
| `ctl` | 40 | 95 % | 1.05 | 1.01 | 1.51 | 0.092 | | |
| `late` | 20 | 100 % | 1.00 | 1.00 | 1.40 | 0.087 | +0.03 [-0.24, +0.31] | |
| `adapt` | 40 | 60 % | 1.40 | 1.10 | 1.51 | 0.153 | -0.10 [-0.19, -0.02] | -0.17 |
| `floor` | 40 | 68 % | 1.73 | 1.64 | 1.51 | 0.150 | -0.09 [-0.20, +0.01] | -0.15 |
| `lam2` | 40 | 35 % | 1.82 | 1.62 | 2.72 | 0.205 | -0.05 [-0.19, +0.08] | |
| `fadapt` | 40 | 30 % | 2.12 | 1.72 | 1.51 | 0.200 | -0.11 [-0.22, +0.01] | -0.21 |
| `floor2` | 40 | **5 %** | 3.00 | 2.73 | 2.72 | 0.286 | -0.08 [-0.23, +0.06] | -0.32 |
| `lam0` | 20 | 0 % | 4.00 | 4.00 | 4.00 | 0.335 | -0.10 [-0.23, +0.04] | -0.53 |

Predictions du 22/09 18h30, confrontees :

1. `floor2` a 40 : **tenue**. 5 % a une racine (predit sous 10 %), 3.00 lignees (predit
   2.6-2.9, juste au-dessus), ESS ponderee 2.73 pour un plafond de 2.72 : le bras rend
   exactement la diversite que sa cible porte (ratio 1.01). `ir_max` -0.08 contre `ctl`
   (predit -0.10 +/- 0.06), +0.03 contre `bon4` (predit sous 0.05).
2. `lam2` a 40 : **tenue**. 35 % a une racine (predit 30-45), 1.82 lignees (predit 1.7-2.0).
   `floor2 - lam2` : +1.18 lignee [+0.85, +1.48] a `ir_max` egal (-0.03 [-0.17, +0.12]).
3. `late` : **ratee sur les lignees, tenue sur le prix**. 1.00 lignee a 20 prompts et 1.14 a
   300 (`sd_s60_full.json`), 86-100 % a une racine, la ou 1.4-1.8 etaient predits : retirer
   t = 80 ne fait que reporter le premier reechantillonnage a t = 60, ou l'ESS mediane vaut
   1.4 et le peigne tue autant. `ir_max` +0.03 contre `ctl` a 20 prompts, +0.04 +/- 0.03 a
   300 : le seul bras qui ne paie pas, et il fait une evaluation de reward de moins.
4. Lecture d'ensemble : **tenue**. Aucun bras a lambda = 10 ne passe sous 25 % de prompts a
   une racine ; le seul qui y passe change la cible (lambda = 2).

Le prix plat du constat 13, en un chiffre : moyenne d'`ir_max` sur les six bras de
correction contre `ctl`, n = 40, **-0.078 [-0.177, +0.024]** ; sans `late`, -0.106 [-0.193,
-0.019]. Le prix est reel pour tout bras qui rend des lignees ; `late` n'en rend pas et ne
paie pas.

## Brouillon de constat : la coalescence se lit sur les poids seuls (22/09, 20h, `n_coalescence.py`)

Pour chaque run, on rejoue la genealogie depuis les poids enregistres a chaque pas planifie,
sans rien savoir du pilotage : la loi du vecteur des racines se propage pas a pas,
exactement, sous le peigne systematique integre sur `u` (les affectations distinctes sont en
nombre fini) et sous le multinomial des auteurs a chaque pas (4^4 affectations). Controles :
premier pas seul de `ctl` sur les 20 prompts de `fk4_stat` = **1.835**, le chiffre de
`b_comb.py` ; `lam0` = 4.000.

| bras | n | observe | predit (peigne) | predit (multinomial a chaque pas) | P(1 racine) obs / predit | reech. |
|---|---|---|---|---|---|---|
| `ctl` | 40 | 1.05 | 1.07 | 1.03 | 95 % / 93 % | 3.75 |
| `late` | 20 | 1.00 | 1.07 | 1.06 | 100 % / 93 % | 2.85 |
| `adapt` | 40 | 1.40 | 1.41 | 1.16 | 60 % / 60 % | 3.88 |
| `floor` | 40 | 1.73 | 1.73 | 1.19 | 68 % / 67 % | 2.00 |
| `lam2` | 40 | 1.82 | 1.79 | 1.30 | 35 % / 37 % | 3.95 |
| `fadapt` | 40 | 2.12 | 2.16 | 1.34 | 30 % / 30 % | 2.02 |
| `floor2` | 40 | 3.00 | 2.97 | 1.50 | 5 % / 3 % | 1.93 |
| `lam0` | 20 | 4.00 | 4.00 | 1.58 | 0 % / 0 % | 0.00 |

La prediction du plan (+/- 0.15, bon ordre) est **tenue** avec de la marge : l'ecart maximal
est 0.07 (`late`), les sept bras sont dans l'ordre, et la part de runs a une racine est
predite a trois points pres. Run par run, la correlation entre l'esperance predite et
l'observe vaut 0.97 pour `ctl`, 0.99 pour `floor`, 0.95 pour `fadapt`, 0.84 pour `floor2`.
Le nombre de lignees est donc une fonction des poids et du nombre de rééchantillonnages,
pas du pilotage : c'est la degenerescence des chemins, et l'ESS d'un pas ne la mesure pas
(`adapt` tient l'ESS a 2 et perd 2.6 racines sur 4).

Le point fin : a poids uniformes le peigne est l'identite, le multinomial ne l'est pas.
Quatre pas a poids plats sous multinomial laissent 1.58 racines sur 4 (`lam0`, colonne
multinomial) : c'est le drift neutre de la genetique des populations, et le code des auteurs
le subit a chaque pas inerte que leur plancher produit. Sur les poids de `ctl`, leur
resampler laisserait 1.03 racines contre 1.07 au peigne : leur depot s'effondre au moins
autant, plancher mis a part.

Prediction pre-enregistree pour `thr05` (`docs/protocol_sd.md`, 19h55) : 1.84 racines, 61 %
a une racine, 1.12 rééchantillonnages, depuis les poids de `floor` et la regle ESS < k/2.
Loin des 2.4-2.8 du plan : avec le plancher les premiers pas sont inertes, les poids
accumules restent plats, et quand le seuil declenche enfin, ils sont pointus et un seul
passage du peigne prend presque tout.

## Nuit 2, la reference : R1 rendu (22/09, 21h45), R0 en cours

**R1, `smc/fk.py` avec les quatre choix du code publie** (plancher a 0, forme statistique,
multinomial a chaque pas planifie, guide decode par le VAE du pipeline, indices
{20, 40, 60, 80, 99}), 100 prompts, seed 2024, apparie :

| | prediction (19h35) | mesure | |
|---|---|---|---|
| `ir_max` R1 - `ctl` | -0.08 +/- 0.05 | **-0.067 +/- 0.055** (42/100 gagnes) | tenue |
| `ir_max` R1 - `bon4` | | **-0.011 +/- 0.041** | |
| lignees | 1.0 a 1.3 | **1.19**, une racine dans 81 % | tenue |
| premier pas inerte | ~90 % | 81 % | tenue, un peu moins |
| reechantillonnages | 4 a chaque run | 4.00 | tenue |
| `ir` moyen des 4 | | 0.609 contre 0.681 pour `ctl` | |

Regle pre-enregistree : l'ecart au papier est **borne**, pas ferme (R1 - `bon4` = -0.01, seuil
+0.12). Lecture : les quatre choix d'implementation du code publie, portes par `smc/`, rendent
exactement le niveau de best-of-4 sur ces prompts ; `ctl` (les choix de ce depot) garde
+0.056 au-dessus. Le plancher et le tirage multinomial a poids plats (le drift neutre du
constat coalescence : 4 -> 3 -> 2 racines avant toute information, visible sur F1) coutent
l'avance que le pilotage achete. Rien ici ne rapproche du +0.161 publie ; tout s'en eloigne.

**R0, le code des auteurs, configuration du papier**, 100 prompts, seed 42 (23h15) :
`ir_max` moyen **0.554** ; contre `bon4` **-0.216 +/- 0.061** (41/100 gagnes), contre `ctl`
-0.272 +/- 0.067, contre `R1` -0.206 +/- 0.072, contre un tirage libre `k1` +0.330. Moyenne
des quatre finales 0.414 ; 3.81 images distinctes sur 4 (le pas terminal en duplique dans 8 %
des runs) ; div_pix 0.104 ; 61 s par run. La prediction 0.77 [0.72, 0.82] est **ratee par le
bas de 0.2** : leur code, sous la configuration de leur table 1, rend ici un peu mieux que
best-of-2 et nettement moins que best-of-4.

Le diagnostic (`ref/diag_authors.py`, prompts 0 et 2, `out/diag_authors_*.json`) ferme la
lecture (b) : leur `score_batched` rend les memes valeurs que l'ImageReward officiel a la
troisieme decimale, sur les x0 decodes comme sur les finales. Et il montre le mecanisme (a) tel
qu'ecrit : a l'indice 20 les quatre rewards valent -2.2, plancheees a 0, poids plats, le tirage
multinomial garde [1, 0, 0, 3] (une racine perdue par pur hasard) ; sur le prompt 2 les rewards
restent negatives jusqu'a l'indice 60 et trois tirages plats de suite ne laissent que deux
racines avant la premiere information ; a l'indice 80 la selection tranche (ESS 1.05 sur le
prompt 0). Leur FK, c'est du drift neutre puis une selection tardive.

Ce qui reste a departager, et qui est pre-enregistre (`protocol_sd.md`, 23h30) : `R1` porte
les memes choix et rend 0.756, soit +0.21 au-dessus de `R0`. La seule difference restante
est le flux de bruit (seed 42 par le RNG global chez eux, `seed_effective` par generateur
ici). `R0g` = leur pipeline avec notre generateur, 40 prompts, apparie a `bon4`/`ctl`/`R1` par
x_T ; tourne dans `nuit2d.sh`. Prediction : si les deux codes sont equivalents, `R0g - R1`
dans +/- 0.05 ; sinon, leur pipeline differe de `smc/` sur quelque chose que la table ne liste
pas.

**Le test des latents (constat 11)**, deux prompts (`out/latents_*.json`) : x_T identique au
bit entre le pipeline et `initial_state` ; correlation par case entre les latents du pipeline
et ceux de notre modele : 1.0000 aux indices 0, 1, 10, 50, et **0.984 a 0.9999 a l'indice
99**. La prediction « divergence chaotique » est **ratee** : les trajectoires restent les
memes a 1 ou 2 % pres. Pourtant `lam0` et `bon4` different case par case (correlation 0.64
sur 80 cases, 0.89 une fois triees dans le prompt), et sur le prompt 0 le plus grand ecart
d'ImageReward (0.05 contre 0.77) tombe sur la case la moins correlee (0.984). Deux lectures,
pre-enregistrees a 23h45 : le fichier `bon4` du 20/09 n'est pas rejouable aujourd'hui, ou
ImageReward bouge de 0.7 pour un ecart fp16 sur les latents. `m_latents.py` decode et score
maintenant les deux finales a cote de la valeur enregistree ; il repasse dans `nuit2d.sh`.
Dans les deux cas, `lam0` est la baseline libre appariee par construction, et les constats 3,
5 et 5 bis se recalculent contre lui.

*Rejoue le 23/09 vers 03h15 avec ImageReward sur les finales.* (a) **Tenue** : le pipeline
d'aujourd'hui rend les rewards enregistrees de `bon4` a la troisieme decimale sur les deux
prompts ; le fichier du 20/09 est rejouable. (b) **Ratee dans sa forme forte** : les finales
de notre modele nu different de celles du pipeline de 0.17 a 0.34 (la case a 0.984 de
correlation passe de 0.048 a -0.292), pas de 0.7. Et elles different aussi des records `lam0`
de la sonde (prompt 0 : 1.11 / -0.29 / 0.69 / 0.87 aujourd'hui, 1.01 / 0.77 / 1.10 / 0.68
enregistres) : le bras `lam0` s'ecarte du modele nu quelque part, et le rejeu de `nuit2e.sh`
sur ce prompt dit d'abord si la sonde est deterministe. Tant que ce point n'est pas leve, ni
`bon4` ni `lam0` ne servent d'appariement par case ; les constats 3, 5 et 5 bis restent
suspendus.

*Rejeu des six prompts de la grille, 23/09 vers 04h00 (`--redo`).* Le `lam0` rejoue rend
**exactement** les finales du modele nu de `m_latents.py` (1.1131 / -0.2924 / 0.6885 / 0.871)
et le `R1` rejoue rend exactement le `R1` de la nuit : la sonde est deterministe, et le bras
`lam0` est bien le modele nu. Mais le `ctl` rejoue **ne rend pas** le `ctl` du 21/09
(`sd_ref_fields100.json`, que la sonde du matin du 22/09 reproduisait a 0.0000), et le `lam0`
rejoue ne rend pas le `lam0` du matin du 22/09. Le pipeline diffusers, lui, rend aujourd'hui
le `bon4` du 20/09 a la troisieme decimale. Donc : deterministe dans une session, pas d'une
session a l'autre pour le chemin `smc.models.StableDiffusion` (les noyaux fp16 choisis a
l'execution sont le suspect ; rien dans `smc/` n'a change), et l'ecart d'arrondi, amplifie sur
100 pas a eta = 1, change la racine gardee (4 prompts sur 5) et `ir_max` de -0.4 a -1.6 sur
trois des cinq prompts rejoues (`007191-0058` : 0.21 ce soir contre 1.83 le 21/09) : ce n'est
pas un bruit de mesure, c'est un autre tirage. Consequence : un
appariement **par case** entre deux fichiers ecrits a des moments differents ne mesure rien
(constat 11, resolu par une troisieme lecture) ; les comparaisons **par prompt** entre bras FK
restent valides en moyenne, le choix de racine etant de toute facon proche du hasard. Pour
garder les constats 3, 5 et 5 bis il faut `ctl` et `lam0` dans la **meme session** : 200 runs,
environ 3 h 20 de T4, a decider par l'auteur.

*Incident et correction, 23/09 vers 04h20.* Le rejeu `--redo` des six prompts de la grille avait
**remplace** dans `probe.json` les records `ctl`, `floor2` et `lam0` de la session A (21-22/09
matin) par ceux de la session B (22-23/09 nuit), plus bas de 0.4 a 1.6 sur trois prompts : toutes
les colonnes « - ctl » de `r_solutions.py` avaient glisse de +0.08. Les 18 records de la session
B sont renommes `ctl_b1`, `floor2_b1`, `lam0_b1` (ils servent aux figures F1 et F2, avec `R1`
qui est de la meme session) ; les six records de session A de `floor2` et `lam0` sont perdus
(`floor2` passe a 34 prompts, `lam0` a 17) ; les six de `ctl` sont **restaures** depuis
`sd_ref_fields100.json` (identique a la session A sur `ir`, ESS, racines) sans `logG` ni
`r_phi` ni `lineages_trace`, marques `partial_from`, et les scripts ne lisent les poids que sur
les records complets. Les six prompts ne sont pas quelconques (choisis par regle comme medians,
plus gros ecarts `ctl - bon4`, meilleurs `floor2`) : les retirer aurait biaise toutes les
colonnes « - ctl » de +0.05 environ, d'ou la restauration. Sauvegarde du fichier contamine :
`out/probe_backup_2309_0630.json`. `floor2` n'a plus que 34 prompts, et les six manquants ne
sont pas quelconques : recalcule sur 34, `floor2 - ctl` passe de -0.083 a +0.009. **Les
chiffres de `floor2` a n = 40 du verdict partiel ci-dessus, calcules avant l'incident, restent
ceux a citer** ; la sortie actuelle de `r_solutions.py` pour `floor2` est biaisee et le dit ici.
Les heures notees dans ce fichier et dans `protocol_sd.md` sont approximatives ; l'ordre est
celui des journaux `out/nuit2*.log`.

## La reference, suite (23/09, vers 03h50) : `R0g`, et leur pipeline sans FK

`R0g` = leur code, configuration du papier, **notre generateur**, 40 prompts : `ir_max`
**0.807** ; contre `R1` **-0.147 +/- 0.092** ; contre `bon4` **-0.039 +/- 0.073** ; contre
`ctl` -0.152 +/- 0.072 ; contre `R0` (leur chemin de graine, memes prompts) **+0.310 +/-
0.077**. La prediction « codes equivalents » (R0g - R1 dans +/- 0.05) est **ratee** ; celle
sur `bon4` est tenue. Et le 0.31 entre les deux chemins de graine du meme code depasse tout
ce que les graines font a best-of-4 (moins de 0.02).

Leur pipeline **sans FK**, notre generateur, 5 prompts (`authors_free_g`) : il ne rend pas
`bon4` case par case (ecarts de 0.33 a 2.69), la ou le pipeline diffusers standard le rend a la
troisieme decimale. Leur copie du pipeline ne consomme pas l'aleatoire comme l'original ; rien
de leur code ne s'apparie au notre par x_T, seulement par prompt. Deux runs en cours pour lire
`R0` et `R0g` contre **leur propre** baseline libre : `authors_free` (leur graine, 40) et
`authors_free_g` (notre generateur, 40). Predictions dans `protocol_sd.md` (05h15, 06h15).

*Leur baseline libre, 40 prompts (23/09, vers 05h20).* `authors_free` (leur pipeline sans FK,
leur graine) : `ir_max` **0.824** contre 0.846 pour `bon4` sur les memes prompts (-0.022 +/-
0.083), 0.295 par particule contre 0.288 : prediction **tenue**, leur chemin de graine est
sain et leur sampleur libre a la loi du notre. Donc, sur le meme pied : **leur FK, leur graine,
rend 0.33 +/- 0.08 de moins que leur propre best-of-4** (`R0 - authors_free`), et leur FK avec
notre generateur 0.11 +/- 0.09 de moins que leur libre correspondant (`R0g - authors_free_g`,
22 prompts, 40 en cours). Le 0.31 entre les deux chemins de graine du meme FK reste sans
explication : les deux baselines libres coincident, seule la boucle FK differe entre les deux
chemins. Reproductible (le smoke test et la nuit rendent les memes chiffres), inexplique, a
dire tel quel.

Ce que la reference dit a ce stade, en une phrase : le code publie, sous la configuration de la
table 1, rend sur ce materiel **moins que best-of-4** (de 0.11 a 0.33 selon le chemin de
graine), jamais +0.16 au-dessus ;
`smc/` avec les memes choix rend best-of-4 (R1) ; `smc/` avec ses propres choix rend +0.056
(`ctl`). L'ecart au papier est **borne**, pas ferme, et il n'est pas dans l'implementation :
aucune des deux implementations ne le produit.

## Nuit 2, suite : bissection, `thr05`, `rise` (23/09, vers 02h40)

Apparie a `ctl`, 40 prompts sauf `rise` (20). Predictions de `protocol_sd.md` (22/09 19h35 et
19h55) en face.

| bras | `ir_max` - `ctl` | prediction | lignees | prediction | reech. | racine(s) = `ctl` |
|---|---|---|---|---|---|---|
| `stat0` | -0.050 +/- 0.096 | -0.09 +/- 0.05 | 1.75 | ~1.7 | 1.88 | 25 % |
| `multi` | -0.061 +/- 0.081 | -0.02 +/- 0.04 | 1.00 | 1.0-1.1 | 4.00 | 38 % |
| `vae` | +0.009 +/- 0.083 | sous 0.03 | 1.05 | | 3.73 | **30 %** (predit >= 70 %) |
| `idx` | -0.051 +/- 0.085 | sous 0.03 | 1.00 | | 3.73 | 22 % |
| `thr05` | -0.058 +/- 0.100 | -0.10 +/- 0.06 | **2.00** | **1.84** (modele) ; 2.4-2.8 (plan) | **0.97** | 22 % |

- **`thr05` : la prediction du modele de coalescence est tenue** (2.00 racines contre 1.84,
  62 % a une racine contre 61 %, 0.97 rééchantillonnement contre 1.12), et celle du plan
  (2.4-2.8) est ratee. Rééchantillonner moins ne garde pas les lignees quand les poids
  accumules sont pointus au moment ou le seuil declenche. `ir` moyen 0.643, entre `floor`
  (0.671) et `floor2` (0.490), comme predit. L'ESS terminale sous 1.5 dans 8 % des runs
  seulement (predit plus de la moitie) : la selection reportee ne s'effondre pas au dernier
  pas, elle a deja eu lieu au pas ou le seuil a declenche.
- **Bissection** : aucun des quatre choix seul ne vaut plus de 0.06, et leur somme (-0.15)
  depasse `R1` (-0.067) : ils ne s'additionnent pas. Le VAE du guide est neutre sur la
  reward (+0.009) mais **change la racine gardee dans 70 % des prompts** : la selection du
  premier pas informatif tient a des ecarts de reward de l'ordre de l'arrondi d'un decodeur.
  C'est la version la plus directe du constat 3.
- Le modele de coalescence, lu sur la colonne du resampler reellement utilise, tient sur les
  quinze bras : `R1` 1.25 predit contre 1.19 observe (multinomial), `multi` 1.02 contre 1.00,
  `stat0` 1.84 contre 1.75, `vae` et `idx` a 0.05 pres.

## Verdict (23/09, vers 06h, tout rendu)

**La question de depart : les corrections resolvent-elles l'effondrement ?** Par couche.

- *Le mecanisme.* Une seule correction passe le critere « moins de 25 % de runs a une racine »
  : `floor2` (plancher + lambda 2), 5 % a n = 40, 3.0 racines sur 4, diversite pixel 0.29
  contre 0.09. Elle le fait en changeant la cible (lambda 2), et elle rend exactement la
  diversite que cette cible porte (ESS ponderee des racines 2.7 pour un plafond de 2.7). Tout
  ce qui garde lambda = 10 reste entre 62 et 100 % de runs a une racine : `floor` 68 %,
  `fadapt` 30 % mais 1.7 d'ESS ponderee pour un plafond de 1.5 (il rend des racines que sa
  cible ecrase), `thr05` 62 %, `late` 86 % a n = 300, `adapt` 60 %.
- *Pourquoi.* Le nombre de racines finales se predit depuis les poids enregistres et le
  resampler, sans rien savoir du pilotage, a 0.05 pres sur quinze bras (`n_coalescence.py`) :
  c'est la degenerescence des chemins, et l'ESS d'un pas ne la mesure pas. A poids plats le
  peigne est l'identite et le multinomial ne l'est pas (1.58 racines sur 4 apres quatre pas
  plats) : le code publie perd des racines avant toute information.
- *Le prix.* Toute correction qui garde des lignees a lambda = 10 rend `ir_max` au niveau de
  best-of-4 (`floor` -0.09, `fadapt` -0.11, `adapt` -0.10, `stat0` -0.05, `multi` -0.06 contre
  `ctl` ; IC bootstrap a n = 40 couvrant zero un a un, moyenne des bras -0.05 [-0.17, +0.08]),
  et l'`ir` moyen des quatre suit ce qu'on achete (-0.15 a -0.53). `late` seul ne paie pas
  (+0.04 +/- 0.03 a n = 300) et ne repare rien.

**La reference : l'ecart au papier est borne, pas ferme.** La configuration du papier est
celle de ce depot. Le code publie sous cette configuration, sur les 100 prompts, rend 0.554 ;
avec notre generateur 0.807 sur 40 ; son propre best-of-4 vaut 0.824 sur ces 40. Il ne depasse
best-of-4 dans aucune lecture (-0.02 +/- 0.07 apparie par bruit, -0.33 +/- 0.08 sous son chemin
de graine). `smc/` avec ses quatre choix d'implementation rend best-of-4 (`R1`, -0.01) ; avec
les siens, +0.056. Le +0.161 de la table 1 n'est produit par aucune des deux implementations
sur ce materiel. Le diagnostic montre le mecanisme : plancher, poids plats, drift, selection
tardive.

**Ce qui est tenu, rate, ouvert** (predictions de `protocol_sd.md`) :

| prediction | issue |
|---|---|
| `floor2`, `lam2` a 40 : lignees, une racine, prix | tenue |
| `late` : 1.4-1.8 lignees | ratee (1.0-1.14) ; prix nul : tenue |
| coalescence a +/- 0.15, bon ordre | tenue (+/- 0.07) |
| `thr05` : 1.84 racines, 61 %, 1.1 reech. (modele) | tenue (2.00, 62 %, 0.97) ; plan (2.4-2.8) ratee |
| latents : divergence chaotique | ratee (corr > 0.98) ; x_T au bit : tenue |
| `bon4` du 20/09 rejouable | tenue (0.000) |
| `R0` = 0.77 [0.72, 0.82] | ratee par le bas (0.554) |
| `R1` - `ctl` = -0.08 +/- 0.05 | tenue (-0.067) |
| `R1` - `R0` dans +/- 0.03 | ratee (+0.21 ; +0.15 avec notre generateur) |
| bissection : chaque choix sous 0.06 | tenue ; `vae` change la racine gardee dans 70 % des prompts (predit < 30 %) : ratee |
| `rise` : lignees 2.0-2.3, pas au-dessus de `ctl` | tenue (1.9-2.0 ; +0.01) |
| leur baseline libre = la notre en loi | tenue (0.824 / 0.846, 0.295 / 0.288) |

Ouvert : (1) le 0.31 entre les deux chemins de graine du FK des auteurs, reproductible et
inexplique ; (2) la non-reproductibilite inter-session du chemin `smc.models` (pipeline
diffusers rejouable, notre modele non), cause non identifiee ; (3) les constats 3, 5 et 5 bis
suspendus tant que `ctl` et `lam0` n'ont pas ete rejoues dans une meme session ; (4) `floor2`
et `lam0` amputes de 6 prompts par l'incident du rejeu.

## Ce que cela dit des « solutions »

`floor2` resout l'effondrement au sens strict (5 % de
runs a une racine, diversite au plafond de sa cible) en changeant la cible ; a lambda = 10,
rien ne le resout parce que la cible n'accepte que 1.2 a 1.5 particules sur 4 (constat 15),
et les bras qui gardent des racines a ce lambda rendent des particules que la cible ecrase
(`floor`, `fadapt` : ESS ponderee au-dessus du plafond). La suite (R0, R1, bissection,
`thr05`, `rise`) : `docs/protocol_sd.md`, puis ici.
