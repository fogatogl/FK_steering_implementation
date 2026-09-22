#!/bin/bash
# Plan de la nuit. Sequentiel, un seul chargement de modele par appel.
# probe.py reprend depuis out/probe.json : les paires (prompt, bras) deja faites
# sont sautees, donc relancer ce fichier ne recalcule rien.
#
# Ordre : lam0 court d'abord. C'est lui qui verifie que la case j de fk et la case j
# de bon4 partagent x_T, hypothese des constats 3, 5 et 5 bis ; si elle est fausse
# autant l'apprendre en quinze minutes. Puis les deux bras qui portent la correction
# (adapt, fadapt), puis le remplissage a 20 et a 40 prompts.
set -x
V=/home/onyxia/work/.venvs/sd/bin/python
cd /home/onyxia/work/diffusion-models
O=collapse_lab/out/probe.json
$V collapse_lab/probe.py --arms lam0                 --limit 10 --out $O
$V collapse_lab/probe.py --arms adapt fadapt         --limit 20 --out $O
$V collapse_lab/probe.py --arms ctl floor            --limit 20 --out $O
$V collapse_lab/probe.py --arms lam0                 --limit 20 --out $O
$V collapse_lab/probe.py --arms adapt fadapt ctl floor --limit 40 --out $O
$V collapse_lab/probe.py --arms lam2 floor2          --limit 20 --out $O
echo "NUIT TERMINEE"
