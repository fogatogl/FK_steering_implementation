#!/bin/bash
# Deuxieme file : les corrections au meme n, et le levier non mesure du constat 14.
# Meme mecanique que nuit.sh : probe.py reprend depuis probe.json, les paires (prompt, bras)
# deja faites sont sautees. La sortie est ecrite par chemin absolu dans le checkout de
# l'auteur, quel que soit le dossier d'ou ce script est lance.
#
# Ordre : ce qui change le plus la conclusion d'abord. floor2 et lam2 portent la
# recommandation du constat 14 et ne sont qu'a 20 prompts ; late est le seul levier
# du constat 14 sans aucune mesure.
set -x
V=/home/onyxia/work/.venvs/sd/bin/python
cd "$(dirname "$0")/.."
O=/home/onyxia/work/diffusion-models/collapse_lab/out/probe.json
$V collapse_lab/probe.py --arms floor2 lam2 --limit 40 --out $O
$V collapse_lab/probe.py --arms late        --limit 20 --out $O
$V collapse_lab/probe.py --arms late        --limit 40 --out $O
echo "NUIT2 TERMINEE"
