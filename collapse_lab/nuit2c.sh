#!/bin/bash
# Suite de la nuit 2 apres l'arret du parent a 22h : R0 finit seul, puis, le GPU libre,
# le test des latents (constat 11, corrige), le diagnostic du code des auteurs sur deux
# prompts (le guide voit-il quelque chose ?), et la reprise de nuit2.sh (B1, bissection,
# thr05, rise ; R1 et R0 sont sautes par la reprise).
set -x
V=/home/onyxia/work/.venvs/sd/bin/python
cd "$(dirname "$0")/.."
while pgrep -f "ref/run_authors.py" > /dev/null; do sleep 60; done
$V collapse_lab/m_latents.py --i 0
$V collapse_lab/m_latents.py --i 1
$V collapse_lab/ref/diag_authors.py --i 0
$V collapse_lab/ref/diag_authors.py --i 2
echo "DIAG TERMINE"
bash collapse_lab/nuit2.sh
