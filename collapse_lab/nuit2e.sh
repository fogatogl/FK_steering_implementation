#!/bin/bash
# B1 rejoue avec --redo : les paires (prompt, bras) des six prompts existaient deja sans
# images ni ancetres, la reprise les avait sautees. Memes seeds, memes chiffres, plus les PNG
# et la matrice des ancetres pour F1 et F2. Attend la fin de nuit2d.sh.
set -x
V=/home/onyxia/work/.venvs/sd/bin/python
cd "$(dirname "$0")/.."
while pgrep -f "collapse_lab/nuit2d" > /dev/null; do sleep 60; done
O=/home/onyxia/work/diffusion-models/collapse_lab/out/probe.json
IDS=$($V collapse_lab/q_image_grid.py --choose)
$V collapse_lab/probe.py --arms lam0 ctl floor2 R1 --limit 100 --prompt-ids $IDS --save-images --redo --out $O
echo "NUIT2E TERMINEE"
