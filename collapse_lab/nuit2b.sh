#!/bin/bash
# Le test des latents (constat 11) a plante en tete de nuit2.sh sur un type de device ;
# corrige, il repasse ici des que nuit2.sh a rendu le GPU. Deux prompts, ~3 min.
set -x
V=/home/onyxia/work/.venvs/sd/bin/python
cd "$(dirname "$0")/.."
while pgrep -f "bash collapse_lab/nuit2.sh" > /dev/null; do sleep 60; done
$V collapse_lab/m_latents.py --i 0
$V collapse_lab/m_latents.py --i 1
echo "LATENTS TERMINES"
