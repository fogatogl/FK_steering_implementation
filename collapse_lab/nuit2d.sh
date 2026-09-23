#!/bin/bash
# Apres la nuit 2 : (1) le test des latents complete par ImageReward sur les finales des deux
# samplers et la valeur enregistree de bon4 ; (2) R0g, le code des auteurs avec notre
# generateur (memes x_T que bon4 / ctl / R1), 40 prompts. Predictions : docs/protocol_sd.md,
# ajouts du 22/09 23h30 et 23h45.
set -x
V=/home/onyxia/work/.venvs/sd/bin/python
cd "$(dirname "$0")/.."
while pgrep -f "collapse_lab/nuit2c" > /dev/null; do sleep 60; done
$V collapse_lab/m_latents.py --i 0
$V collapse_lab/m_latents.py --i 1
$V collapse_lab/ref/run_authors.py --config paper --generator --limit 40
echo "NUIT2D TERMINEE"
