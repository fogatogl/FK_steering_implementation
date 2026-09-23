#!/bin/bash
# Nuit 3 (23/09) : session C, un seul processus pour ctl, lam0 et floor2 a 100 prompts (appariement
# par x_T valide dans la session), puis R0 a la graine 2024 sous leur chemin de graine.
# Predictions : docs/protocol_sd.md, "Pre-registration of session C". Attend la fin du trace.
set -x
export HF_HOME=/home/onyxia/work/hf_cache
V=/home/onyxia/work/.venvs/sd/bin/python
cd "$(dirname "$0")/.."
while pgrep -f "ref/diag_authors.py" > /dev/null; do sleep 30; done
O=/home/onyxia/work/diffusion-models/collapse_lab/out/probe_C.json
$V collapse_lab/probe.py --arms ctl lam0 floor2 --limit 100 --redo --out $O
$V collapse_lab/ref/run_authors.py --config paper --seed 2024 --limit 40 --out /home/onyxia/work/diffusion-models/results/sd_authors_R0.json
echo "NUIT3 TERMINEE"
