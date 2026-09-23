#!/bin/bash
# R0g24 : le code des auteurs, configuration du papier, notre generateur a la graine 2024000 + i,
# 40 prompts : memes x_T et meme bruit DDIM que bon4 / ctl / R1. Prediction : protocol_sd.md,
# correction du 23/09 (~06h20). HF_HOME pointe le cache de l'auteur, pas ~/.cache.
set -x
export HF_HOME=/home/onyxia/work/hf_cache
V=/home/onyxia/work/.venvs/sd/bin/python
cd "$(dirname "$0")/.."
$V collapse_lab/ref/run_authors.py --config paper --generator --seed 2024 --limit 40 --out /home/onyxia/work/diffusion-models/results/sd_authors_R0.json
echo "NUIT2I TERMINEE"
