#!/bin/bash
# R0 : le code des auteurs, configuration du papier, 100 prompts, une graine. Reprend depuis
# results/sd_authors_R0.json. `--config defaults` pour les defauts du script publie (5-30-5, diff).
set -x
V=/home/onyxia/work/.venvs/sd/bin/python
cd "$(dirname "$0")/../.."
$V collapse_lab/ref/run_authors.py --config paper --limit "${1:-100}"
