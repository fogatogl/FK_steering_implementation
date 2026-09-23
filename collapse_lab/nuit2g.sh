#!/bin/bash
# Leur pipeline sans FK sous LEUR chemin de graine (torch.manual_seed, generator=None),
# 40 prompts : les tirages libres de ce chemin ont-ils la moyenne de bon4 ? R0 (ce chemin)
# rend 0.31 de moins que R0g (notre generateur) sur les memes prompts ; si les tirages libres
# portent le meme ecart, c'est le chemin de graine de leur pipeline, pas leur FK. Attend nuit2f.
set -x
V=/home/onyxia/work/.venvs/sd/bin/python
cd "$(dirname "$0")/.."
while pgrep -f "collapse_lab/nuit2[ef]" > /dev/null; do sleep 60; done
$V collapse_lab/ref/run_authors.py --no-smc --limit 40
echo "NUIT2G TERMINEE"
