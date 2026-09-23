#!/bin/bash
# Leur pipeline sans FK avec notre generateur, etendu a 40 prompts : il ne rend pas bon4 case
# par case (nuit2f, ecarts jusqu'a 2.7), la question devient s'il a la meme loi. Attend nuit2g.
set -x
V=/home/onyxia/work/.venvs/sd/bin/python
cd "$(dirname "$0")/.."
while pgrep -f "collapse_lab/nuit2g" > /dev/null; do sleep 60; done
$V collapse_lab/ref/run_authors.py --no-smc --generator --limit 40
echo "NUIT2H TERMINEE"
