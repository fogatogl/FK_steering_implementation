#!/bin/bash
# Leur pipeline sans FK, avec notre generateur, cinq prompts : rend-il les tirages libres de
# bon4 au bit pres, comme le pipeline diffusers standard (m_latents) ? Attend nuit2e.sh.
set -x
V=/home/onyxia/work/.venvs/sd/bin/python
cd "$(dirname "$0")/.."
while pgrep -f "collapse_lab/nuit2[de]" > /dev/null; do sleep 60; done
$V collapse_lab/ref/run_authors.py --no-smc --generator --limit 5
echo "NUIT2F TERMINEE"
