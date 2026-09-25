#!/bin/bash
# After session D: HPS v2.1 of its 1200 finals (collapse_lab/w_hps_finals.py), for F1.
# Waits for the end line of nuitD.log, so it never shares the GPU with the session.
set -x
export HF_HOME=/home/onyxia/work/hf_cache
V=/home/onyxia/work/.venvs/sd/bin/python
cd "$(dirname "$0")/.."
L=collapse_lab/out/session_D/nuitD.log
until grep -q "NUIT TERMINEE" $L; do sleep 60; done
$V collapse_lab/w_hps_finals.py collapse_lab/out/session_D/probe_D.json
echo "HPS TERMINE"
