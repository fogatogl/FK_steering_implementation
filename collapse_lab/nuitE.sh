#!/bin/bash
# Session E (25/09, the NVIDIA T4): (1) the machine check, ctl and lam0 of probe.py on the first two
# prompts against session C; (2) the released code at seed 2024 completed to the 100 prompts, global
# seed then generator, into results/sd_authors_R0_100.json (the frozen sd_authors_R0.json is untouched).
# Predictions: docs/protocol_sd.md, "Pre-registration of session E".
set -x
V=/home/onyxia/work/.venvs/sd/bin/python
cd "$(dirname "$0")/.."
export HF_HOME=/home/onyxia/work/hf_cache
nvidia-smi --query-gpu=name --format=csv,noheader
$V collapse_lab/probe.py --arms ctl lam0 --limit 100 --prompt-ids 005695-0057 005784-0093 \
    --out /home/onyxia/work/diffusion-models/collapse_lab/out/session_E/t4_check.json
$V collapse_lab/ref/run_authors.py --config paper --seed 2024 --limit 100 --out results/sd_authors_R0_100.json
$V collapse_lab/ref/run_authors.py --config paper --seed 2024 --limit 100 --generator --out results/sd_authors_R0_100.json
echo "SESSION E DONE"
