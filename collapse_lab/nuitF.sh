#!/bin/bash
# Session F (25/09, the NVIDIA T4): the released code at its commit before "address max potential
# bug" (6726324), paper configuration, seed 2024, global seed, 100 prompts. Waits for session E.
# Predictions: docs/protocol_sd.md, "Pre-registration of session F".
set -x
V=/home/onyxia/work/.venvs/sd/bin/python
cd "$(dirname "$0")/.."
export HF_HOME=/home/onyxia/work/hf_cache
until grep -q "SESSION E DONE" collapse_lab/out/session_E/nuitE.log; do sleep 60; done
FKD_ROOT=/home/onyxia/work/fkd_ref/prefix_6726324 $V collapse_lab/ref/run_authors.py --config paper \
    --seed 2024 --limit 100 --tag _prefix --out results/sd_authors_prefix.json
echo "SESSION F DONE"
