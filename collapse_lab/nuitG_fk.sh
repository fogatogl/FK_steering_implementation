#!/bin/bash
# Session G, amendment (26/09, the NVIDIA A2): FK of this repository at seeds 2025 and 2026, so that
# section 7's column "against FK" reads on three seeds. Waits for session G. One process per seed, so
# a seed is complete before the next starts (run_sd_baseline.py loops seeds inside prompts).
# Predictions: docs/protocol_sd.md, "Pre-registration of session G", its amendment.
set -x
V=/home/onyxia/work/.venvs/sd/bin/python
cd "$(dirname "$0")/.."
export HF_HOME=/home/onyxia/work/hf_cache
until grep -q "SESSION G DONE" collapse_lab/out/session_G/nuitG.log; do sleep 60; done
nvidia-smi --query-gpu=name --format=csv,noheader
for S in 2025 2026; do
    $V scripts/run_sd_baseline.py --prompts data/imagereward-benchmark-prompts.json --samplers fk4 --seeds $S \
        --out results/sd_seeds_fk4.json
done
echo "SESSION G FK DONE"
