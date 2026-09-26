#!/bin/bash
# Session G (26/09, the NVIDIA A2): seeds 2025 and 2026 of the released code against best-of-4, so
# that the rows of section 7 read on three seeds instead of one. Per seed, in one block: bon4 (this
# repository, same flags as sd_baseline.json), the released code (9413005), then its commit before the
# max-potential fix (6726324), paper configuration, global seed, 100 prompts. A seed is complete before
# the next starts, so an interrupted night still leaves one same-machine triple.
# New files only: results/sd_seeds_bon4.json, results/sd_seeds_authors.json.
# Predictions: docs/protocol_sd.md, "Pre-registration of session G".
set -x
V=/home/onyxia/work/.venvs/sd/bin/python
cd "$(dirname "$0")/.."
export HF_HOME=/home/onyxia/work/hf_cache
nvidia-smi --query-gpu=name --format=csv,noheader
for S in 2025 2026; do
    $V scripts/run_sd_baseline.py --prompts data/imagereward-benchmark-prompts.json --samplers bon4 --seeds $S \
        --out results/sd_seeds_bon4.json
    $V collapse_lab/ref/run_authors.py --config paper --seed $S --limit 100 --out results/sd_seeds_authors.json
    FKD_ROOT=/home/onyxia/work/fkd_ref/prefix_6726324 $V collapse_lab/ref/run_authors.py --config paper \
        --seed $S --limit 100 --tag _prefix --out results/sd_seeds_authors.json
done
echo "SESSION G DONE"
