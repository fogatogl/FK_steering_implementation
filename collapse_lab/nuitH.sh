#!/bin/bash
# Session H (29/09, the NVIDIA A2): appendix D of the paper on the collapse, then the released launcher's seeding.
# H0: two prompts of ctl, compared slot by slot with session D; stops the queue if this pod is in
#     another machine group, since H1 pairs with session D's ctl and lam0.
# H1: appendix D of the paper, 200 steps and the last resampling at t = 120, on the first 40 prompts,
#     seed 2024 (free200, d200, st200, pos100; recipeD only with H1B=1).
# H2: the released code before its fix (6726324), their seeding (--seed-once) at 42, 43, 44, FK and
#     their best-of-4, 100 prompts (skipped with H2=0).
# New files only: collapse_lab/out/session_H/, results/sd_authors_once.json.
# Predictions: docs/protocol_sd.md, "Pre-registration of session H".
set -x
V=/home/onyxia/work/.venvs/sd/bin/python
cd "$(dirname "$0")/.."
export HF_HOME=/home/onyxia/work/hf_cache
OUT=collapse_lab/out/session_H
PREFIX=/home/onyxia/work/fkd_ref/prefix_6726324
nvidia-smi --query-gpu=name --format=csv,noheader

$V collapse_lab/probe.py --arms ctl --limit 2 --out $OUT/check_H.json
$V - <<'EOF' || { echo "SESSION H STOPPED AT H0"; exit 1; }
import json
D = {r["prompt_id"]: r for r in json.load(open("collapse_lab/out/session_D/probe_D.json"))["runs"] if r["arm"] == "ctl"}
H = json.load(open("collapse_lab/out/session_H/check_H.json"))["runs"]
bad = [(r["prompt_id"], j) for r in H for j in range(4)
       if abs(r["ir"][j] - D[r["prompt_id"]]["ir"][j]) > 1e-3 or r["root_slots"] != D[r["prompt_id"]]["root_slots"]]
print(f"H0: {8 - len(bad)} of 8 slots as in session D", bad)
raise SystemExit(1 if bad else 0)
EOF

ARMS="free200 d200 st200 pos100"
[ "${H1B:-0}" = 1 ] && ARMS="$ARMS recipeD"
$V collapse_lab/probe.py --arms $ARMS --limit 40 --save-images --out $OUT/probe_H.json
echo "SESSION H1 DONE"

[ "${H2:-1}" = 0 ] && exit 0
for S in 42 43 44; do
    FKD_ROOT=$PREFIX $V collapse_lab/ref/run_authors.py --config paper --seed-once --seed $S --limit 100 \
        --tag _prefix --out results/sd_authors_once.json
    FKD_ROOT=$PREFIX $V collapse_lab/ref/run_authors.py --no-smc --seed-once --seed $S --limit 100 \
        --tag _prefix --out results/sd_authors_once.json
done
echo "SESSION H DONE"
