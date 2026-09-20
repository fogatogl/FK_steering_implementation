#!/usr/bin/env bash
# L'écran des variantes FK sur SD (docs/protocol_sd.md, "Follow-up") : les 20 premiers
# prompts, seed 2024, un JSON par variante. Même index de prompt et même seed que
# results/sd_baseline.json, donc mêmes x_T : la comparaison est appariée.
#
# Un fichier par variante n'est pas un confort : la clé de reprise est
# (prompt_id, sampler, seed), une variante écrite dans le fichier principal
# serait sautée comme déjà faite.
#
# Depuis la racine : `nohup scripts/run_sd_variants.sh > /home/onyxia/work/ddpm/sd_variants.log 2>&1 &`
set -uo pipefail
cd "$(dirname "$0")/.."

export HF_HOME=${HF_HOME:-/home/onyxia/work/hf_cache}
PY=/home/onyxia/work/.venvs/sd/bin/python
OUT=results/sd_variants
mkdir -p "$OUT"
COMMON="--prompts data/imagereward-benchmark-prompts.json --seeds 2024 --limit 20"

run() {  # tag, puis les flags propres à la variante
  local tag=$1; shift
  echo "=== $tag  $(date +%H:%M:%S)  $*"
  $PY scripts/run_sd_baseline.py $COMMON --out "$OUT/$tag.json" "$@" \
    || echo "!!! $tag a échoué, on continue"
}

run S60 --samplers fk4 --fk-schedule 0 20 40 60
run S40 --samplers fk4 --fk-schedule 0 20 40
run L2  --samplers fk4 --lam 2
run L5  --samplers fk4 --lam 5
run L20 --samplers fk4 --lam 20
run A05 --samplers fk4 --fk-threshold 0.5
run K8  --samplers bon8 fk8

echo "=== terminé $(date +%H:%M:%S)"
