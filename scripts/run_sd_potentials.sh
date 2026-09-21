#!/usr/bin/env bash
# Les deux runs qui tranchent la sous-performance de max (docs/max_potential.md).
#
# `difference` n'a jamais tourne sur SD : c'est ce que l'annexe du papier dit avoir
# produit la table 1. `statistic` est la forme du papier et du code publie. Les deux
# sont apparies contre le fk4 de sd_baseline.json, donc memes prompts, meme seed,
# memes x_T : le fichier de prompts fait partie de l'identite de l'experience, son
# index nourrit la graine, et le defaut de --prompts n'est PAS celui de l'ecran.
#
# Attend que la T4 se libere : la nuit 1 finit encore ref_fields100.
#
#   nohup scripts/run_sd_potentials.sh > /home/onyxia/work/ddpm/sd_potentials.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

export HF_HOME=${HF_HOME:-/home/onyxia/work/hf_cache}
PY=/home/onyxia/work/.venvs/sd/bin/python
OUT=results/sd_variants
mkdir -p "$OUT"
COMMON="--prompts data/imagereward-benchmark-prompts.json --seeds 2024 --limit 20 --samplers fk4"

while pgrep -f "scripts/run_sd_baseline.py" >/dev/null; do
  echo "T4 occupee, attente $(date -u +%H:%M:%S)"
  sleep 120
done
echo "=== T4 libre $(date -u +%F_%T) UTC"

$PY -m pytest tests/test_fk.py -q || { echo "!!! tests/test_fk.py rouge, rien lance"; exit 1; }

run() {  # tag, puis les flags propres au run
  local tag=$1; shift
  echo "=== $tag  $(date -u +%H:%M:%S)  $*"
  $PY scripts/run_sd_baseline.py $COMMON --out "$OUT/$tag.json" "$@" \
    || echo "!!! $tag a echoue, on continue"
}

run fk4_diff --fk-potential difference
run fk4_stat --fk-potential-form statistic

echo "=== termine $(date -u +%F_%T) UTC"
