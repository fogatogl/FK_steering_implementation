#!/usr/bin/env bash
# Troisième volet de l'écran FK sur SD (docs/protocol_sd.md, "Where the ramp's deficit
# is paid") : les mêmes rampes que run_sd_lambda_t.sh, sous placement tempering,
# G_t = pi_t / pi_{t-1}, seules ou croisées avec le seuil 0.5. La comparaison
# T1 vs T1t, à x_T égal, isole l'effet du placement.
#
# Le chemin tempering de smc/fk.py est récent : le lanceur refuse de partir si le test
# télescopique est rouge, plutôt que d'écrire vingt enregistrements d'un potentiel faux.
#
# Depuis la racine : `nohup scripts/run_sd_tempering.sh > /home/onyxia/work/ddpm/sd_tempering.log 2>&1 &`
set -uo pipefail
cd "$(dirname "$0")/.."

export HF_HOME=${HF_HOME:-/home/onyxia/work/hf_cache}
PY=/home/onyxia/work/.venvs/sd/bin/python
OUT=results/sd_variants
mkdir -p "$OUT"
COMMON="--prompts data/imagereward-benchmark-prompts.json --seeds 2024 --limit 20"

$PY -m pytest tests/test_fk.py -q || { echo "!!! tests/test_fk.py rouge, rien lancé"; exit 1; }

run() {  # tag, puis les flags propres à la variante
  local tag=$1; shift
  echo "=== $tag  $(date +%H:%M:%S)  $*"
  $PY scripts/run_sd_baseline.py $COMMON --out "$OUT/$tag.json" "$@" \
    || echo "!!! $tag a échoué, on continue"
}

run T1t    --samplers fk4 --fk-lam-schedule linear --fk-lam-placement tempering
run T2t    --samplers fk4 --fk-lam-schedule quad   --fk-lam-placement tempering
run T1tA05 --samplers fk4 --fk-lam-schedule linear --fk-lam-placement tempering --fk-threshold 0.5
run T2tA05 --samplers fk4 --fk-lam-schedule quad   --fk-lam-placement tempering --fk-threshold 0.5

echo "=== terminé $(date +%H:%M:%S)"
