#!/usr/bin/env bash
# Le second volet de l'écran FK sur SD (docs/protocol_sd.md) : les deux rampes
# lambda_t, T1 linéaire et T2 quadratique, lambda_T = 10 dans les deux cas.
#
# Lanceur séparé de run_sd_variants.sh pour ne pas relancer les sept variantes
# déjà écrites dans results/sd_variants/ : la reprise est par fichier, pas par
# tag, et un run() de plus dans l'autre script rejouerait tout.
#
# Mêmes prompts, même seed, donc mêmes x_T que sd_baseline.json et que les sept
# autres : compare_sd_variants.py globe le répertoire et appariera T1 et T2 sans
# rien de plus.
#
# Depuis la racine : `nohup scripts/run_sd_lambda_t.sh > /home/onyxia/work/ddpm/sd_lambda_t.log 2>&1 &`
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

run T1 --samplers fk4 --fk-lam-schedule linear
run T2 --samplers fk4 --fk-lam-schedule quad
# Au seuil 1.0 on rééchantillonne à chaque point du calendrier quoi qu'il arrive : quatre
# tirages systématiques sur k=4 finissent sur une seule racine, rampe ou pas (le smoke sur
# le prompt 0 : ESS 2.07 à t=80 et pourtant n_lineages=1). La rampe ne peut protéger la
# diversité que si le seuil laisse passer les pas où elle a maintenu l'ESS : d'où le
# croisement avec A05.
run T1A05 --samplers fk4 --fk-lam-schedule linear --fk-threshold 0.5
run T2A05 --samplers fk4 --fk-lam-schedule quad --fk-threshold 0.5

echo "=== terminé $(date +%H:%M:%S)"
