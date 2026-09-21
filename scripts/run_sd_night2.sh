#!/usr/bin/env bash
# Nuit 2 (docs/paper_plan.md) : le second résultat, la diversité, porté à
# l'échelle de la table, et la référence best-of-4 qu'il lui faut.
#
# T2tA05 est la variante que l'écran donne gagnante sur l'effondrement : rampe
# quadratique, placement tempering, seuil 0.5, 2.05 lignées sur 4 et 38 % de
# l'écart de diversité à best-of-4 fermé, à 5 erreurs-types. À 20 prompts.
# Ici à 100 x 3, contre la référence fk4 de la nuit 1 qui partage ses x_T.
#
# bon4 va dans le fichier de la nuit 1 : même clé de reprise, fk4 y est déjà, et
# les deux références doivent porter les mêmes champs pour être lues ensemble.
#
# Se lance après la nuit 1, pas en parallèle : une seule T4.
#
# Depuis la racine : `nohup scripts/run_sd_night2.sh > /home/onyxia/work/ddpm/sd_night2.log 2>&1 &`
set -uo pipefail
cd "$(dirname "$0")/.."

export HF_HOME=${HF_HOME:-/home/onyxia/work/hf_cache}
PY=/home/onyxia/work/.venvs/sd/bin/python
PROMPTS="--prompts data/imagereward-benchmark-prompts.json"
FULL="$PROMPTS --seeds 2024 2025 2026"

$PY -m pytest tests/test_fk.py -q || { echo "!!! tests/test_fk.py rouge, rien lancé"; exit 1; }

run() {  # fichier de sortie, puis les flags propres au run
  local out=$1; shift
  echo "=== $out  $(date +%H:%M:%S)  $*"
  $PY scripts/run_sd_baseline.py --out "$out" "$@" \
    || echo "!!! $out a échoué, on continue"
}

# ~5.0 h
run results/sd_t2ta05_full.json $FULL --samplers fk4 \
    --fk-lam-schedule quad --fk-lam-placement tempering --fk-threshold 0.5

# ~1.6 h. La référence best-of-4 avec div_clip : sans elle l'écart de diversité
# à fermer n'est mesuré qu'avec le proxy pixel.
run results/sd_ref_fields100.json $PROMPTS --seeds 2024 --samplers bon4

echo "=== terminé $(date +%H:%M:%S)"
