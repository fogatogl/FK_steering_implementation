#!/usr/bin/env bash
# Nuit 1 (plan editorial du 21/09, "Sixteen days") : ce qui ferme l'ablation du
# calendrier, puis le run confirmatoire qui entre dans la table.
#
# Ordre court -> long : une nuit coupée en route garde l'ablation, qui est ce
# qui manque au billet, et perd le confirmatoire, qui peut se relancer.
#
# S80 et D10 sont les deux variantes de l'écran jamais lancées ; elles restent à
# 20 prompts seed 2024, donc mêmes x_T que sd_baseline.json et que les quinze
# autres, et compare_sd_variants.py les apparie sans rien de plus.
#
# S60full et la référence fk4 sont à 100 prompts, donc dans results/ et pas dans
# results/sd_variants/ : compare_sd_variants.py globe ce répertoire et lirait un
# fichier de 300 enregistrements contre une référence de 20.
#
# La référence fk4 est régénérée pour ses nouveaux champs root_slots et
# div_clip ; à seeds égales elle redonne le même ir_max (vérifié sur div20,
# écart maximum 0.0), donc elle remplace l'ancienne sans la contredire.
#
# Depuis la racine : `nohup scripts/run_sd_night1.sh > /home/onyxia/work/ddpm/sd_night1.log 2>&1 &`
set -uo pipefail
cd "$(dirname "$0")/.."

export HF_HOME=${HF_HOME:-/home/onyxia/work/hf_cache}
PY=/home/onyxia/work/.venvs/sd/bin/python
PROMPTS="--prompts data/imagereward-benchmark-prompts.json"
SCREEN="$PROMPTS --seeds 2024 --limit 20"
FULL="$PROMPTS --seeds 2024 2025 2026"
mkdir -p results/sd_variants

$PY -m pytest tests/test_fk.py -q || { echo "!!! tests/test_fk.py rouge, rien lancé"; exit 1; }

run() {  # fichier de sortie, puis les flags propres au run
  local out=$1; shift
  echo "=== $out  $(date +%H:%M:%S)  $*"
  $PY scripts/run_sd_baseline.py --out "$out" "$@" \
    || echo "!!! $out a échoué, on continue"
}

# ~21 min. Une seule sélection à t = 80 puis quatre continuations : si S80 tient
# l'ir_max de fk4, le gain à k = 4 est une sélection précoce et les
# rééchantillonnages suivants ne portent rien.
run results/sd_variants/S80.json $SCREEN --samplers fk4 --fk-schedule 0 80

# ~25 min (cinq décodages de plus par particule). La dernière variante costée de
# l'écran jamais lancée : ferme la liste des suspects du max courant.
run results/sd_variants/D10.json $SCREEN --samplers fk4 \
    --fk-schedule 0 10 20 30 40 50 60 70 80 90

# ~5.0 h. Le confirmatoire : la variante que l'écran a classée première, sur les
# 100 prompts x 3 seeds, erreur-type appariée attendue ~0.026.
run results/sd_s60_full.json $FULL --samplers fk4 --fk-schedule 0 20 40 60

# ~1.7 h. fk4 au réglage du papier avec root_slots et div_clip, à 100 prompts :
# sans elle le taux de mauvaise racine et les lignées n'existent qu'à 20.
run results/sd_ref_fields100.json $PROMPTS --seeds 2024 --samplers fk4

echo "=== terminé $(date +%H:%M:%S)"
