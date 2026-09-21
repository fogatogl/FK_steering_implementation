#!/usr/bin/env bash
# Enchaîne la nuit 1 puis la nuit 2 sous un seul nohup, et pousse les records
# sur S3 pendant que ça tourne.
#
# Pourquoi : le 2026-09-21 à 08:00:18 UTC Onyxia a suspendu le service en
# scalant le StatefulSet à 0. Le SIGTERM ne va qu'au PID 1, un job en nohup ne
# le voit pas, et le SIGKILL du cgroup tombe 30 s plus tard. La suspension est
# subie et se répètera : une chaîne de ~13 h la croisera.
#
# Ce qui rend la coupure bon marché :
#   - run_sd_baseline.py reprend sur (prompt_id, sampler, seed) et écrit après
#     chaque prompt, désormais par tmp + rename : au pire un prompt perdu ;
#   - le pousseur ci-dessous met results/ sur S3 toutes les 10 min, donc au pire
#     10 min de records restent seulement sur le volume ;
#   - le marqueur CHAINE_EN_VOL n'est lu par personne : il existe pour qu'un
#     humain, ou un personalInit futur, sache au redémarrage qu'une chaîne
#     était en vol.
#
# Reprise après une suspension : relancer exactement la même commande.
#
#   nohup scripts/run_sd_chain.sh > /home/onyxia/work/ddpm/sd_chain.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

MARQUEUR=/home/onyxia/work/ddpm/CHAINE_EN_VOL

echo "=== chaîne démarrée $(date -u +%F_%T) UTC"
date -u +%F_%T > "$MARQUEUR"

# Pousseur S3 : results/ fait 2,6 Mo, le mirror prend une seconde.
( while true; do sleep 600; bash scripts/sync_s3.sh results >/dev/null 2>&1; done ) &
POUSSEUR=$!
trap 'kill $POUSSEUR 2>/dev/null' EXIT

bash scripts/run_sd_night1.sh
echo "=== nuit 1 rendue $(date -u +%F_%T) UTC"
bash scripts/sync_s3.sh results

bash scripts/run_sd_night2.sh
echo "=== nuit 2 rendue $(date -u +%F_%T) UTC"
bash scripts/sync_s3.sh results

rm -f "$MARQUEUR"
echo "=== chaîne terminée $(date -u +%F_%T) UTC"
