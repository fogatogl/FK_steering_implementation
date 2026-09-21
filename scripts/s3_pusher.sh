#!/usr/bin/env bash
# Pousse results/ sur S3 toutes les 10 minutes, en tache de fond.
#
# Le pousseur vivait dans run_sd_chain.sh et mourait avec lui (trap EXIT). Ici il
# est autonome : on le lance quand une campagne tourne sans la chaine, et on
# l'arrete avec `pkill -f scripts/s3_pusher.sh`.
#
#   nohup scripts/s3_pusher.sh > /home/onyxia/work/ddpm/s3_pusher.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

echo "=== pousseur S3 demarre $(date -u +%F_%T) UTC"
while true; do
  bash scripts/sync_s3.sh results >/dev/null 2>&1 \
    && echo "push $(date -u +%H:%M:%S)" || echo "!!! push rate $(date -u +%H:%M:%S)"
  sleep 600
done
