#!/usr/bin/env bash
# Push the run records, the lab's outputs and the working tree to S3 every 10 minutes, in the
# background (scripts/sync_s3.sh results, lab, repo; mc mirror only sends what changed).
# The credentials come from /home/onyxia/work/.s3_env when it exists (see sync_s3.sh).
#
#   nohup scripts/s3_pusher.sh > /home/onyxia/work/ddpm/s3_pusher.log 2>&1 &
#   pkill -f scripts/s3_pusher.sh      # to stop it
set -uo pipefail
cd "$(dirname "$0")/.."

echo "=== S3 pusher started $(date -u +%F_%T) UTC"
while true; do
  for what in results lab repo; do
    bash scripts/sync_s3.sh "$what" >/dev/null 2>&1 \
      && echo "push $what $(date -u +%H:%M:%S)" || echo "!!! push $what FAILED $(date -u +%H:%M:%S)"
  done
  sleep 600
done
