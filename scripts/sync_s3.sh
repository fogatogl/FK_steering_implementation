#!/bin/bash
# Back up / restore the heavy artefacts on the personal MinIO bucket.
#
#   ./sync_s3.sh push    -> send dataset + weights + samples + fid + hf (Hub models) to S3
#   ./sync_s3.sh results -> send results/ alone (2.6 MB, seconds): the run records
#   ./sync_s3.sh pull    -> fetch them back
#   ./sync_s3.sh status  -> what the bucket holds
#
# The bucket is named after the *Datalab* user (gfogato). $USERNAME is "onyxia"
# inside the service: do not use it as a default. The access token lives 7 days
# and is regenerated on launch: a "red" service in "My services" means the token
# expired on the service side -> relaunch it (the bucket itself is untouched).
set -uo pipefail

BUCKET="${BUCKET:-gfogato}"
PROJ="${PROJ:-/home/onyxia/work/ddpm}"
# The JSON records live in the git repo, not in PROJ. They are small and they
# are what a run costs hours to rebuild: the PVC is per-service, so a service
# relaunched under a new id would not see them again.
REPO="${REPO:-/home/onyxia/work/diffusion-models}"
REMOTE="s3/${BUCKET}/ddpm"
ACTION="${1:-push}"

command -v mc >/dev/null || { echo "mc not found"; exit 1; }

case "${ACTION}" in
  push)
    echo "-> ${REMOTE}"
    mc mirror --overwrite "${PROJ}/weights"  "${REMOTE}/weights"
    mc mirror --overwrite "${PROJ}/samples"  "${REMOTE}/samples"
    mc mirror --overwrite "${PROJ}/dataset"  "${REMOTE}/dataset"
    mc mirror --overwrite "${PROJ}/fid"      "${REMOTE}/fid"
    mc mirror --overwrite "${PROJ}/hf"       "${REMOTE}/hf"
    mc mirror --overwrite --exclude "*.tmp" "${REPO}/results" "${REMOTE}/results"
    ;;

  results)
    echo "-> ${REMOTE}/results"
    mc mirror --overwrite --exclude "*.tmp" "${REPO}/results" "${REMOTE}/results"
    ;;
  pull)
    echo "<- ${REMOTE}"
    mkdir -p "${PROJ}"/{weights,samples,dataset,fid,hf}
    mc mirror --overwrite "${REMOTE}/weights" "${PROJ}/weights"
    mc mirror --overwrite "${REMOTE}/samples" "${PROJ}/samples"
    mc mirror --overwrite "${REMOTE}/dataset" "${PROJ}/dataset"
    mc mirror --overwrite "${REMOTE}/fid"     "${PROJ}/fid"
    mc mirror --overwrite "${REMOTE}/hf"      "${PROJ}/hf"
    mc mirror --overwrite "${REMOTE}/results" "${REPO}/results"
    ;;
  status)
    # Always name the bucket: a bare "mc ls s3/" hangs, the stsonly policy
    # does not allow ListBuckets.
    mc ls -r "${REMOTE}" 2>/dev/null || echo "nothing in ${REMOTE}"
    mc du "${REMOTE}" 2>/dev/null
    ;;
  *)
    echo "usage: $0 {push|pull|results|status}"; exit 1;;
esac
