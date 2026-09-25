#!/bin/bash
# Back up / restore the heavy artefacts on the personal MinIO bucket.
#
#   ./sync_s3.sh check   -> are the credentials alive? (expiry date, one listing)
#   ./sync_s3.sh push    -> dataset + weights + samples + fid + hf (Hub models), then lab and repo
#   ./sync_s3.sh results -> send results/ alone (2.6 MB, seconds): the run records
#   ./sync_s3.sh lab     -> collapse_lab/out: probe JSONs, logs, images and thumbnails of every session
#   ./sync_s3.sh repo    -> the working tree of the repository, .git included: what is not pushed to GitHub
#   ./sync_s3.sh pull    -> fetch them back
#   ./sync_s3.sh status  -> what the bucket holds
#
# The bucket is named after the *Datalab* user (gfogato). $USERNAME is "onyxia"
# inside the service: do not use it as a default. The access token lives 7 days
# and is written into the service's environment when the service is created: a
# pod restart keeps the old one. Fresh credentials come from the Onyxia page
# "My account > Storage access" (Mon compte > Connexion au stockage); paste its
# export lines into $S3_ENV_FILE (default /home/onyxia/work/.s3_env, outside the
# repository, chmod 600) and this script uses them instead of the expired ones.
set -uo pipefail

S3_ENV_FILE="${S3_ENV_FILE:-/home/onyxia/work/.s3_env}"
if [ -f "${S3_ENV_FILE}" ]; then
  # shellcheck disable=SC1090
  . "${S3_ENV_FILE}"
  # mc reads its alias from MC_HOST_s3; rebuild it from the fresh AWS variables
  export MC_HOST_s3="https://${AWS_ACCESS_KEY_ID}:${AWS_SECRET_ACCESS_KEY}:${AWS_SESSION_TOKEN}@${AWS_S3_ENDPOINT:-minio.lab.sspcloud.fr}"
fi

BUCKET="${BUCKET:-gfogato}"
PROJ="${PROJ:-/home/onyxia/work/ddpm}"
# The JSON records live in the git repo, not in PROJ. They are small and they
# are what a run costs hours to rebuild: the PVC is per-service, so a service
# relaunched under a new id would not see them again.
REPO="${REPO:-/home/onyxia/work/diffusion-models}"
REMOTE="s3/${BUCKET}/ddpm"
ACTION="${1:-push}"

command -v mc >/dev/null || { echo "mc not found"; exit 1; }

expiry() {
  python3 - <<'PY'
import base64, datetime as dt, json, os
t = os.environ.get("AWS_SESSION_TOKEN", "")
if t.count(".") != 2:
    print("no session token in the environment")
else:
    p = t.split(".")[1]; p += "=" * (-len(p) % 4)
    exp = dt.datetime.fromtimestamp(json.loads(base64.urlsafe_b64decode(p))["exp"], dt.timezone.utc)
    left = exp - dt.datetime.now(dt.timezone.utc)
    print(f"token expires {exp:%Y-%m-%d %H:%M} UTC ({'EXPIRED' if left.total_seconds() < 0 else f'{left.days} d {left.seconds // 3600} h left'})")
PY
}

lab()  { mc mirror --overwrite --exclude "*.tmp" "${REPO}/collapse_lab/out" "${REMOTE}/collapse_lab_out"; }
# .git/config is left out: it can carry a token in the remote URL
repo() { mc mirror --overwrite --exclude "*.tmp" --exclude "*/__pycache__/*" --exclude "*.pyc" --exclude ".git/config" \
            --exclude "samples/*" --exclude "collapse_lab/out/*" "${REPO}" "${REMOTE}/repo"; }

case "${ACTION}" in
  check)
    expiry
    mc ls "${REMOTE}/" >/dev/null && echo "listing ${REMOTE}: ok" || { echo "listing ${REMOTE}: FAILED"; exit 1; }
    ;;
  push)
    echo "-> ${REMOTE}"
    mc mirror --overwrite "${PROJ}/weights"  "${REMOTE}/weights"
    mc mirror --overwrite "${PROJ}/samples"  "${REMOTE}/samples"
    mc mirror --overwrite "${PROJ}/dataset"  "${REMOTE}/dataset"
    mc mirror --overwrite "${PROJ}/fid"      "${REMOTE}/fid"
    mc mirror --overwrite "${PROJ}/hf"       "${REMOTE}/hf"
    mc mirror --overwrite --exclude "*.tmp" "${REPO}/results" "${REMOTE}/results"
    lab
    repo
    ;;
  results)
    echo "-> ${REMOTE}/results"
    mc mirror --overwrite --exclude "*.tmp" "${REPO}/results" "${REMOTE}/results"
    ;;
  lab)
    echo "-> ${REMOTE}/collapse_lab_out"
    lab
    ;;
  repo)
    echo "-> ${REMOTE}/repo"
    repo
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
    # the lab outputs and the working tree come back only on request: they would overwrite local work
    echo "lab and repo: mc mirror ${REMOTE}/collapse_lab_out <dir>, mc mirror ${REMOTE}/repo <dir>"
    ;;
  status)
    # Always name the bucket: a bare "mc ls s3/" hangs, the stsonly policy
    # does not allow ListBuckets.
    expiry
    mc ls -r "${REMOTE}" 2>/dev/null | tail -20 || echo "nothing in ${REMOTE}"
    mc du "${REMOTE}" 2>/dev/null
    ;;
  *)
    echo "usage: $0 {check|push|results|lab|repo|pull|status}"; exit 1;;
esac
