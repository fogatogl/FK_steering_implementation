#!/usr/bin/env bash
# onyxia_bootstrap.sh — bring an Onyxia instance back to a working state.
# Idempotent. Usage:  bash scripts/onyxia_bootstrap.sh && source ~/.bashrc
#
# Useful variables:
#   FULL_WEIGHTS=1   fetch ALL S3 checkpoints (2 GB) instead of only
#                    ddpm_last.pt / ft_class3/ddpm_last.pt (257 MB)
#   NO_S3=1          skip the S3 step entirely
set -uo pipefail

# ---- project values ----------------------------------------------------------
GH_OWNER="${GH_OWNER:-fogatogl}"
REPO_NAME="${REPO_NAME:-diffusion-models}"
# The MinIO bucket is named after the *Datalab* user (gfogato), not $USERNAME,
# which is "onyxia" inside the service.
BUCKET="${BUCKET:-gfogato}"
# -----------------------------------------------------------------------------

WORK="/home/onyxia/work"
REPO="${WORK}/${REPO_NAME}"
VENV="${WORK}/.venvs/ddpm"
ART="${WORK}/ddpm"                       # heavy artefacts, outside git, mirror of s3://$BUCKET/ddpm
TOKEN="${GIT_PERSONAL_ACCESS_TOKEN:-}"

echo "== bootstrap: repo=${REPO_NAME} bucket=${BUCKET} =="

# 1. Repo: clone if absent, pull otherwise
if [ -d "${REPO}/.git" ]; then
  echo "-- repo present, pull"
  git -C "${REPO}" pull --ff-only || echo "!! pull failed (uncommitted local work?)"
elif [ -n "${TOKEN}" ]; then
  echo "-- clone"
  git clone "https://${TOKEN}@github.com/${GH_OWNER}/${REPO_NAME}.git" "${REPO}"
else
  echo "!! no local repo and no \$GIT_PERSONAL_ACCESS_TOKEN: add the token to the Datalab account"
fi

# 2. Git identity (~/.gitconfig is not persistent). When the Git tab of the
#    form is filled, Onyxia has already set name and e-mail: only fill gaps.
[ -n "$(git config --global user.name  || true)" ] || git config --global user.name  "${GIT_NAME:-fogatogl}"
[ -n "$(git config --global user.email || true)" ] || git config --global user.email "${GIT_MAIL:-fogatogl05@gmail.com}"
git config --global pull.ff only
echo "-- git: $(git config --global user.name) <$(git config --global user.email)>"

# 3. Venv — --system-site-packages: inherit the image's PyTorch.
#    NEVER reinstall torch here: 2-3 GB and the CUDA pairing breaks.
mkdir -p "${WORK}/.venvs" "${ART}"/{weights,dataset,samples,hf}
if [ ! -x "${VENV}/bin/python" ]; then
  echo "-- creating the venv"
  python3 -m venv --system-site-packages "${VENV}"
else
  echo "-- venv reused"
fi
"${VENV}/bin/pip" install --quiet --upgrade pip
if [ -f "${REPO}/requirements-onyxia.txt" ]; then
  # The particles/numpy conflict warning is expected, see the file.
  "${VENV}/bin/pip" install --quiet -r "${REPO}/requirements-onyxia.txt"
  "${VENV}/bin/pip" install --quiet -e "${REPO}"
else
  "${VENV}/bin/pip" install --quiet ipykernel matplotlib tqdm pytest
fi

# 4. Jupyter kernel (~/.local is not persistent)
"${VENV}/bin/python" -m ipykernel install \
    --prefix "${HOME}/.local" --name ddpm --display-name "Python (ddpm)" >/dev/null

# 5. Venv auto-activation + variables (~/.bashrc is not persistent)
if ! grep -q "${VENV}/bin/activate" "${HOME}/.bashrc" 2>/dev/null; then
  cat >> "${HOME}/.bashrc" <<BASHRC

# --- project ${REPO_NAME} (rewritten each session: ~/.bashrc is not persistent)
export BUCKET=${BUCKET}
export REPO=${REPO}
# HuggingFace cache on the persistent volume: ~/.cache dies with the service
export HF_HOME=${ART}/hf
source ${VENV}/bin/activate
BASHRC
fi

# 6. Heavy artefacts from S3 (never in git).
#    WARNING: "mc ls s3/" without a bucket hangs (the stsonly policy does not
#    allow ListBuckets) — always target s3/<bucket>/... explicitly.
if [ -n "${NO_S3:-}" ]; then
  echo "-- S3 skipped (NO_S3)"
elif ! command -v mc >/dev/null 2>&1; then
  echo "-- mc missing, S3 restore skipped"
elif [ -n "${FULL_WEIGHTS:-}" ]; then
  echo "-- full weights mirror (2 GB)"
  mc mirror --overwrite "s3/${BUCKET}/ddpm/weights" "${ART}/weights" \
    || echo "!! weights not fetched (S3 token expired?)"
else
  echo "-- weights: latest checkpoints only (FULL_WEIGHTS=1 for the 2 GB)"
  for f in weights/ddpm_last.pt weights/ft_class3/ddpm_last.pt samples/samples.png; do
    [ -f "${ART}/${f}" ] || mc cp "s3/${BUCKET}/ddpm/${f}" "${ART}/${f}" \
      || echo "!! ${f} not fetched (S3 token expired?)"
  done
fi

# 7. Check
"${VENV}/bin/python" - <<'PY'
import torch
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
print(f"   torch {torch.__version__} | {gpu}")
PY

echo "== done. Now run:  source ~/.bashrc =="
