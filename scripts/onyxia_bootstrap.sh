#!/usr/bin/env bash
# onyxia_bootstrap.sh — remet une instance Onyxia en état de marche.
# Idempotent. Usage :  bash scripts/onyxia_bootstrap.sh && source ~/.bashrc
set -uo pipefail

# ---- à adapter une fois ------------------------------------------------------
GH_OWNER="${GH_OWNER:-gfogato}"
REPO_NAME="${REPO_NAME:-fk-smc}"
GIT_NAME="${GIT_NAME:-Giulio Leone Fogato}"
GIT_MAIL="${GIT_MAIL:-adresse@du-compte-github}"
# -----------------------------------------------------------------------------

WORK="/home/onyxia/work"
REPO="${WORK}/${REPO_NAME}"
VENV="${WORK}/.venvs/ddpm"
ART="${WORK}/ddpm"                       # artefacts lourds, hors git
BUCKET="${BUCKET:-${USERNAME:-$(whoami)}}"
TOKEN="${GIT_PERSONAL_ACCESS_TOKEN:-}"

echo "== bootstrap : repo=${REPO_NAME} bucket=${BUCKET} =="

# 1. Dépôt : clone si absent, pull sinon
if [ -d "${REPO}/.git" ]; then
  echo "-- dépôt présent, pull"
  git -C "${REPO}" pull --ff-only || echo "!! pull impossible (travail local non commité ?)"
elif [ -n "${TOKEN}" ]; then
  echo "-- clone"
  git clone "https://${TOKEN}@github.com/${GH_OWNER}/${REPO_NAME}.git" "${REPO}"
else
  echo "!! ni dépôt local ni \$GIT_PERSONAL_ACCESS_TOKEN : ajoute le jeton au compte Datalab"
fi

# 2. Identité git (~/.gitconfig n'est pas persistant)
git config --global user.name  "${GIT_NAME}"
git config --global user.email "${GIT_MAIL}"
git config --global pull.ff only

# 3. Venv — --system-site-packages : on hérite du PyTorch GPU de l'image.
#    NE JAMAIS réinstaller torch ici : 2-3 Go et l'appairage CUDA casse.
mkdir -p "${WORK}/.venvs" "${ART}"/{weights,dataset,samples}
if [ ! -x "${VENV}/bin/python" ]; then
  echo "-- création du venv"
  python3 -m venv --system-site-packages "${VENV}"
else
  echo "-- venv réutilisé"
fi
"${VENV}/bin/pip" install --quiet --upgrade pip
if [ -f "${REPO}/requirements-onyxia.txt" ]; then
  "${VENV}/bin/pip" install --quiet -r "${REPO}/requirements-onyxia.txt"
else
  "${VENV}/bin/pip" install --quiet ipykernel matplotlib tqdm pytest
fi

# 4. Kernel Jupyter (~/.local n'est pas persistant)
"${VENV}/bin/python" -m ipykernel install \
    --prefix "${HOME}/.local" --name ddpm --display-name "Python (ddpm GPU)" >/dev/null

# 5. Activation automatique du venv (~/.bashrc n'est pas persistant)
grep -q "${VENV}/bin/activate" "${HOME}/.bashrc" 2>/dev/null \
  || echo "source ${VENV}/bin/activate" >> "${HOME}/.bashrc"

# 6. Artefacts lourds depuis S3 (jamais dans git)
if command -v mc >/dev/null 2>&1; then
  mc mirror --overwrite "s3/${BUCKET}/ddpm/weights"  "${ART}/weights"  2>/dev/null \
    || echo "-- pas de poids sur S3 (ou jeton expiré)"
  mc mirror --overwrite "s3/${BUCKET}/ddpm/dataset"  "${ART}/dataset"  2>/dev/null
else
  echo "-- mc absent, restauration S3 sautée"
fi

# 7. Vérification
"${VENV}/bin/python" - <<'PY'
import torch
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU seul"
print(f"   torch {torch.__version__} | {gpu}")
PY

echo "== terminé. Fais maintenant :  source ~/.bashrc =="
