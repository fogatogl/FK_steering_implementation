#!/bin/bash
# Onyxia / SSP Cloud init script for the DDPM project.
#
# Give it in the "Init" tab > "personalInit" when launching the service, as a
# public URL (public git repo or public MinIO object). PersonalInitArgs
# (optional): the bucket name, otherwise $USERNAME is used.
#
# WARNING: this runs as root. Everything it creates belongs to root, hence the
# final chown (see the SSP Cloud docs).
set -uo pipefail

USER_NAME="${USERNAME:-onyxia}"
GROUP_NAME="${GROUPNAME:-users}"
HOME_DIR="/home/${USER_NAME}"
WORK="${HOME_DIR}/work"                 # <-- persistent volume
PROJ="${WORK}/ddpm"
VENV="${WORK}/.venvs/ddpm"
BUCKET="${1:-${USER_NAME}}"

echo "== DDPM init: user=${USER_NAME} bucket=${BUCKET} =="

mkdir -p "${PROJ}/dataset" "${PROJ}/weights" "${PROJ}/samples" "${WORK}/.venvs"

# 1. Python environment. --system-site-packages: inherit the image's GPU
#    PyTorch, never reinstall it (2-3 GB, and the CUDA pairing breaks).
BASE_PY="$(command -v python3 || echo /opt/conda/bin/python)"

if [ ! -x "${VENV}/bin/python" ]; then
  echo "-- creating venv ${VENV}"
  "${BASE_PY}" -m venv --system-site-packages "${VENV}"
else
  echo "-- venv already present (persistent volume), reusing"
fi

"${VENV}/bin/pip" install --quiet --upgrade pip
"${VENV}/bin/pip" install --quiet ipykernel matplotlib tqdm

# 2. Jupyter kernel. ~/.local is NOT on the persistent volume: the kernel must
#    be re-registered at every service start. Instant.
"${VENV}/bin/python" -m ipykernel install \
    --prefix "${HOME_DIR}/.local" \
    --name ddpm --display-name "Python (ddpm GPU)"

# 3. VS Code settings: automatic venv discovery
mkdir -p "${PROJ}/.vscode"
cat > "${PROJ}/.vscode/settings.json" <<SETTINGS
{
  "python.defaultInterpreterPath": "${VENV}/bin/python",
  "python.venvPath": "${WORK}/.venvs",
  "python.terminal.activateEnvironment": true,
  "jupyter.kernels.excludePythonEnvironments": []
}
SETTINGS

# 4. Restore from S3 (MinIO). Run as the onyxia user: the "s3" alias of mc
#    lives in its ~/.mc/config.json, root does not see it.
if command -v mc >/dev/null 2>&1; then
  su "${USER_NAME}" -c "
    mc mirror --overwrite 's3/${BUCKET}/ddpm/weights' '${PROJ}/weights' 2>/dev/null
    mc mirror --overwrite 's3/${BUCKET}/ddpm/dataset' '${PROJ}/dataset' 2>/dev/null
  " || echo "-- S3 restore skipped (mc alias not ready yet, run ./sync_s3.sh pull)"
else
  echo "-- mc missing, S3 restore skipped"
fi

# 5. Ownership (mandatory: the script runs as root)
chown -R "${USER_NAME}:${GROUP_NAME}" "${HOME_DIR}"

echo "== init done =="
echo "   venv    : ${VENV}"
echo "   project : ${PROJ}"
echo "   kernel  : Python (ddpm GPU)"
