#!/bin/bash
# =============================================================================
# Script d'initialisation Onyxia / SSP Cloud pour le projet DDPM
#
# À renseigner dans l'onglet "Init" > "personalInit" au lancement du service,
# sous forme d'URL publique (dépôt git public ou objet public sur MinIO).
# PersonalInitArgs (optionnel) : le nom du bucket, sinon $USERNAME est utilisé.
#
# ATTENTION : ce script est exécuté en tant que root. Tout ce qu'il crée
# appartient à root, d'où le chown final (cf. doc SSP Cloud).
# =============================================================================
set -uo pipefail

USER_NAME="${USERNAME:-onyxia}"
GROUP_NAME="${GROUPNAME:-users}"
HOME_DIR="/home/${USER_NAME}"
WORK="${HOME_DIR}/work"                 # <-- volume persistant
PROJ="${WORK}/ddpm"
VENV="${WORK}/.venvs/ddpm"
BUCKET="${1:-${USER_NAME}}"

echo "== init DDPM : user=${USER_NAME} bucket=${BUCKET} =="

mkdir -p "${PROJ}/dataset" "${PROJ}/weights" "${PROJ}/samples" "${WORK}/.venvs"

# -----------------------------------------------------------------------------
# 1. Environnement Python
#    --system-site-packages : on hérite du PyTorch GPU de l'image, on ne le
#    réinstalle SURTOUT pas (2-3 Go, et on casserait l'appairage CUDA).
# -----------------------------------------------------------------------------
BASE_PY="$(command -v python3 || echo /opt/conda/bin/python)"

if [ ! -x "${VENV}/bin/python" ]; then
  echo "-- création du venv ${VENV}"
  "${BASE_PY}" -m venv --system-site-packages "${VENV}"
else
  echo "-- venv déjà présent (volume persistant), on réutilise"
fi

"${VENV}/bin/pip" install --quiet --upgrade pip
"${VENV}/bin/pip" install --quiet ipykernel matplotlib tqdm

# -----------------------------------------------------------------------------
# 2. Kernel Jupyter
#    ~/.local n'est PAS sur le volume persistant : il faut réenregistrer le
#    kernel à chaque démarrage du service. C'est instantané.
# -----------------------------------------------------------------------------
"${VENV}/bin/python" -m ipykernel install \
    --prefix "${HOME_DIR}/.local" \
    --name ddpm --display-name "Python (ddpm GPU)"

# -----------------------------------------------------------------------------
# 3. Réglages VS Code : découverte automatique du venv
# -----------------------------------------------------------------------------
mkdir -p "${PROJ}/.vscode"
cat > "${PROJ}/.vscode/settings.json" <<EOF
{
  "python.defaultInterpreterPath": "${VENV}/bin/python",
  "python.venvPath": "${WORK}/.venvs",
  "python.terminal.activateEnvironment": true,
  "jupyter.kernels.excludePythonEnvironments": []
}
EOF

# -----------------------------------------------------------------------------
# 4. Restauration depuis S3 (MinIO)
#    Exécuté en tant qu'utilisateur onyxia : l'alias "s3" de mc vit dans son
#    ~/.mc/config.json, root ne le voit pas.
# -----------------------------------------------------------------------------
if command -v mc >/dev/null 2>&1; then
  su "${USER_NAME}" -c "
    mc mirror --overwrite 's3/${BUCKET}/ddpm/weights' '${PROJ}/weights' 2>/dev/null
    mc mirror --overwrite 's3/${BUCKET}/ddpm/dataset' '${PROJ}/dataset' 2>/dev/null
  " || echo "-- restauration S3 ignorée (alias mc pas encore prêt, faire ./sync_s3.sh pull)"
else
  echo "-- mc absent, restauration S3 sautée"
fi

# -----------------------------------------------------------------------------
# 5. Droits (obligatoire : le script tourne en root)
# -----------------------------------------------------------------------------
chown -R "${USER_NAME}:${GROUP_NAME}" "${HOME_DIR}"

echo "== init terminé =="
echo "   venv    : ${VENV}"
echo "   projet  : ${PROJ}"
echo "   kernel  : Python (ddpm GPU)"
