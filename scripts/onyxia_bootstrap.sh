#!/usr/bin/env bash
# onyxia_bootstrap.sh — remet une instance Onyxia en état de marche.
# Idempotent. Usage :  bash scripts/onyxia_bootstrap.sh && source ~/.bashrc
#
# Variables utiles :
#   FULL_WEIGHTS=1   rapatrie TOUS les checkpoints S3 (2 Go) au lieu des
#                    seuls ddpm_last.pt / ft_class3/ddpm_last.pt (257 Mo)
#   NO_S3=1          saute complètement l'étape S3
set -uo pipefail

# ---- valeurs du projet -------------------------------------------------------
GH_OWNER="${GH_OWNER:-fogatogl}"
REPO_NAME="${REPO_NAME:-diffusion-models}"
# Le bucket MinIO porte le nom d'utilisateur *Datalab* (gfogato), pas $USERNAME
# qui vaut "onyxia" à l'intérieur du service : ne pas remplacer par $USERNAME.
BUCKET="${BUCKET:-gfogato}"
# -----------------------------------------------------------------------------

WORK="/home/onyxia/work"
REPO="${WORK}/${REPO_NAME}"
VENV="${WORK}/.venvs/ddpm"
ART="${WORK}/ddpm"                       # artefacts lourds, hors git, miroir de s3://$BUCKET/ddpm
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

# 2. Identité git (~/.gitconfig n'est pas persistant).
#    Quand l'onglet Git du formulaire est rempli, Onyxia a déjà posé le bon
#    nom et le bon e-mail : on ne les écrase pas, on ne comble que les trous.
[ -n "$(git config --global user.name  || true)" ] || git config --global user.name  "${GIT_NAME:-fogatogl}"
[ -n "$(git config --global user.email || true)" ] || git config --global user.email "${GIT_MAIL:-fogatogl05@gmail.com}"
git config --global pull.ff only
echo "-- git : $(git config --global user.name) <$(git config --global user.email)>"

# 3. Venv — --system-site-packages : on hérite du PyTorch de l'image.
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
  # L'avertissement de conflit sur particles/numpy est attendu, cf. le fichier.
  "${VENV}/bin/pip" install --quiet -r "${REPO}/requirements-onyxia.txt"
else
  "${VENV}/bin/pip" install --quiet ipykernel matplotlib tqdm pytest
fi

# 4. Kernel Jupyter (~/.local n'est pas persistant)
"${VENV}/bin/python" -m ipykernel install \
    --prefix "${HOME}/.local" --name ddpm --display-name "Python (ddpm)" >/dev/null

# 5. Activation automatique du venv + variables (~/.bashrc n'est pas persistant)
if ! grep -q "${VENV}/bin/activate" "${HOME}/.bashrc" 2>/dev/null; then
  cat >> "${HOME}/.bashrc" <<EOF

# --- projet ${REPO_NAME} (réécrit à chaque session : ~/.bashrc n'est pas persistant)
export BUCKET=${BUCKET}
export REPO=${REPO}
source ${VENV}/bin/activate
EOF
fi

# 6. Artefacts lourds depuis S3 (jamais dans git).
#    ATTENTION : "mc ls s3/" sans bucket reste bloqué (la politique stsonly
#    n'autorise pas ListBuckets) — toujours viser s3/<bucket>/... explicitement.
if [ -n "${NO_S3:-}" ]; then
  echo "-- S3 sauté (NO_S3)"
elif ! command -v mc >/dev/null 2>&1; then
  echo "-- mc absent, restauration S3 sautée"
elif [ -n "${FULL_WEIGHTS:-}" ]; then
  echo "-- miroir complet des poids (2 Go)"
  mc mirror --overwrite "s3/${BUCKET}/ddpm/weights" "${ART}/weights" \
    || echo "!! poids non récupérés (jeton S3 expiré ?)"
else
  echo "-- poids : derniers checkpoints seulement (FULL_WEIGHTS=1 pour les 2 Go)"
  for f in weights/ddpm_last.pt weights/ft_class3/ddpm_last.pt samples/samples.png; do
    [ -f "${ART}/${f}" ] || mc cp "s3/${BUCKET}/ddpm/${f}" "${ART}/${f}" \
      || echo "!! ${f} non récupéré (jeton S3 expiré ?)"
  done
fi

# 7. Vérification
"${VENV}/bin/python" - <<'PY'
import torch
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU seul"
print(f"   torch {torch.__version__} | {gpu}")
PY

echo "== terminé. Fais maintenant :  source ~/.bashrc =="
