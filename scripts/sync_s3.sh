#!/bin/bash
# Sauvegarde / restauration du projet DDPM sur le bucket MinIO personnel.
#
#   ./sync_s3.sh push    -> envoie dataset + poids + samples vers S3
#   ./sync_s3.sh pull    -> récupère depuis S3
#   ./sync_s3.sh status  -> ce qu'il y a dans le bucket
#
# Le bucket personnel porte le nom d'utilisateur. Le jeton d'accès est valide
# 7 jours et régénéré automatiquement : si le service devient "rouge" dans
# "Mes services", c'est que le jeton a expiré côté service -> relancer le
# service (les données du bucket, elles, ne bougent pas).
set -uo pipefail

# Le bucket porte le nom d'utilisateur *Datalab* (gfogato). $USERNAME vaut
# "onyxia" à l'intérieur du service : ne pas s'en servir comme défaut.
BUCKET="${BUCKET:-gfogato}"
PROJ="${PROJ:-/home/onyxia/work/ddpm}"
REMOTE="s3/${BUCKET}/ddpm"
ACTION="${1:-push}"

command -v mc >/dev/null || { echo "mc introuvable"; exit 1; }

case "${ACTION}" in
  push)
    echo "-> ${REMOTE}"
    mc mirror --overwrite "${PROJ}/weights"  "${REMOTE}/weights"
    mc mirror --overwrite "${PROJ}/samples"  "${REMOTE}/samples"
    mc mirror --overwrite "${PROJ}/dataset"  "${REMOTE}/dataset"
    ;;
  pull)
    echo "<- ${REMOTE}"
    mkdir -p "${PROJ}"/{weights,samples,dataset}
    mc mirror --overwrite "${REMOTE}/weights" "${PROJ}/weights"
    mc mirror --overwrite "${REMOTE}/samples" "${PROJ}/samples"
    mc mirror --overwrite "${REMOTE}/dataset" "${PROJ}/dataset"
    ;;
  status)
    # Toujours viser le bucket explicitement : "mc ls s3/" tout court reste
    # bloqué, la politique stsonly n'autorise pas ListBuckets.
    mc ls -r "${REMOTE}" 2>/dev/null || echo "rien dans ${REMOTE}"
    mc du "${REMOTE}" 2>/dev/null
    ;;
  *)
    echo "usage: $0 {push|pull|status}"; exit 1;;
esac
