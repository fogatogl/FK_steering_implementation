#!/usr/bin/env bash
# Venv du bloc C : SD v1.5 + ImageReward + HPSv2. Voir requirements-sd.txt.
# Usage : bash scripts/setup_sd_env.sh
set -euo pipefail

VENV=${VENV:-/home/onyxia/work/.venvs/sd}
export HF_HOME=${HF_HOME:-/home/onyxia/work/hf_cache}

python -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r requirements-sd.txt

SP=$("$VENV/bin/python" -c "import site; print(site.getsitepackages()[0])")

# hpsv2 1.2.0 : import parasite laisse par un IDE, et le fichier est en CRLF
# (le motif sed doit donc rester sans ancre de fin de ligne).
sed -i '/from turtle import forward/d' "$SP/hpsv2/src/open_clip/factory.py"

# hpsv2 1.2.0 : la sdist oublie le vocabulaire BPE de son open_clip vendorise.
# C'est le meme fichier que celui d'openai-clip.
cp "$SP/clip/bpe_simple_vocab_16e6.txt.gz" "$SP/hpsv2/src/open_clip/"

# Poids SD v1.5 (~4 Go). runwayml/stable-diffusion-v1-5 renvoie 404 depuis 2024.
"$VENV/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("stable-diffusion-v1-5/stable-diffusion-v1-5",
                  allow_patterns=["*.json", "*.txt", "*fp16.safetensors", "tokenizer/*", "*.yaml"])
PY

# ImageReward.load() ignore HF_HOME : il passe local_dir=download_root a
# hf_hub_download, avec pour defaut ~/.cache/ImageReward. Or ~ est sur l'overlay
# ephemere de l'instance Onyxia, donc 1,7 Go repartent a chaque demarrage. Seul
# /home/onyxia/work survit : appeler RM.load(..., download_root="$IR_CACHE").
IR_CACHE=${IR_CACHE:-/home/onyxia/work/ir_cache}
mkdir -p "$IR_CACHE"

echo "OK. Pense a exporter HF_HOME=$HF_HOME avant d'utiliser $VENV/bin/python."
