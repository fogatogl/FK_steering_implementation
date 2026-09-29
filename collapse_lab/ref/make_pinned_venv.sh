#!/bin/bash
# The released code's pinned versions (requirements.txt of Fk-Diffusion-Steering): torch 2.4.0,
# transformers 4.38.2, diffusers at af28ae2d, ImageReward at 2ca71bac (https instead of their ssh URL),
# on Python 3.12, since torch 2.4.0 has no 3.13 wheels. openai-clip and setuptools < 81 as in
# requirements-sd.txt (their file does not list clip). Session H, check H3.
set -e
UV=/opt/python/bin/uv
P=${1:-/home/onyxia/work/.venvs/fkd_pinned}
TORCH_INDEX=https://download.pytorch.org/whl/cu124
[ -x $P/bin/python ] || $UV venv $P --python /usr/bin/python3.12
$UV pip install --python $P/bin/python torch==2.4.0 torchvision==0.19.0 --index-url $TORCH_INDEX
# ImageReward and openai-clip import pkg_resources in their setup.py: built against this setuptools
$UV pip install --python $P/bin/python "setuptools<81" wheel
$UV pip install --python $P/bin/python --extra-index-url $TORCH_INDEX --index-strategy unsafe-best-match \
    --no-build-isolation-package image-reward --no-build-isolation-package openai-clip \
    "setuptools<81" torch==2.4.0 torchvision==0.19.0 transformers==4.38.2 tokenizers==0.15.2 accelerate==1.2.1 \
    huggingface-hub==0.27.1 numpy==1.26.3 safetensors==0.5.2 pillow==11.1.0 timm==1.0.13 fairscale==0.4.13 \
    ftfy==6.3.1 regex==2024.11.6 scipy==1.13.1 sentencepiece==0.2.0 protobuf==3.20.3 einops==0.8.0 \
    tqdm==4.66.4 openai-clip==1.0.1 hpsv2==1.2.0 \
    "diffusers @ git+https://github.com/huggingface/diffusers@af28ae2d5ba0ef80d99fff7859ebea730e1cf3f8" \
    "image-reward @ git+https://github.com/THUDM/ImageReward.git@2ca71bac4ed86b922fe53ddaec3109fe94d45fd3"
$P/bin/python -c "import torch, diffusers, transformers, ImageReward, hpsv2, clip; \
print(torch.__version__, torch.cuda.is_available(), diffusers.__version__, transformers.__version__)"
