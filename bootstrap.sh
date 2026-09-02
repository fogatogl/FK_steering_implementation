#!/bin/bash
source /home/onyxia/work/.venvs/ddpm/bin/activate
echo "HF_HOME=/home/onyxia/work/.hf" >> ~/.bashrc  # évite de re-télécharger MDLM
python -m ipykernel install --user --name ddpm --display-name "ddpm"
