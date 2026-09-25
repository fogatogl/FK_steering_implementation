#!/bin/bash
# Session D (23/09 evening): ctl, lam0 and floor2 at 100 prompts in one process, on the GPU the
# service allocated (an NVIDIA A2 since 18:02), with the four finals and the guide's Tweedie
# thumbnails saved for every prompt and arm. Predictions: docs/protocol_sd.md, "Pre-registration
# of session D". A new file: nothing of session C is touched.
set -x
export HF_HOME=/home/onyxia/work/hf_cache
V=/home/onyxia/work/.venvs/sd/bin/python
cd "$(dirname "$0")/.."
O=/home/onyxia/work/diffusion-models/collapse_lab/out/session_D/probe_D.json
$V collapse_lab/probe.py --arms ctl lam0 floor2 --limit 100 --save-images --out $O
echo "NUIT TERMINEE"
