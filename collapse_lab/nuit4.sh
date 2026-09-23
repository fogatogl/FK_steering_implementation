#!/bin/bash
# 23/09, apres-midi : le chemin smc.models est-il rejouable d'un processus a l'autre ?
# ctl sur les prompts 0 et 1, trois processus : A et B sans rien changer, C avec
# cudnn.benchmark = False et use_deterministic_algorithms(True). Un fichier par processus,
# rien n'est ecrase. Lecture : collapse_lab/u_determinism.py (compare aussi a probe_C.json,
# session C du matin, et a sd_ref_fields100.json, 21/09).
set -x
export HF_HOME=/home/onyxia/work/hf_cache
export CUBLAS_WORKSPACE_CONFIG=:4096:8
V=/home/onyxia/work/.venvs/sd/bin/python
cd "$(dirname "$0")/.."
O=collapse_lab/out
$V collapse_lab/probe.py --arms ctl --limit 2 --out $O/det_a.json
$V collapse_lab/probe.py --arms ctl --limit 2 --out $O/det_b.json
$V -c "
import sys, runpy, torch
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
sys.argv = ['probe.py', '--arms', 'ctl', '--limit', '2', '--out', '$O/det_c.json']
runpy.run_path('collapse_lab/probe.py', run_name='__main__')
"
echo "NUIT4 TERMINEE"
