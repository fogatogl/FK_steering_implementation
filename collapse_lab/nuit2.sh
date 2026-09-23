#!/bin/bash
# Nuit 2 (22-23/09) : la reference, puis le collapse. Pre-enregistree dans
# docs/protocol_sd.md, "Pre-registration of the reference arms and the collapse night".
# Sequentiel, un chargement de modele par appel, reprise par (prompt, bras) dans
# probe.json et par (prompt, sampler, seed) dans sd_authors_R0.json. Les sorties vont par
# chemin absolu dans le checkout de l'auteur.
set -x
V=/home/onyxia/work/.venvs/sd/bin/python
cd "$(dirname "$0")/.."
O=/home/onyxia/work/diffusion-models/collapse_lab/out/probe.json
# 1. le test des latents, deux prompts
$V collapse_lab/m_latents.py --i 0
$V collapse_lab/m_latents.py --i 1
# 2. la reference : smc avec leurs choix, puis leur code
$V collapse_lab/probe.py --arms R1 --limit 100 --out $O
$V collapse_lab/ref/run_authors.py --config paper --limit 100
# 3. les images, six prompts par regle, quatre bras
IDS=$($V collapse_lab/q_image_grid.py --choose)
$V collapse_lab/probe.py --arms lam0 ctl floor2 R1 --limit 100 --prompt-ids $IDS --save-images --out $O
# 4. la bissection, un choix a la fois
$V collapse_lab/probe.py --arms stat0 multi --limit 40 --out $O
$V collapse_lab/probe.py --arms vae idx   --limit 40 --out $O
# 5. B2, B3
$V collapse_lab/probe.py --arms thr05 --limit 40 --out $O
$V collapse_lab/probe.py --arms rise  --limit 20 --out $O
echo "NUIT TERMINEE"
