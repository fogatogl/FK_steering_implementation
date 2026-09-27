# Reproducing every run

The main commands behind the records of `results/` and `collapse_lab/out/`, the environment they
need, and where the heavy files live; `docs/results.md` describes every run, block by block. The post's numbers and figures need none of this: they are
rebuilt on CPU from the committed records (`scripts/post_numbers.py`, the figure scripts).

## Environment

Python ≥ 3.11, plain `pip`. Two requirement files:

```bash
pip install -r requirements.txt && pip install -e .    # pinned, anywhere
pip install -r requirements-onyxia.txt && pip install -e .   # inside the SSP Cloud image, keeps its torch
```

The experiments ran on one shared 16 GB GPU, which the computing service
allocated per session: an NVIDIA A2 for some sessions, a T4 for the others. Stable
Diffusion runs reproduce to the fourth decimal on one GPU model and not across
two, so a rerun on another card returns other images and other numbers for the
same seeds; runs are paired by noise only within one machine
(`collapse_lab/commun.py`, `groupe`). Since 23/09 every `collapse_lab/probe.py`
record names its process and device. The 256 px model needs about 11 GB at
k = 16 in fp16; the CIFAR model and the tests need much less. The
fast tests run on CPU:

```bash
pytest              # fast, no GPU, no weights
pytest -m slow      # full runs on the real model, needs the weights
```

## Data and weights

Nothing heavy is in git. Everything lives outside the repository, in `/home/onyxia/work/ddpm/` on the
machine that ran it, backed up to private storage and not distributed:

| | |
|---|---|
| `weights/ddpm_last.pt`, `weights/ft_class3/` | CIFAR-10 DDPM, base and fine-tuned on class 3 (EMA and raw weights) |
| `weights/classifier_*.pt` | guides A (small VGG) and judges B (ResNet-18), CIFAR-10 and CelebA-HQ Eyeglasses |
| `dataset/` | CIFAR-10; CelebA-HQ 256 with its 40 attributes (arrow) and its 64 px uint8 cache |
| `hf/` | HuggingFace cache (`HF_HOME`), so the Hub model survives a restart |
| `fid/` | reference and generated PNGs, Inception statistics |

Sample tensors (`samples/*.pt`) are written next to each results JSON and are
git-ignored; the JSON records the path.

## Records

No experiment tracker. Each run writes one flat JSON in `results/`, one record
per (method, potential, λ, resampler, seed) with the generator seed, so a single
cell can be replayed. Figures are rebuilt from those JSON files only.

## The runs, in order

From the root, in this order. Each script says in its docstring which figure
it feeds.

```bash
# CIFAR-10, red reward
python -m experiments.run_free_samples            # results/free_samples_*.json   -> plot_fig0.py
python -m experiments.run_best_of_n               # results/bestofn.json           -> plot_fig1.py
python scripts/run_sweep_lambda.py                # results/sweep_lambda.json      -> plot_fig2.py

# CIFAR-10, classifier reward (target: cat)
python experiments/train_classifier.py --arch small --seed 0                 # guide A
python experiments/train_classifier.py --arch resnet18 --seed 1 --epochs 90  # judge B
python experiments/train_classifier.py --arch resnet18 --seed 1 --epochs 90 --calibrate
python scripts/run_sweep_lambda.py --reward classifier --lam 0 0.5 1 2 4 \
       --out results/sweep_lambda_classifier.json --save-images
python scripts/plot_fig3.py                       # figures/fig3_samples_by_reward.png
scripts/run_fid_all.sh                            # results/fid.json

# CelebA-HQ 256, Hub model (DDIM 50 steps, eta = 1)
python experiments/train_classifier.py --dataset celebahq --attr Eyeglasses --arch small --seed 0
python experiments/train_classifier.py --dataset celebahq --attr Eyeglasses --arch resnet18 --seed 1
python scripts/run_sweep_lambda.py --weights hub:google/ddpm-ema-celebahq-256 --steps 50 --eta 1 \
       --potentials difference --resamplers systematic --save-images --out results/sweep_lambda_hub_red.json
python scripts/run_sweep_lambda.py --weights hub:google/ddpm-ema-celebahq-256 --steps 50 --eta 1 \
       --reward classifier --target 1 --classifier-weights .../classifier_eyeglasses64_small_seed0.pt \
       --lam 0 0.5 1 2 4 --potentials difference --resamplers systematic --save-images \
       --out results/sweep_lambda_hub_classifier.json
python scripts/plot_grid.py --json results/sweep_lambda_hub_red.json --judge .../classifier_eyeglasses64_resnet18_seed1.pt

# what k buys at a matched budget
python scripts/run_sweep_lambda.py --reward classifier --k 2 4 8 16 --lam 0.5 1 2 \
       --potentials difference --resamplers systematic --save-images \
       --out results/sweep_k_classifier.json
python scripts/run_sweep_lambda.py --reward classifier --k 4 8 16 --lam 0.5 1 2 \
       --potentials max --resamplers systematic --save-images \
       --out results/sweep_k_max_classifier.json
python scripts/plot_fig5_k.py                     # figures/fig5_sweep_k.png

# SD v1.5 (separate venv, see scripts/setup_sd_env.sh)
HF_HOME=/home/onyxia/work/hf_cache python scripts/run_sd_baseline.py \
       --prompts data/imagereward-benchmark-prompts.json --seeds 2024 2025 2026
python scripts/plot_fig4_sd.py                    # figures/f3_reproduction.png
python scripts/make_table_sd.py
scripts/run_sd_variants.sh && scripts/run_sd_lambda_t.sh && scripts/run_sd_tempering.sh   # results/sd_variants/
python scripts/compare_sd_variants.py             # the paired table of the screen
python scripts/plot_fig6_sd_grid.py --rows bon4 fk4 T2A05 T2tA05   # figures/fig6_sd_collapse.png
python scripts/analyze_sd_collapse.py             # figures/fig7_ess_vs_gain.png, and the bootstrap

# the collapse lab, sessions C and D (one process each, ctl, lam0, floor2 at 100 prompts)
collapse_lab/nuitD.sh                             # collapse_lab/out/session_D/, with images and Tweedie thumbnails
collapse_lab/nuitD_hps.sh                         # HPS v2.1 of session D's finals
python scripts/select_visual_prompts.py collapse_lab/out/session_D/probe_D.json   # data/visual_selection.json

# sessions E and F, the T4: the machine check, the released code at seed 2024 on the 100 prompts,
# and its commit before the max-potential fix (a second checkout at 6726324, FKD_ROOT)
collapse_lab/nuitE.sh                             # session_E/t4_check.json, results/sd_authors_R0_100.json
collapse_lab/nuitF.sh                             # results/sd_authors_prefix.json

# session G, the A2: seeds 2025 and 2026 of best-of-4, the released code and its commit before the
# fix, then FK, each seed paired by x_T on one machine
collapse_lab/nuitG.sh                             # results/sd_seeds_bon4.json, results/sd_seeds_authors.json
collapse_lab/nuitG_fk.sh                          # results/sd_seeds_fk4.json, after nuitG.sh
python collapse_lab/z_sessionG.py                 # the three-seed rows of docs/results.md, block 20

# every number the post quotes, from the committed records (CPU, about 10 min), and the checks
/home/onyxia/work/.venvs/ddpm/bin/python scripts/post_numbers.py   # results/post_numbers.json
bash scripts/check_all.sh
```

`results/` and `figures/` in the repo are the outputs of these commands.
