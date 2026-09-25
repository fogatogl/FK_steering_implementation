# Sequential Monte Carlo for diffusion models: FK Steering, then PG-DLM

Giulio Fogato, ENSAE Paris.

## Overview

Reproduction from the equations of two Sequential Monte Carlo methods for
guided sampling of diffusion models:

- **FK Steering** (Singhal et al., 2025): a particle filter along the denoising
  trajectory, with three intermediate potentials (`difference`, `max`, `sum`)
  that all target `p(x) exp(λ r(x))`.
- **PG-DLM** (Dang et al., 2025): Particle Gibbs on full trajectories, for
  masked diffusion language models. Coming next.

All the SMC is plain PyTorch, in `smc/`. Three base models sit behind the same
`step` / `predict_x0` interface:

- Stable Diffusion v1.5 with ImageReward as the reward, the paper's own experiment;
- a DDPM trained here on CIFAR-10 (8.4M-parameter U-Net, `notebooks/demo_DDPM.ipynb`);
- `google/ddpm-ema-celebahq-256` from the Hub, 256 px faces, where the damage
  done by λ is visible to the eye.

The write-up is the blog post [`docs/paper.md`](docs/paper.md), *Four particles, one image*.

### The reproduction on Stable Diffusion v1.5

![SD v1.5, ImageReward and HPS](figures/f3_reproduction.png)

Stable Diffusion v1.5 with ImageReward as the reward, the SD row of the paper's
table 1 at the paper's setting (λ = 10, k = 4, MAX potential, five scheduled
steps), 100 prompts, three seeds, budget matched on UNet rows:

| | ImageReward | HPS v2.1 | paper (IR / HPS) |
|---|---|---|---|
| one sample | 0.237 ± 0.082 | 0.245 | 0.187 / 0.245 |
| best-of-4 | 0.758 ± 0.069 | 0.258 | 0.737 / 0.265 |
| FK, k = 4 | 0.820 ± 0.069 | 0.259 | 0.898 / 0.263 |

Best-of-4 lands on the paper; one sample sits 0.05 above it. FK beats best-of-4
at equal budget, by +0.062 ± 0.026 on 100 paired prompts (69 prompts won), where
the paper has +0.161; HPS, the judge nobody optimises, does not move. The
diagnostics say where the rest went: at the first scheduled step the median ESS
is 1.18 out of 4, and 93 to 96 FK runs of 100 (one rerun on each GPU) end with
their four particles descending from one initial noise.

![The four finals of one prompt](figures/f5_root_grid.png)

The four finals of one prompt under the free sampler, FK at the paper's setting and
floor + λ = 2, on the same four noises, each framed in the colour of its root: the FK
row is four near-copies of one image.

The collapse has a mechanism and a price (`collapse_lab/FINDINGS.md`,
`docs/results.md` block 18). Reweighting best-of-4's four free draws by
exp(10 ir) already gives an ESS of 1.23 out of 4: at lambda = 10 the target
itself carries about one particle. The number of roots that survive is a
function of the recorded weights and the resampler alone, recovered within 0.09
on eighteen steered arms without knowing anything about the steering. Of the corrections
tried (the released code's floor at 0, a smaller lambda, a lambda bisected to
hold the ESS, a threshold, a later schedule), one keeps the lineages, the floor
with lambda = 2: 3.0 roots of 4, at -0.01 against FK on the best image paired by
x_T and -0.25 on the mean of the four. No configuration beats best-of-4 by more than the
noise on the best image, while the diversity varies by a factor three.

The reference: the paper's stated configuration is this repository's, up to two
choices that were screened (the increment form of MAX and the systematic resampler).
The authors' released code, run from its own clone on the same prompts, returns
best-of-4's rewards to the fourth decimal without its filter; with it, four runs on
40 prompts land between -0.35 and +0.10 against best-of-4. Moving the released
code's choices into this repository's filter one at a time does not close the gap
to the paper, which stays open (`docs/reference_config.md`, post section 7).

Every number of the post is printed by a script that `scripts/post_numbers.py` runs,
or stands in a dated block of `docs/results.md`; `bash scripts/check_all.sh` runs the
post's checks (prose, numbers, build, format, figures, language), and `docs/data_freeze.md`
holds the sha256 of every record it reads. `docs/collapse_findings_en.md` is an English
summary of the collapse lab, and `docs/demo/index.html` an interactive version of its
figures.

### Earlier stages: CIFAR-10 and CelebA-HQ

The first reward is a deliberately simple "red" score. FK Steering pushes it up
with λ, and that is the problem: `sum` pushes it past its own bound of 10.19,
the images leave [−1, 1] (at λ = 8 the pixels span [−1.26, 1.30]), and the CIFAR
samples are flat red squares that the judge B gives a mean p(cat) of 0.54 over
three seeds, against 0.27 for the free model. A reward that can be gamed, and a
metric that follows it: the diagnosis of reward hacking
(`figures/fig2_sweep_lambda.png`, `figures/fig3_samples_by_reward.png`).

The classifier reward is the interesting part. `r(x) = log p_A(cat | x)` with A
a small VGG (89 %); with the `difference` potential the product telescopes to
`p(x) p_A(cat | x)^λ`, so λ = 1 is the Bayes posterior under A. A second
classifier B (ResNet-18, 93 %, calibrated) only judges and never guides.

![classifier sweep](figures/fig2_sweep_lambda_classifier.png)

CIFAR-10, k = 16, three seeds: the log-probability of the drawn particle goes
from −8.3 (free) to −0.5 at λ = 1 and −0.2 at λ = 4, while the minimum ESS
falls from 16 to 2.1 and then to 1.05. Judged by B over the three seeds, 15 of
48 final particles are cats at λ = 1 against 11 of 48 for the free model, on
images that stay plausible; the count keeps climbing to 32 of 48 at λ = 2, and
the pairwise pixel distance between the sixteen finals falls from 0.34 to 0.18
along the way. On seed 2024 alone the λ = 1 column reads 10 of 16, on seed 2025
it reads 0 of 16.
Reference point: the DDPM fine-tuned on the cat class reaches FID 51.4 against
80.2 for the base model.

![CelebA-HQ samples](figures/fig3b_hub_samples_by_reward.png)

CelebA-HQ 256 with a glasses classifier as reward: over three seeds λ = 1 puts
glasses on 21 of 48 faces for B against 2 of 48 for the free model, on faces
that stay clean, with per-seed counts of 7, 0 and 14. From λ = 2 the ESS drops
to 1, the sixteen particles descend from one ancestor chosen on blurry Tweedie
estimates where A is fooled, and B then finds glasses on 0 of 48. Same failure
as the red squares, visible to the eye this time.

The FID says the same thing on 2048 images per lot, against two references: the
1468 faces with glasses, and all 30000 faces.

| | vs glasses | vs all faces |
|---|---|---|
| free model | 123.8 | 43.6 |
| FK glasses, λ = 1 | 65.1 | 52.6 |
| FK glasses, λ = 2 | 73.7 | 67.3 |

At λ = 1 the distance to the target is halved for nine points of global FID,
without retraining anything; on CIFAR, fine-tuning the whole DDPM on the cat
class bought a comparable gap (80.2 to 51.4). At λ = 2 both columns get worse.
The lots keep the sixteen particles of each run, duplication included, so these
are pessimistic bounds. Each choice behind these numbers is one entry of
`docs/decisions.md`.

## Getting started

### Code and environment

Python ≥ 3.11, plain `pip`. Two requirement files:

```bash
pip install -r requirements.txt && pip install -e .    # pinned, anywhere
pip install -r requirements-onyxia.txt && pip install -e .   # inside the SSP Cloud image, keeps its torch
```

The experiments ran on one shared 16 GB GPU, which the computing service
allocated per session: an NVIDIA A2 for some sessions, a faster card for others
(its model was not recorded before 23/09; the first week's notes say T4). Stable
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

On Onyxia, `scripts/onyxia_bootstrap.sh` brings an instance back to a working
state; `docs/ONYXIA_setup.md` says what survives a restart and what does not.

### Data and weights

Nothing heavy is in git. Everything lives in `/home/onyxia/work/ddpm/` and is
mirrored on an S3 bucket with `scripts/sync_s3.sh {push,pull,status}`:

| | |
|---|---|
| `weights/ddpm_last.pt`, `weights/ft_class3/` | CIFAR-10 DDPM, base and fine-tuned on class 3 (EMA and raw weights) |
| `weights/classifier_*.pt` | guides A (small VGG) and judges B (ResNet-18), CIFAR-10 and CelebA-HQ Eyeglasses |
| `dataset/` | CIFAR-10; CelebA-HQ 256 with its 40 attributes (arrow) and its 64 px uint8 cache |
| `hf/` | HuggingFace cache (`HF_HOME`), so the Hub model survives a restart |
| `fid/` | reference and generated PNGs, Inception statistics |

Sample tensors (`samples/*.pt`) are written next to each results JSON and are
git-ignored; the JSON records the path.

### Logging

No experiment tracker. Each run writes one flat JSON in `results/`, one record
per (method, potential, λ, resampler, seed) with the generator seed, so a single
cell can be replayed. Figures are rebuilt from those JSON files only.

## Reproduction

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

# every number the post quotes, from the committed records (CPU, about 10 min), and the checks
/home/onyxia/work/.venvs/ddpm/bin/python scripts/post_numbers.py   # results/post_numbers.json
bash scripts/check_all.sh
```

`results/` and `figures/` in the repo are the outputs of these commands.

## Repository structure

```
smc/
  weights.py, resampling.py   normalised log-weights, ESS, resamplers (return indices)
  fk.py                       fk_steer, best_of_n, the three potentials
  models.py, scheduler.py     the DDPM and SD wrappers behind step / predict_x0; DDPM and DDIM schedulers
  unet.py, ema.py             the CIFAR-10 network and its EMA
  pretrained.py               a Hub UNet behind the same interface
  rewards.py, rewards_sd.py   red and classifier rewards; ImageReward on SD
  classifier.py, rng.py       small VGG and ResNet-18; one torch.Generator per run
tests/                        one property per test; `particles` is the oracle for the resamplers
experiments/                  training and sampling runs, JSON in results/
scripts/
  run_*.py, run_*.sh          the lambda sweep, FID, the SD baseline and its launchers
  plot_*.py, fig_*.py         the figures, from results/ only (figstyle.py holds the palette)
  post_numbers.py, check_*.py every number the post quotes, and the post's checks (check_all.sh)
  build_post.py               the post in the conference's format
  sync_s3.sh, onyxia_*.sh     S3 backup and the Onyxia bootstrap
collapse_lab/                 the collapse study: probe, readouts, coalescence model, released-code driver
results/                      one JSON per run; post_numbers/ holds what each analysis script prints
figures/                      every figure, rebuilt from results/ and collapse_lab/out/
data/                         the ImageReward benchmark prompts and the selection of the prompts shown
docs/
  paper.md, post/             the blog post and its .bib
  results.md                  one block per run: what it tested, what it cost, what it gave
  protocol_sd.md              the SD block, written before it was launched, predictions included
  reference_config.md         the paper against the released code, cell by cell
  decisions.md                why each choice, one short entry each
  architecture.md             the three stacks, every network and every parameter value
  chronology.md               the thread from one stage to the next, and the questions that drove it
  max_potential.md            why the max potential underperforms, and what was ruled out
  protocol_potentials.md      the two paper-code mismatches, and the run that settles the second
  collapse_findings_en.md     the collapse lab's final state, in English
  visual_selection.md         the rule that picked the prompts shown as images
  data_freeze.md              the sha256 of every record the post reads
  demo/                       the interactive figures, one standalone HTML page
  ONYXIA_setup.md             bringing an ephemeral instance back
LEARNING.md                   the bugs that cost more than twenty minutes
notebooks/, third_party/      the DDPM training notebook; the MDLM checkpoint for PG-DLM
```

## License

MIT, see `LICENSE`.
