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

All the SMC is plain PyTorch, in `smc/`. Two base models share the same
schedule (linear β, T = 1000) and the same `step` / `predict_x0` interface:

- a DDPM trained here on CIFAR-10 (8.4M-parameter U-Net, `notebooks/demo_DDPM.ipynb`);
- `google/ddpm-ema-celebahq-256` from the Hub, 256 px faces, where the damage
  done by λ is visible to the eye.

### Results so far

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

![SD v1.5, ImageReward and HPS](figures/fig4_sd_ir_hps.png)

Stable Diffusion v1.5 with ImageReward as the reward, the SD row of the paper's
table 1 at the paper's setting (λ = 10, k = 4, MAX potential, five scheduled
steps), 100 prompts, three seeds, budget matched on UNet rows:

| | ImageReward | HPS v2.1 | paper (IR / HPS) |
|---|---|---|---|
| one sample | 0.237 ± 0.082 | 0.245 | 0.187 / 0.245 |
| best-of-4 | 0.758 ± 0.069 | 0.258 | 0.737 / 0.265 |
| FK, k = 4 | 0.820 ± 0.069 | 0.259 | 0.898 / 0.263 |

The two baselines land on the paper. FK beats best-of-4 at equal budget, by
+0.062 on 100 paired prompts (2.4 standard errors, 69 prompts won), where the
paper has +0.161; HPS, the judge nobody optimises, does not move. The
diagnostics say where the rest went: at the first scheduled step the median ESS
is 1.18 out of 4, and every FK run ends with its four particles descending from
one initial noise. A screen of fifteen variants on 20 prompts
(`docs/results.md`, runs 12 to 15) says that dropping the first scheduled step
is the one change that helps the reward, that a lambda growing with the
denoising does not, and that tempering the potential with such a ramp is what
buys back the lineages: two lineages out of four and 38 % of the diversity gap
to best-of-4 closed, at no change in ImageReward.

## Getting started

### Code and environment

Python ≥ 3.11, plain `pip`. Two requirement files:

```bash
pip install -r requirements.txt && pip install -e .    # pinned, anywhere
pip install -r requirements-onyxia.txt && pip install -e .   # inside the SSP Cloud image, keeps its torch
```

The experiments ran on one Tesla T4 (16 GB). The 256 px model needs about
11 GB at k = 16 in fp16; the CIFAR model and the tests need much less. The
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
python scripts/plot_fig4_sd.py                    # figures/fig4_sd_ir_hps.png
python scripts/make_table_sd.py
scripts/run_sd_variants.sh && scripts/run_sd_lambda_t.sh && scripts/run_sd_tempering.sh   # results/sd_variants/
python scripts/compare_sd_variants.py             # the paired table of the screen
python scripts/plot_fig6_sd_grid.py --rows bon4 fk4 T2A05 T2tA05   # figures/fig6_sd_collapse.png
python scripts/analyze_sd_collapse.py             # figures/fig7_ess_vs_gain.png, and the bootstrap
```

`results/` and `figures/` in the repo are the outputs of these commands.

## Repository structure

```
smc/
  weights.py, resampling.py   normalised log-weights, ESS, resamplers (return indices)
  fk.py                       fk_steer, best_of_n, the three potentials
  models.py, scheduler.py     the DDPM behind step / predict_x0; DDPM and DDIM schedulers
  unet.py, ema.py             the CIFAR-10 network and its EMA
  pretrained.py               a Hub UNet behind the same interface
  rewards.py, classifier.py   red and classifier rewards; small VGG and ResNet-18
  rng.py                      one torch.Generator per run
experiments/                  training and sampling runs, JSON in results/
scripts/                      lambda sweep, FID, figures, Onyxia and S3 plumbing
tests/                        one property per test; `particles` is the oracle for the resamplers
docs/decisions.md             why each choice, one short entry each
docs/results.md               one block per run: what it tested, what it cost, what it gave
docs/chronology.md            the thread from one stage to the next, and the questions that drove it
docs/protocol_sd.md           the SD block, written before it was launched, predictions included
docs/ONYXIA_setup.md          bringing an ephemeral instance back
LEARNING.md                   the bugs that cost more than twenty minutes
notebooks/, third_party/      the DDPM training notebook; the MDLM checkpoint for PG-DLM
```

## License

MIT, see `LICENSE`.
