# Four particles, one image: FK Steering reproduced on Stable Diffusion

Giulio Fogato, ENSAE Paris.

A reimplementation from the equations of FK Steering ([Singhal et al., 2025](https://arxiv.org/abs/2501.06848)),
a particle filter that steers a diffusion model toward a reward at inference time, and a rerun of its
Stable Diffusion v1.5 experiment. The filter is plain PyTorch in `smc/`; every number and figure is
rebuilt from the run records committed in `results/` and `collapse_lab/out/`.

The write-up is the blog post [`docs/paper.md`](docs/paper.md).

![Four finals of one prompt under three samplers](figures/f5_root_grid.png)

*One prompt, the same four initial noises: the free sampler (top), FK at the paper's setting (middle)
and FK with a floor and λ = 2 (bottom), each image framed in the colour of the noise it descends from.
FK returns four variations of one image.*

## Results

SD v1.5 with ImageReward as the reward, at the paper's setting (λ = 10, k = 4, MAX potential, five
scheduled steps), on the 100 ImageReward benchmark prompts × 3 seeds, at an equal number of UNet
evaluations:

| | ImageReward, best of k | HPS v2.1 | paper (IR / HPS) |
|---|---|---|---|
| one sample | 0.237 | 0.245 | 0.187 / 0.245 |
| best-of-4 | 0.758 | 0.258 | 0.737 / 0.265 |
| FK, k = 4 | 0.820 | 0.259 | 0.898 / 0.263 |

- **Best-of-4 reproduces, FK's gain over it is smaller:** +0.062 ± 0.026 on 100 prompts paired on
  their initial noises, against +0.161 in the paper (post, section 3).
- **The four particles become one.** In 93 to 96 runs of 100 the four final images descend from a
  single initial noise: the first resampling weighs rewards read on a blurred estimate of the image.
  Replaying the recorded weights through the resampler recovers the number of surviving lineages
  within 0.09 on eighteen steered runs of the probe (sections 4 and 5).
- **Keeping the lineages changes the target.** Of eight corrections, only a floor at 0 with λ = 2
  keeps three roots of four; its best image stays at best-of-4's level and the mean of its four
  images costs 0.250 ± 0.045 against FK (section 6).
- **The released code does not close the gap.** Run under the paper's configuration, its four runs
  on 40 prompts span -0.35 to +0.10 against best-of-4, two seedings of one seed disagreeing by more
  than their noise (section 7).

## Repository

```
smc/            the filter: weights, resamplers, the three potentials, fk_steer, the model wrappers
tests/          one mathematical property per test: the product constraint, λ = 0 is the free
                model, the resamplers against the `particles` library
scripts/        the SD runs and their launchers, the figures, the post's numbers and checks
collapse_lab/   the study of the collapse: the probe, its readouts, the lineage replay, the
                driver of the released code
results/        one JSON record per run; post_numbers/ holds what each analysis script prints
figures/        the post's eight figures, and those of the CIFAR-10 and CelebA-HQ stages
docs/           the post, the log of every run (results.md), the predictions written before
                the runs (protocol_sd.md), the design decisions, the reproduction commands
experiments/, notebooks/   the CIFAR-10 DDPM and the classifiers of the first stages
```

## Setup

```bash
pip install -r requirements.txt && pip install -e .
pytest                  # CPU, no weights: the mathematical properties of smc/
```

Stable Diffusion needs its own environment (`requirements-sd.txt`, `scripts/setup_sd_env.sh`).

## Reproducing

The numbers and figures of the post come from the committed records, on CPU:

```bash
python scripts/post_numbers.py     # every number the post quotes, about 10 min
bash scripts/check_all.sh          # the post's checks: numbers, prose, figures, build
```

The GPU runs, their order, their commands and the weights they need are in
[`docs/reproduction.md`](docs/reproduction.md). They ran on two GPU models, an NVIDIA A2 and a T4:
a Stable Diffusion run reproduces image by image on one model and not across the two, so runs are
paired by initial noise only within one machine.

## Earlier stages, on CIFAR-10 and CelebA-HQ

The same filter first ran on a DDPM trained here on CIFAR-10 and on a CelebA-HQ 256 model from the
Hub, with rewards simple enough to read: a redness score, then classifiers for "cat" and for
"glasses", judged by a second classifier that never guides. Raising λ drives the reward up and the
effective sample size to one particle at both scales, and a reward that can be gamed is gamed: the
redness score ends above its own bound on flat red squares. The numbers are in the post's appendix
A.6 and in `docs/results.md`.

## Who wrote what

I wrote the filter and its model wrappers (`smc/`), the tests of its mathematical properties, the
CIFAR-10 DDPM and its training (`notebooks/demo_DDPM.ipynb`), the classifiers used as rewards and
judges (`smc/classifier.py`, `experiments/train_classifier.py`), and the analysis of the collapse.
An AI assistant (Claude, Anthropic) wrote the launch scripts, the figure scripts and the
documentation.

## Citation

The reproduced paper:

```bibtex
@inproceedings{singhal2025fk,
  title     = {A General Framework for Inference-time Scaling and Steering of Diffusion Models},
  author    = {Singhal, Raghav and Horvitz, Zachary and Teehan, Ryan and Ren, Mengye and Yu, Zhou and
               McKeown, Kathleen and Ranganath, Rajesh},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2025},
  note      = {arXiv:2501.06848}
}
```

## License

MIT, see `LICENSE`.
