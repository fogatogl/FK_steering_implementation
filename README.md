# Four particles, one image: FK Steering reproduced on Stable Diffusion

Giulio Fogato, ENSAE Paris.

A reimplementation from the equations of FK Steering ([Singhal et al., 2025](https://arxiv.org/abs/2501.06848)),
a particle filter that steers a diffusion model toward a reward at inference time, and a rerun of its
Stable Diffusion v1.5 experiment. The filter is plain PyTorch in `smc/`; every number and figure is
rebuilt from the run records committed in `results/` and `collapse_lab/out/`.

The write-up is the blog post [`docs/paper.md`](docs/paper.md); its appendix A.7 gives, section by
section, the detail behind the numbers of the main text.

![Four finals of one prompt under the free sampler and FK](figures/f5_root_grid.png)

*One prompt, the same four initial noises: the free sampler (top) and FK at the paper's setting
(bottom), each image framed in the colour of the noise it descends from. The free sampler returns four
images from four roots, FK four near-copies of one.*

## Results

SD v1.5 with ImageReward as the reward, at the paper's setting (λ = 10, k = 4, MAX potential, five
scheduled steps), on the 100 ImageReward benchmark prompts × 3 seeds, at an equal number of UNet
evaluations:

| | ImageReward, best of k | HPS v2.1, best of k | paper (IR / HPS) |
|---|---|---|---|
| one sample | 0.237 | 0.245 | 0.187 / 0.245 |
| best-of-4 | 0.758 | 0.266 | 0.737 / 0.265 |
| FK, k = 4 | 0.820 | 0.265 | 0.898 / 0.263 |

- **Best-of-4 reproduces, FK's gain over it is smaller:** +0.062 ± 0.026 on 100 prompts paired on
  their initial noises, against +0.161 in the paper (post, section 3).
- **The four particles become one.** In 93 to 96 runs of 100 the four final images descend from a
  single initial noise. The first resampling, made on a blurred estimate of the image, already leaves
  one in about half the runs, and the later ones finish the job.
  Replaying the recorded weights through the resampler recovers the number of surviving lineages
  within 0.09 on the eighteen steered arms recorded (sections 4 and 5).
- **Keeping the lineages changes the target.** Of eight corrections, only a floor at 0 with λ = 2
  keeps three roots of four; its best image stays at best-of-4's level and the mean of its four
  images costs 0.250 ± 0.045 against FK (section 6).
- **The released code does not close the gap.** Under the paper's configuration, on the same 100
  prompts and noises, it gains -0.003 ± 0.025 over best-of-4 on three seeds, and +0.074 ± 0.023 at
  its commit from before a fix of the MAX potential, both under the paper's +0.161 (section 7).

## Repository

```
smc/            the filter: weights, resamplers, the three potentials, fk_steer, the model wrappers
tests/          one mathematical property per test: the product constraint, λ = 0 is the free
                model, the resamplers against the `particles` library
scripts/        the SD runs and their launchers, the figures, the post's numbers and checks
collapse_lab/   the study of the collapse: the probe, its readouts, the lineage replay, the
                driver of the released code
results/        one JSON record per run; post_numbers/ holds what each analysis script prints
figures/        the post's eight figures (f1 to f8), and the earlier ones the run log cites
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
An AI assistant (Claude, Anthropic) wrote the scripts that launch the experimental runs, the
figure scripts and the documentation, and drafted the post, which I reread and corrected; it wrote
none of the code of the models themselves.

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

The code is under the MIT license (`LICENSE`). The text of the post and the documentation (`docs/`) and
the figures (`figures/`) are under CC BY 4.0 (`LICENSE-CC-BY-4.0`).
